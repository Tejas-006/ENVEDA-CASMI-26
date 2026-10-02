# sim2 setup (replaces the baseline's smoke switch): a pseudo-competition folder built from held-out TRAINING compounds.
# Queries are natural products (compounds also in COCONUT/ChEBI/LIPID MAPS), so the true structure stays in the
# candidate pool after its spectra are removed from the library. timsTOF spectra are preferred (like the hidden test).
#   class 1 = query spectra from one instrument; the compound's other spectra stay in the library
#   class 2 = every spectrum of the compound is removed from the library
SHARD, SHARD_SIZE = 0, 500
CLASS1_FRAC, MAX_SPEC, SEED, SIM_TOPK, FILTER_TRAIN_STRUCTS = 0.3, 10, 2026, 300, True
import zlib, pickle, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()


def metric_key(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToInchiKey(_TE.Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def scaffold_of(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
    except Exception:
        return ''


def _inst(x):
    return str(x).lower().replace('-', '').replace(' ', '') if x is not None else 'unk'


IS_RERUN = True
REAL_COMP = COMP
SIM_OUT = '/kaggle/working/sim_out'; os.makedirs(SIM_OUT, exist_ok=True)
SM = '/kaggle/working/sim_comp'; os.makedirs(SM, exist_ok=True)
TEST_SCHEMA = pq.read_schema(os.path.join(REAL_COMP, 'test.parquet'))
TR_PATH = os.path.join(REAL_COMP, 'train.parquet')

meta = pq.read_table(TR_PATH, columns=['inchikey14', 'normalized_smiles', 'instrument_type']).to_pandas()
meta['row'] = np.arange(len(meta))
meta['inst'] = [_inst(x) for x in meta.instrument_type]
meta = meta.dropna(subset=['inchikey14', 'normalized_smiles'])

NP_KEYS = set()
for _f in ['coco_meta.pkl', 'bio_meta.pkl']:
    try:
        NP_KEYS |= set(pickle.load(open(find(_f), 'rb'))['keys'])
    except Exception as e:
        print('natural-product key set: could not load', _f, repr(e))
comp = meta.groupby('inchikey14').agg(smiles=('normalized_smiles', 'first'), n=('row', 'size'),
                                      n_inst=('inst', 'nunique')).reset_index()
comp['has_tims'] = comp.inchikey14.isin(set(meta.inchikey14[meta.inst == 'timstof']))
elig = comp[comp.inchikey14.isin(NP_KEYS)] if NP_KEYS else comp
print(f'train: {len(meta):,} spectra / {len(comp):,} compounds; eligible natural products: {len(elig):,}'
      + ('' if NP_KEYS else '  (NO natural-product set found -> all compounds eligible)'))

rng = np.random.default_rng(SEED)
w = 1.0 + 2.0 * elig.has_tims.values
order = np.argsort(-(np.log(rng.random(len(elig))) / w), kind='stable')
pick = elig.iloc[order[SHARD * SHARD_SIZE:(SHARD + 1) * SHARD_SIZE]].reset_index(drop=True)
assert len(pick), f'shard {SHARD} is past the end of the eligible compounds'

by_ik = {k: g for k, g in meta[meta.inchikey14.isin(set(pick.inchikey14))].groupby('inchikey14')}
int_ids = pa.types.is_integer(TEST_SCHEMA.field('molecule_id').type)
q_rows, drop_rows, qmeta = [], set(), []
for qi, r in pick.iterrows():
    g = by_ik[r.inchikey14]
    insts = g.inst.value_counts()
    qinst = 'timstof' if 'timstof' in insts.index else insts.index[0]
    cls = 1 if (r.n_inst >= 2 and (zlib.crc32(r.inchikey14.encode()) % 1000) / 1000 < CLASS1_FRAC) else 2
    rows = g.row[g.inst == qinst].values
    if len(rows) > MAX_SPEC:
        rows = np.sort(np.random.default_rng(SEED + qi).choice(rows, MAX_SPEC, replace=False))
    mid = 910_000_000 + SHARD * 100_000 + qi if int_ids else f'sim{SHARD}_{qi:05d}'
    q_rows += [(int(x), mid) for x in rows]
    drop_rows |= set(int(x) for x in (rows if cls == 1 else g.row.values))
    qmeta.append(dict(molecule_id=mid, inchikey14=r.inchikey14, smiles=r.smiles, truth_key=metric_key(r.smiles),
                      sim_class=cls, scaffold=scaffold_of(r.smiles), query_inst=qinst, n_spec=len(rows),
                      shard=SHARD))
QUERIES = pd.DataFrame(qmeta)
TRUTH = {m: {k for k in (t, ik) if k} for m, t, ik in zip(QUERIES.molecule_id, QUERIES.truth_key, QUERIES.inchikey14)}
TRUTH_KEY = dict(zip(QUERIES.molecule_id, QUERIES.truth_key))
CLASS2_IKS = set(QUERIES.inchikey14[QUERIES.sim_class == 2])

_tbl = pq.read_table(TR_PATH)
keep = np.ones(_tbl.num_rows, bool); keep[list(drop_rows)] = False
pq.write_table(_tbl.filter(pa.array(keep)), os.path.join(SM, 'train.parquet'), row_group_size=50_000)
_qt = _tbl.take(pa.array([x for x, _ in q_rows])).to_pandas()
del _tbl
_qt['molecule_id'] = [m for _, m in q_rows]
if 'spectrum_id' in TEST_SCHEMA.names:
    _qt['spectrum_id'] = [f'sim{SHARD}_{i}' for i in range(len(_qt))] if not int_ids else \
        910_000_000 + SHARD * 100_000 * MAX_SPEC + np.arange(len(_qt))
for c in TEST_SCHEMA.names:
    if c not in _qt.columns:
        print('WARNING test column missing from train, filled with None:', c); _qt[c] = None
_qt[TEST_SCHEMA.names].to_parquet(os.path.join(SM, 'test.parquet'), index=False)
_sc = pd.read_csv(os.path.join(REAL_COMP, 'sample_submission.csv'), nrows=1).columns
pd.DataFrame({c: (QUERIES.molecule_id.values if c == 'molecule_id' else 'CCO') for c in _sc}).to_csv(
    os.path.join(SM, 'sample_submission.csv'), index=False)
QUERIES.to_parquet(os.path.join(SIM_OUT, 'queries.parquet'), index=False)

# train_structs / train_fp_sel are row-aligned train-structure tables used by the v1 engine: drop class-2 compounds
os.makedirs('work', exist_ok=True)
STRUCT_FILTER = 'off'
if FILTER_TRAIN_STRUCTS:
    _ts = pd.read_parquet(os.path.join(POOL_DIR, 'train_structs.parquet'))
    _tf = np.load(os.path.join(POOL_DIR, 'train_fp_sel.npy'), mmap_mode='r')
    _kc = next((c for c in ['inchikey14', 'ik14', 'ik', 'key', 'inchikey'] if c in _ts.columns), None)
    if _kc and len(_ts) == len(_tf):
        _km = ~_ts[_kc].astype(str).str[:14].isin(CLASS2_IKS).values
        _ts[_km].reset_index(drop=True).to_parquet('work/train_structs.parquet', index=False)
        np.save('work/train_fp_sel.npy', np.asarray(_tf)[_km])
        STRUCT_FILTER = f'removed {int((~_km).sum())} rows by column {_kc}'
    else:
        STRUCT_FILTER = f'NOT filtered: columns {list(_ts.columns)[:12]}, rows {len(_ts)} vs fp {len(_tf)}'
    del _ts, _tf

SIMD, SIM_NAMES = {}, None


def _simdump(mid, C, XF, fnames, score):
    global SIM_NAMES
    if SIM_NAMES is None:
        SIM_NAMES = list(fnames)
    keys = np.asarray(C.key, dtype=object)
    t = TRUTH.get(mid, set())
    pos = np.array([k in t for k in keys], bool)
    o = np.union1d(np.argsort(-score, kind='stable')[:SIM_TOPK], np.where(pos)[0])
    SIMD[mid] = dict(keys=keys[o].astype(str), smiles=np.asarray(C.smiles, dtype=object)[o].astype(str),
                     X=np.asarray(XF[o], np.float32), score=np.asarray(score[o], np.float32), y=pos[o],
                     n_all=len(keys))


COMP = SM
print(f'SIM2 shard {SHARD}: {len(QUERIES)} queries ({int((QUERIES.sim_class == 1).sum())} class-1, '
      f'{int((QUERIES.sim_class == 2).sum())} class-2), {len(q_rows)} spectra, {len(drop_rows)} library rows removed;'
      f' train_structs: {STRUCT_FILTER} | ICE_BUDGET {ICE_BUDGET}')
