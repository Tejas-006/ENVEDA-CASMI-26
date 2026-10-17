# sim2 question picker: timsTOF spectra from the public Enveda-180 release (Zenodo 21346580, CC BY 4.0; host-approved,
# discussion 745148) of compounds NOT in the competition train set (tools/ext/enveda180_questions.py -> private dataset
# casmi26-enveda180-q). Same instrument as the hidden test, both ion modes, real precursors (no precursor fix).
# Caveat: Enveda-180 is synthetic drug-like screening chemistry, not natural products.
EXT_FILE = 'enveda180_questions.parquet'
from rdkit.Chem import inchi as _rdinchi
_ext = pd.read_parquet(find(EXT_FILE))
_ext['inchikey14'] = [_k14(k) for k in _ext.inchikey14]


def _smiles_of(s, i):
    if isinstance(s, str) and s and Chem.MolFromSmiles(s) is not None:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s))
    if isinstance(i, str) and i.startswith('InChI='):
        m = _rdinchi.MolFromInchi(i)
        return Chem.MolToSmiles(m) if m is not None else None
    return None


# Keep compounds with >= 1 standard-adduct spectrum: compounds seen only as in-source fragments ([M-H2O+H]+ etc.) are
# probably why train's preparation dropped them, and the hidden test was QC'd by precursor/structure consistency.
STD_ADDUCTS = {'[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-'}
_ext = _ext[_ext.inchikey14.isin(set(_ext.inchikey14[_ext.adduct.isin(STD_ADDUCTS)]))]
_ext = _ext.reset_index(drop=True)
_ext['row'] = np.arange(len(_ext))
TRAIN_IKS = {_k14(k) for k in pq.read_table(TR_PATH, columns=['inchikey14']).column(0).to_pylist() if k is not None}
_first = _ext.groupby('inchikey14').first()
ecomp = pd.DataFrame({'inchikey14': _first.index,
                      'smiles': [_smiles_of(s, i) for s, i in zip(_first.smiles, _first.inchi)],
                      'n': _ext.groupby('inchikey14').size().reindex(_first.index).values})
n_all = len(ecomp)
ecomp = ecomp[ecomp.smiles.notna() & ~ecomp.inchikey14.isin(TRAIN_IKS)].reset_index(drop=True)
NP_FILTER = f'enveda180: {len(ecomp):,} of {n_all:,} {EXT_FILE} compounds usable (structure parsed, not in train)'
print(NP_FILTER)

rng = np.random.default_rng(SEED)
order = rng.permutation(len(ecomp))
pick = ecomp.iloc[order[SHARD * SHARD_SIZE:(SHARD + 1) * SHARD_SIZE]].reset_index(drop=True)
assert len(pick), f'shard {SHARD} is past the end of the eligible Enveda-180 compounds'

try:
    POOL_KEYS = {_k14(k) for k in pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key']).key}
except Exception as e:
    POOL_KEYS = set()
    print('pool keys not loaded (in_pool will be False):', repr(e))

by_ik = {k: g for k, g in _ext[_ext.inchikey14.isin(set(pick.inchikey14))].groupby('inchikey14')}
int_ids = pa.types.is_integer(TEST_SCHEMA.field('molecule_id').type)
q_rows, qmeta, n_taut = [], [], 0
for qi, r in pick.iterrows():
    tk = metric_key(r.smiles)
    if tk in TRAIN_IKS:                     # tautomer of a train compound: the metric would count it as seen
        n_taut += 1
        continue
    g = by_ik[r.inchikey14]
    rows = g.row.values
    if len(rows) > MAX_SPEC:
        rows = np.sort(np.random.default_rng(SEED + qi).choice(rows, MAX_SPEC, replace=False))
    mid = 930_000_000 + SHARD * 100_000 + qi if int_ids else f'e180{SHARD}_{qi:05d}'
    q_rows += [(int(x), mid) for x in rows]
    sel = g[g.row.isin(rows)]
    fml = sel.formula.dropna()
    qmeta.append(dict(molecule_id=mid, inchikey14=r.inchikey14, smiles=r.smiles, truth_key=tk, sim_class=3,
                      scaffold=scaffold_of(r.smiles), query_inst='timstof', n_spec=len(rows),
                      adducts=';'.join(sorted(set(map(str, sel.adduct)))),
                      ion_mode=';'.join(sorted(set(map(str, sel.ionization_mode)))),
                      precursor_mz=float(sel.precursor_mz.median()), formula=str(fml.iloc[0]) if len(fml) else '',
                      n_lib_spectra=0, in_pool=bool(r.inchikey14 in POOL_KEYS or tk in POOL_KEYS), shard=SHARD))
QUERIES = pd.DataFrame(qmeta)
TRUTH = {m: {k for k in (t, ik) if k} for m, t, ik in zip(QUERIES.molecule_id, QUERIES.truth_key, QUERIES.inchikey14)}
TRUTH_KEY = dict(zip(QUERIES.molecule_id, QUERIES.truth_key))
print(f'enveda180 queries: {len(QUERIES)} (dropped {n_taut} tautomers of train compounds); truth in candidate pool: '
      f'{QUERIES.in_pool.mean():.1%}; ion modes {QUERIES.ion_mode.value_counts().to_dict()}')

_qt = _ext.iloc[[x for x, _ in q_rows]].reset_index(drop=True)
_qt['molecule_id'] = [m for _, m in q_rows]
# Peaks above precursor + 2 Da are removed from the hidden test (Data tab); do the same here.
PEAK_CAP_DA = 2.0
_n_before = int(sum(len(m) for m in _qt.ms2_mzs))
_caps = (_qt.precursor_mz.astype(float) + PEAK_CAP_DA).values
_kept = [np.asarray(m, float) <= c for m, c in zip(_qt.ms2_mzs, _caps)]
_qt['ms2_mzs'] = [list(np.asarray(m, float)[k]) for m, k in zip(_qt.ms2_mzs, _kept)]
_qt['ms2_normalized_intensities'] = [list(np.asarray(i, float)[k]) for i, k in zip(_qt.ms2_normalized_intensities, _kept)]
_qt['ms2_normalized_intensities'] = [list(np.asarray(i) / max(i)) if len(i) else i for i in _qt.ms2_normalized_intensities]
print(f'peak cap (precursor + {PEAK_CAP_DA} Da): kept {int(sum(len(m) for m in _qt.ms2_mzs)):,} of {_n_before:,} peaks; '
      f'spectra left empty: {int(sum(len(m) == 0 for m in _qt.ms2_mzs))}')
if 'spectrum_id' in TEST_SCHEMA.names:
    _qt['spectrum_id'] = [f'e180{SHARD}_{i}' for i in range(len(_qt))] if not int_ids else \
        930_000_000 + SHARD * 100_000 * MAX_SPEC + np.arange(len(_qt))
for c in TEST_SCHEMA.names:
    if c not in _qt.columns:
        print('WARNING test column missing from Enveda-180 data, filled with None:', c); _qt[c] = None
_qt[TEST_SCHEMA.names].to_parquet(os.path.join(SM, 'test.parquet'), index=False)
_sc = pd.read_csv(os.path.join(REAL_COMP, 'sample_submission.csv'), nrows=1).columns
pd.DataFrame({c: (QUERIES.molecule_id.values if c == 'molecule_id' else 'CCO') for c in _sc}).to_csv(
    os.path.join(SM, 'sample_submission.csv'), index=False)
if not os.path.exists(os.path.join(SM, 'train.parquet')):
    os.symlink(TR_PATH, os.path.join(SM, 'train.parquet'))   # library untouched
QUERIES.to_parquet(os.path.join(SIM_OUT, 'queries.parquet'), index=False)
os.makedirs('work', exist_ok=True)
STRUCT_FILTER = 'off (Enveda-180 queries are not in train)'
