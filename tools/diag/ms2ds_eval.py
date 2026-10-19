# Plan step 9, offline check: does MS2DeepScore 2.x analog evidence improve E006's final lists?
# For every practice question (Enveda-180 shards 0/1, MassSpecGym), embed its spectra with the dual-mode model, find the
# most similar train-library spectra in EITHER ion mode, and score each candidate of E006's saved final top 25 by how
# similar it is to those analogs (Tanimoto, weighted by MS2DeepScore). Then fuse that order into E006's at a few weights
# and report the paired MRR change (official key), split by ion mode. No candidate is added: only the saved lists move.
# Inputs: competition data, notebook outputs casmi26-ms2ds-package / casmi26-sim2-e180-gl-ice05 / casmi26-sim2-e180s1-gl-frag /
# casmi26-sim2-msg-e006-gl-frag / casmi26-enveda180-questions, dataset casmi26-massspecgym.
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def find(pattern, must=''):
    hits = sorted((h for h in glob.glob(f'/kaggle/input/**/{pattern}', recursive=True) if must in h), key=len)
    assert hits, f'{pattern} ({must}) not found'
    return hits[0]


# ---- MS2DeepScore, installed exactly as an internet-off notebook would ----
PKG = os.path.dirname(find('manifest.json', 'ms2ds'))
MAN = json.load(open(os.path.join(PKG, 'manifest.json')))
TGT = '/tmp/ms2ds_site'
whl = sorted(glob.glob(os.path.join(PKG, 'wheels', '*')))
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', TGT] + whl,
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
sys.path.insert(0, TGT)
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from matchms import Spectrum
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
MODEL = MS2DeepScore(load_model(os.path.join(PKG, 'model', 'ms2deepscore_model.pt')))
log('ms2deepscore loaded offline; package check:', MAN.get('check'), '| cuda', torch.cuda.is_available())


def spec(mz, it, prec, mode, cap=True):
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    if cap and np.isfinite(prec):                     # hidden test: no peaks above precursor + 2 Da
        k = mz <= prec + 2.0; mz, it = mz[k], it[k]
    o = np.argsort(mz); mz, it = mz[o], it[o]
    if len(mz) == 0 or it.max() <= 0:
        return None
    return Spectrum(mz=mz, intensities=it / it.max(), metadata={'precursor_mz': float(prec), 'ionmode': str(mode).lower()})


def embed(specs, bs=4096):
    out = []
    for i in range(0, len(specs), bs):
        e = np.asarray(MODEL.get_embedding_array(specs[i:i + bs]), np.float32)
        out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
        if i // bs % 20 == 0:
            log(f'  embedded {i + len(e):,} / {len(specs):,}')
    return np.concatenate(out)


# ---- library: <= 2 spectra per (compound, ion mode) from the competition train set ----
TR = find('train.parquet')
meta = pq.read_table(TR, columns=['inchikey14', 'normalized_smiles', 'ionization_mode', 'precursor_mz']).to_pandas()
meta['row'] = np.arange(len(meta))
meta = meta.dropna(subset=['inchikey14', 'normalized_smiles'])
pick = meta.sample(frac=1.0, random_state=0).groupby(['inchikey14', 'ionization_mode']).head(2).sort_values('row')
log(f'library: {len(meta):,} spectra -> {len(pick):,} picked ({pick.inchikey14.nunique():,} compounds; '
    f'{pick.ionization_mode.value_counts().to_dict()})')
pf = pq.ParquetFile(TR)
want = np.zeros(pf.metadata.num_rows, bool); want[pick.row.values] = True
pos_of = np.full(pf.metadata.num_rows, -1, np.int64); pos_of[pick.row.values] = np.arange(len(pick))
LMZ, LIT = [None] * len(pick), [None] * len(pick)
off = 0
for g in range(pf.num_row_groups):
    t = pf.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities'])
    loc = np.where(want[off:off + t.num_rows])[0]
    if len(loc):
        mzc, itc = t.column(0).combine_chunks(), t.column(1).combine_chunks()
        o = mzc.offsets.to_numpy(); mv = mzc.values.to_numpy(zero_copy_only=False); iv = itc.values.to_numpy(zero_copy_only=False)
        for j in loc:
            LMZ[pos_of[off + j]], LIT[pos_of[off + j]] = mv[o[j]:o[j + 1]], iv[o[j]:o[j + 1]]
    off += t.num_rows
lspecs = [spec(m, i, p, md, cap=False) for m, i, p, md in zip(LMZ, LIT, pick.precursor_mz.values, pick.ionization_mode.values)]
okl = np.array([s is not None for s in lspecs])
pick = pick[okl].reset_index(drop=True); lspecs = [s for s in lspecs if s is not None]
del LMZ, LIT
_cuda = torch.cuda.is_available()
LE = torch.tensor(embed(lspecs), dtype=torch.float16 if _cuda else torch.float32, device='cuda' if _cuda else 'cpu')
LIK = pick.inchikey14.values
del lspecs
log(f'library embeddings {tuple(LE.shape)}')
SMI_OF = meta.groupby('inchikey14').normalized_smiles.first()

FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fp = {}


def fp(s):
    if s not in _fp:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        _fp[s] = FPG.GetFingerprint(m) if m is not None else None
    return _fp[s]


TE = rdMolStandardize.TautomerEnumerator()
_ok = {}


def okey(s):
    if s not in _ok:
        try:
            m = Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); _ok[s] = Chem.MolToInchiKey(TE.Canonicalize(m))[:14]
        except Exception:
            _ok[s] = None
    return _ok[s]


# ---- practice runs ----
E180 = pd.read_parquet(find('enveda180_questions.parquet'))
MSG = pd.read_parquet(find('msg_spectra.parquet'))
RUNS = {'e180_s0': ('casmi26-sim2-e180-gl-ice05', E180), 'e180_s1': ('casmi26-sim2-e180s1-gl-frag', E180),
        'msg': ('casmi26-sim2-msg-e006-gl-frag', MSG)}
K_ANALOG, MAX_Q_SPEC = 20, 8
results = {}
for name, (slug, src) in RUNS.items():
    so = os.path.dirname(find('queries.parquet', slug))
    q = pd.read_parquet(os.path.join(so, 'queries.parquet'))
    sub = pd.read_csv(os.path.join(so, 'sub_final_A.csv'), dtype=str)          # arm A = unchanged E006
    lists = {str(m): str(s).split(';')[:25] for m, s in zip(sub.molecule_id, sub.smiles)}
    by_ik = {k: g for k, g in src[src.inchikey14.astype(str).str[:14].isin(set(q.inchikey14))].groupby(src.inchikey14.astype(str).str[:14])}
    rows = []
    for r in q.itertuples():
        g = by_ik.get(r.inchikey14)
        if g is None:
            continue
        sp = [spec(a, b, c, d) for a, b, c, d in zip(g.ms2_mzs, g.ms2_normalized_intensities, g.precursor_mz, g.ionization_mode)]
        sp = [s for s in sp if s is not None][:MAX_Q_SPEC]
        if not sp:
            continue
        qe = torch.tensor(embed(sp), dtype=LE.dtype, device=LE.device)
        sims = (qe @ LE.T).float().max(0).values                                # best over the molecule's spectra
        top = torch.topk(sims, K_ANALOG * 3).indices.cpu().numpy()
        an, seen = [], set()
        for i in top:                                                              # one entry per analog compound
            if LIK[i] not in seen:
                seen.add(LIK[i]); an.append((LIK[i], float(sims[i])))
            if len(an) == K_ANALOG:
                break
        afp = [(fp(SMI_OF[k]), s) for k, s in an if fp(SMI_OF[k]) is not None]
        cands = [c for c in lists.get(str(r.molecule_id), []) if c and c != 'CCO']
        t = okey(r.smiles)
        trank = next((i for i, c in enumerate(cands, 1) if okey(c) == t), None)
        sc_max, sc_soft = [], []
        w = np.array([s for _, s in afp]); w = np.exp((w - w.max()) / 0.05) if len(w) else w
        for c in cands:
            f = fp(c)
            tan = np.array(DataStructs.BulkTanimotoSimilarity(f, [a for a, _ in afp])) if f is not None and afp else np.zeros(0)
            sc_max.append(float((tan * np.array([s for _, s in afp])).max()) if len(tan) else 0.0)
            sc_soft.append(float((tan * w).sum() / w.sum()) if len(tan) else 0.0)
        tf = fp(r.smiles)
        rows.append(dict(mid=str(r.molecule_id), mode=str(getattr(r, 'ion_mode', '')), trank=trank, cands=cands, sc_max=sc_max,
                         sc_soft=sc_soft, top_sim=an[0][1] if an else np.nan,
                         truth_tan_top1=float(DataStructs.TanimotoSimilarity(tf, afp[0][0])) if tf is not None and afp else np.nan,
                         truth_tan_best=float(max(DataStructs.BulkTanimotoSimilarity(tf, [a for a, _ in afp]))) if tf is not None and afp else np.nan))
    results[name] = rows
    log(f'{name}: {len(rows)} questions scored')


def mrr_after(rows, key, lam, only=None):
    out = []
    for r in rows:
        n = len(r['cands'])
        if not n or r['trank'] is None:
            out.append(1.0 / r['trank'] if r['trank'] else 0.0); continue
        if only is not None and only(r) is False:
            out.append(1.0 / r['trank']); continue
        s = np.array(r[key]); o = np.argsort(-s, kind='stable'); ar = np.empty(n); ar[o] = np.arange(1, n + 1)
        f = 1.0 / (3 + np.arange(1, n + 1)) + lam / (3 + ar)               # RRF of E006's order and the analog order
        new = np.argsort(-f, kind='stable')
        out.append(1.0 / (int(np.where(new == r['trank'] - 1)[0][0]) + 1))
    return np.array(out)


rep = {}
for name, rows in results.items():
    base = mrr_after(rows, 'sc_max', 0.0)
    neg = lambda r: 'negative' in r['mode']
    rep[name] = dict(n=len(rows), e006=round(base.mean(), 4), n_negative=int(sum(neg(r) for r in rows)),
                     truth_tan_top_analog=round(float(np.nanmean([r['truth_tan_top1'] for r in rows])), 3),
                     truth_tan_best_of_20=round(float(np.nanmean([r['truth_tan_best'] for r in rows])), 3))
    for key in ('sc_max', 'sc_soft'):
        for lam in (0.25, 0.5, 1.0):
            for tag, only in (('all', None), ('neg_only', neg)):
                d = mrr_after(rows, key, lam, only) - base
                bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(1000)]
                rep[name][f'{key}_lam{lam}_{tag}'] = dict(delta=round(float(d.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4),
                                                         round(float(np.percentile(bs, 97.5)), 4)], better=int((d > 0).sum()), worse=int((d < 0).sum()))
json.dump(rep, open('/kaggle/working/ms2ds_eval.json', 'w'), indent=1)
print('MS2DS EVAL ' + json.dumps(rep))
