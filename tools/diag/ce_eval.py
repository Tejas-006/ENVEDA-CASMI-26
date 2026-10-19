# Multi-energy fix evaluation on timsTOF natural products (GPU, ~1 h).
# enveda-np-examples (train) = 250 natural products measured and processed like the hidden test; ~35% of its spectra
# are merged from several collision energies. For each molecule: candidates = the true structure + up to N_ISO
# same-formula isomers from the candidate pool. ICEBERG and GLACIER score them twice -- runners as shipped (merged
# spectrum vs one prediction at the mean energy) and with tools/lb/multi_ce.py (vs merged predictions at each listed
# energy) -- and we compare the truth's rank among its isomers. Both variants see identical molecules and candidates,
# so memorisation (both models were trained on public spectra) affects them equally. Prints aggregates only.
N_ISO, SEED, BUDGET = 19, 0, 3000
# library -> molecules sampled (preferring ones with a merged positive [M+H]+/[M+Na]+ spectrum); 0 = all
LIBS = {'enveda-np-examples': 0, 'enveda-180': 400}
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


T0 = time.time()
ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
sys.path[:0] = [ICE_PKG, GL_PKG]
import fuse as ice_fuse, gl_fuse

cols = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz',
        'adduct', 'ionization_mode', 'instrument_type', 'collision_energy_ev', 'collision_energy_orig',
        'collision_energy_orig_units']
# pass 1: metadata only (enveda-180 has 1.15M spectra) -> pick molecules; pass 2: full rows for those molecules
meta = pq.read_table(find('train.parquet'), columns=['inchikey14', 'ingest_lib', 'collision_energy_ev', 'adduct',
                                                     'ionization_mode'],
                     filters=[('ingest_lib', 'in', list(LIBS))]).to_pandas()
_mc = meta.collision_energy_ev.map(lambda v: len(set(v)) if v is not None and hasattr(v, '__len__') else 0) > 1
_cov = meta.ionization_mode.eq('positive') & meta.adduct.isin(['[M+H]+', '[M+Na]+'])
pick_by_lib = {}
for lib, n in LIBS.items():
    g = meta[meta.ingest_lib == lib]
    iks = g.inchikey14.unique()
    if n and len(iks) > n:
        merged = set(g.inchikey14[(_mc & _cov)[g.index]])
        order = np.random.default_rng(SEED).permutation(len(iks))
        order = sorted(order, key=lambda i: iks[i] not in merged)   # stable: random order kept within each group
        iks = [iks[i] for i in order[:n]]
    pick_by_lib[lib] = set(iks)
del meta
tr = pq.read_table(find('train.parquet'), columns=cols + ['ingest_lib'],
                   filters=[('ingest_lib', 'in', list(LIBS)),
                            ('inchikey14', 'in', sorted(set().union(*pick_by_lib.values())))]).to_pandas()
tr = tr[[ik in pick_by_lib[lib] for ik, lib in zip(tr.inchikey14, tr.ingest_lib)]].copy()
tr['molecule_id'] = tr.ingest_lib.str.replace('enveda-', 'e') + ':' + tr.inchikey14
LIB_OF = dict(zip(tr.molecule_id, tr.ingest_lib))
pool = pd.read_parquet(find('pool_meta.parquet'), columns=['key', 'smiles', 'formula'])
by_formula = {f: g for f, g in pool.groupby('formula')}
rng = np.random.default_rng(SEED)
cands, truth_smi = {}, {}
for mid, g in tr.groupby('molecule_id'):
    ik, f, s = g.inchikey14.iloc[0], g.molecular_formula.iloc[0], g.normalized_smiles.iloc[0]
    iso = by_formula.get(f)
    if iso is None:
        continue
    iso = iso[iso.key != ik].drop_duplicates('key')
    if len(iso) == 0:
        continue
    pick = iso.smiles.values[rng.permutation(len(iso))[:N_ISO]]
    cands[mid] = [s] + [str(x) for x in pick]; truth_smi[mid] = s
te = tr[tr.molecule_id.isin(cands)]
n_ce = te.collision_energy_ev.map(lambda v: len(set(v)) if v is not None and hasattr(v, '__len__') else 0)
covered = te.ionization_mode.eq('positive') & te.adduct.isin(['[M+H]+', '[M+Na]+'])
merged_mols = set(te.molecule_id[(n_ce > 1) & covered])
print(f'molecules with isomers: {len(cands)}; with a merged covered spectrum: {len(merged_mols)}; '
      f'isomers per molecule median {int(np.median([len(c) - 1 for c in cands.values()]))}', flush=True)
items = ice_fuse.build_ice_input(te, cands)

# multi-energy wrapper: tools/lb/multi_ce.py runs as the cell before this one (same file the submission uses)
assert os.path.exists(MCE_PYTHON), 'multi_ce.py wrapper not set up'

out = {}
for name, py in (('shipped', None), ('multi_ce', MCE_PYTHON)):
    ice = ice_fuse.run_ice(ICE_PKG, items, workdir=f'/kaggle/working/ice_{name}', device='cuda', budget_s=BUDGET,
                           site='/kaggle/working/ice_site', python=py)
    gl = gl_fuse.run_gl(GL_PKG, items, workdir=f'/kaggle/working/gl_{name}', device='cuda', budget_s=BUDGET,
                        site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'), python=py)
    out[name] = dict(ice=ice, gl=gl)
    print(f'{name}: ICE mols {len(ice)}, GL mols {len(gl)}, {time.time() - T0:.0f}s', flush=True)


def truth_rank(scores, mid):
    sc = {s: v for s, v in (scores.get(mid) or {}).items() if v is not None}
    t = truth_smi[mid]
    if t not in sc or len(sc) < 2:
        return None
    better = sum(v > sc[t] for v in sc.values()); ties = sum(v == sc[t] for v in sc.values()) - 1
    return 1 + better + ties / 2


report = {}
for model in ('ice', 'gl'):
    subsets = [('merged-spectrum molecules', merged_mols), ('all molecules', set(cands))]
    subsets += [(f'{lib} merged', {m for m in merged_mols if LIB_OF.get(m) == lib}) for lib in LIBS]
    for subset, mids in subsets:
        r = {}
        for name in out:
            rk = pd.Series({m: truth_rank(out[name][model], m) for m in mids}).dropna()
            r[name] = rk
        common = r['shipped'].index.intersection(r['multi_ce'].index)
        a, b = 1 / r['shipped'][common], 1 / r['multi_ce'][common]
        d = (b - a).values
        bs = np.array([d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(2000)]) if len(d) else np.array([0.0])
        report[f'{model} | {subset}'] = dict(n=int(len(common)), mrr_shipped=round(float(a.mean()), 4) if len(d) else None,
                                            mrr_multi_ce=round(float(b.mean()), 4) if len(d) else None,
                                            top1_shipped=round(float((a == 1).mean()), 4) if len(d) else None,
                                            top1_multi_ce=round(float((b == 1).mean()), 4) if len(d) else None,
                                            delta=round(float(d.mean()), 4) if len(d) else None,
                                            ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                            better=int((d > 0).sum()), worse=int((d < 0).sum()))
for k, v in report.items():
    print(f'{k:40s} {v}')
json.dump(report, open('/kaggle/working/ce_eval.json', 'w'), indent=1)
# raw scores for offline analysis (simulator weighting, aggregation): public-model scores of public/train structures
json.dump(dict(truth=truth_smi, lib=LIB_OF, merged=sorted(merged_mols), cands=cands,
               scores={n: {m: v for m, v in d.items()} for n, d in out.items()}),
          open('/kaggle/working/ce_eval_scores.json', 'w'))
print('CE EVAL ' + json.dumps(report), f'{time.time() - T0:.0f}s')
