# Precursor mass-error check (CPU, ~10 min): how far is each train spectrum's precursor m/z from its structure's
# exact mass, by instrument? Decides the candidate mass window for CMatch dense retrieval: the pipeline searches
# +-10 ppm (30 ppm fallback), and MassSpecGym practice questions are often 10-160 ppm off, so sim2 may overstate what
# a wider window gains on the (timsTOF) hidden test.
SAMPLE_ROWS, SEED = 300_000, 0
import glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')

PROTON = 1.007276
ADDUCT_SHIFT = {'[M+H]+': PROTON, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
                '[M-H]-': -PROTON, '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M+FA-H]-': 44.998201,
                '[M+HCOO]-': 44.998201, '[M+H-H2O]+': PROTON - 18.010565, '[M-H2O+H]+': PROTON - 18.010565}


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def inst(x):
    return str(x).lower().replace('-', '').replace(' ', '') if x is not None else 'unk'


TR = find('train.parquet')
cols = ['normalized_smiles', 'precursor_mz', 'adduct', 'instrument_type']
n = pq.ParquetFile(TR).metadata.num_rows
idx = np.sort(np.random.default_rng(SEED).choice(n, min(SAMPLE_ROWS, n), replace=False))
tr = pq.read_table(TR, columns=cols).take(idx).to_pandas()
tr['inst'] = tr.instrument_type.map(inst)
tr = tr[tr.adduct.isin(ADDUCT_SHIFT)].copy()
mass = {}
for s in tr.normalized_smiles.unique():
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    mass[s] = ExactMolWt(m) if m is not None else np.nan
tr['exact'] = tr.normalized_smiles.map(mass)
tr['neutral'] = tr.precursor_mz.astype(float) - tr.adduct.map(ADDUCT_SHIFT)
tr['ppm'] = (tr.neutral - tr.exact) / tr.exact * 1e6
tr = tr[np.isfinite(tr.ppm)]
report = {'rows': int(len(tr))}


def stats(a):
    a = np.abs(np.asarray(a, float))
    return dict(n=int(len(a)), p50=round(float(np.percentile(a, 50)), 2), p90=round(float(np.percentile(a, 90)), 2),
                p95=round(float(np.percentile(a, 95)), 2), p99=round(float(np.percentile(a, 99)), 2),
                over10=round(float((a > 10).mean()), 4), over30=round(float((a > 30).mean()), 4),
                over50=round(float((a > 50).mean()), 4))


report['all'] = stats(tr.ppm)
top = tr.inst.value_counts().head(8).index
report['by_instrument'] = {i: stats(tr.ppm[tr.inst == i]) for i in top}
report['timstof_by_adduct'] = {a: stats(g.ppm) for a, g in tr[tr.inst == 'timstof'].groupby('adduct') if len(g) >= 200}
for k, v in [('all', report['all'])] + list(report['by_instrument'].items()):
    print(f'{k:22s} {v}')
print('\ntimsTOF by adduct:')
for k, v in report['timstof_by_adduct'].items():
    print(f'{k:22s} {v}')
json.dump(report, open('/kaggle/working/mass_error_check.json', 'w'), indent=1)
print('MASS ERROR CHECK ' + json.dumps(report))
