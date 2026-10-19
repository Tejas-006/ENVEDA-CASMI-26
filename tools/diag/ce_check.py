# Collision-energy check (CPU, ~5 min): how often are timsTOF spectra merged from several collision energies, and
# which adducts / modes do they use? ICEBERG/GLACIER's runners predict ONE spectrum at mean(collision_energy_ev), so a
# spectrum merged from [20, 40, 60, 80] eV is compared with a single 50 eV prediction. enveda-np-examples is processed
# like the hidden test (Data tab); enveda-180 is the same instrument. Prints counts only.
import glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


cols = ['ingest_lib', 'collision_energy_ev', 'collision_energy_orig', 'adduct', 'ionization_mode']
tr = pq.read_table(find('train.parquet'), columns=cols,
                   filters=[('ingest_lib', 'in', ['enveda-np-examples', 'enveda-180'])]).to_pandas()
te = pq.read_table(find('test.parquet'), columns=cols[1:]).to_pandas()
te['ingest_lib'] = 'visible test.parquet (stand-in)'
report = {}
for lib, g in pd.concat([tr, te]).groupby('ingest_lib'):
    n_ce = g.collision_energy_ev.map(lambda v: len(v) if v is not None and hasattr(v, '__len__') else 0)
    combos = g.collision_energy_ev.map(lambda v: str([int(x) for x in v]) if v is not None and hasattr(v, '__len__') else 'none')
    covered_ice = g.ionization_mode.eq('positive') & g.adduct.isin(['[M+H]+', '[M+Na]+'])
    covered_gl = g.ionization_mode.eq('positive') & g.adduct.eq('[M+H]+')
    r = dict(spectra=int(len(g)), n_ce_counts=n_ce.value_counts().sort_index().to_dict(),
             multi_ce_share=round(float((n_ce > 1).mean()), 4), top_ce_lists=combos.value_counts().head(8).to_dict(),
             iceberg_covered_share=round(float(covered_ice.mean()), 4), glacier_covered_share=round(float(covered_gl.mean()), 4),
             negative_share=round(float(g.ionization_mode.eq('negative').mean()), 4),
             adducts=g.adduct.value_counts().head(10).to_dict())
    report[lib] = r
    print(f'\n=== {lib}'); [print(f'  {k}: {v}') for k, v in r.items()]
print('CE CHECK ' + json.dumps(report, default=str))
