# Phase 2 P4: would the newer-library compounds (not in train) make a second IN-POOL practice set? Share of them whose
# structure is in E006's candidate pool (casmi26-v2-pool pool_meta 'key', matched by InChIKey-14 and by the
# tautomer-canonical score key, like sim2's in_pool), by instrument, library and ion mode. CPU; internet only for RDKit.
import glob, json, subprocess, sys
import pandas as pd
try:
    from rdkit import Chem
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'], check=False)
    from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
TE = rdMolStandardize.TautomerEnumerator()
find = lambda p: sorted(glob.glob(f'/kaggle/input/**/{p}', recursive=True), key=len)[0]
u = pd.read_parquet(find('newer_lib_unseen.parquet'),
                    columns=['source', 'library', 'inchikey14', 'adduct', 'inst_class', 'n_peaks', 'ionization_mode', 'smiles'])
STD = {'[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-'}
u = u[u.adduct.isin(STD) & (u.n_peaks >= 3)]
PK = {str(k)[:14] for k in pd.read_parquet(find('pool_meta.parquet'), columns=['key']).key if k}


def skey(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToInchiKey(TE.Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


c = u.groupby('inchikey14').agg(inst=('inst_class', lambda s: 'QTOF' if (s == 'QTOF').any() else s.iloc[0]),
                                library=('library', 'first'), smiles=('smiles', 'first'),
                                modes=('ionization_mode', lambda s: ';'.join(sorted(set(map(str, s)))))).reset_index()
c['in_pool'] = c.inchikey14.isin(PK) | c.smiles.map(skey).isin(PK)
top = lambda g: {str(k): [int(v.size), round(float(v.mean()), 3)] for k, v in g.in_pool.groupby(g.library) if v.size >= 10}
rep = dict(pool_keys=len(PK), compounds=len(c), in_pool=round(float(c.in_pool.mean()), 3),
           by_instrument={k: [int(v.size), round(float(v.mean()), 3)] for k, v in c.in_pool.groupby(c.inst)},
           by_mode={k: [int(v.size), round(float(v.mean()), 3)] for k, v in c.in_pool.groupby(c.modes)},
           qtof_by_library=top(c[c.inst == 'QTOF']))
json.dump(rep, open('/kaggle/working/newlib_pool_check.json', 'w'), indent=1)
print('NEWLIB POOL CHECK ' + json.dumps(rep))
