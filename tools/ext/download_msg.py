# Download MassSpecGym (MIT, https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) and convert it to the
# competition's spectrum format. Needs internet ON. Output: /kaggle/working/msg/msg_spectra.parquet
import os, sys, subprocess, json
import numpy as np, pandas as pd
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=False)
from huggingface_hub import HfApi, hf_hub_download

REPO = 'roman-bushuiev/MassSpecGym'
OUT = '/kaggle/working/msg'
os.makedirs(OUT, exist_ok=True)
api = HfApi()
files = api.list_repo_files(REPO, repo_type='dataset')
info = api.dataset_info(REPO)
print('repo revision', info.sha, '| files:', [f for f in files if f.startswith('data/')][:40])
tsv = [f for f in files if f.lower().endswith('.tsv') and 'massspecgym' in f.lower() and 'auxiliary' not in f.lower()]
assert tsv, 'MassSpecGym main .tsv not found; file list printed above'
path = hf_hub_download(REPO, sorted(tsv, key=len)[0], repo_type='dataset')
df = pd.read_csv(path, sep='\t')
print('loaded', sorted(tsv, key=len)[0], df.shape, '\ncolumns:', list(df.columns))
print(df.head(3).T)


def col(*names):
    for n in names:
        for c in df.columns:
            if c.lower() == n:
                return c
    raise KeyError(f'none of {names} in {list(df.columns)}')


c_mz, c_it = col('mzs', 'mz', 'ms2_mzs'), col('intensities', 'intensity', 'ms2_intensities')
c_smi, c_ik = col('smiles'), col('inchikey', 'inchikey14')
c_prec, c_add = col('precursor_mz'), col('adduct')
c_inst = next((c for c in df.columns if c.lower() == 'instrument_type'), None)
c_ce = next((c for c in df.columns if c.lower() == 'collision_energy'), None)
c_form = next((c for c in df.columns if c.lower() == 'formula'), None)
c_id = next((c for c in df.columns if c.lower() == 'identifier'), None)


def floats(s):
    if isinstance(s, str):
        return [float(x) for x in s.replace('[', '').replace(']', '').split(',') if x.strip()]
    return [float(x) for x in s]


mz = [floats(s) for s in df[c_mz]]
it = [floats(s) for s in df[c_it]]
it = [list(np.asarray(v) / max(v)) if len(v) and max(v) > 0 else v for v in it]
out = pd.DataFrame(dict(
    spectrum_id=(df[c_id].astype(str) if c_id else pd.Series(np.arange(len(df))).astype(str)),
    inchikey14=df[c_ik].astype(str).str.strip().str.upper().str[:14],
    normalized_smiles=df[c_smi].astype(str),
    molecular_formula=(df[c_form].astype(str) if c_form else None),
    precursor_mz=pd.to_numeric(df[c_prec], errors='coerce'),
    adduct=df[c_add].astype(str),
    ionization_mode=np.where(df[c_add].astype(str).str.strip().str.endswith('-'), 'negative', 'positive'),
    instrument_type=(df[c_inst].astype(str) if c_inst else 'unknown'),
    collision_energy_ev=(pd.to_numeric(df[c_ce], errors='coerce') if c_ce else np.nan),
    ms2_mzs=mz, ms2_normalized_intensities=it, source='MassSpecGym'))
out = out[out.precursor_mz.notna() & (out.ms2_mzs.map(len) > 0)].reset_index(drop=True)
out.to_parquet(os.path.join(OUT, 'msg_spectra.parquet'), index=False)
json.dump(dict(repo=REPO, revision=info.sha, file=sorted(tsv, key=len)[0], n_spectra=len(out),
               n_compounds=int(out.inchikey14.nunique()), licence='MIT'), open(os.path.join(OUT, 'SOURCE.json'), 'w'), indent=1)
print(f'saved {len(out):,} spectra / {out.inchikey14.nunique():,} compounds')
print(out.adduct.value_counts().head(), out.instrument_type.value_counts().head(), sep='\n')
