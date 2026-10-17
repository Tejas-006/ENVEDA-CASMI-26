# Build timsTOF practice questions from the public Enveda-180 release (Zenodo 21346580, CC BY 4.0; the host confirmed
# it may be used: discussion 745148 / 743569). Internet ON, CPU. Keeps compounds whose InChIKey-14 is NOT in the
# competition train set (the host: "likely filtered out in the train.parquet preparation") and the hidden test's ten
# adducts, and writes /kaggle/working/enveda180_questions.parquet in the test.parquet layout plus truth columns.
# Output is derived from train membership -> save it only as a PRIVATE dataset.
URL = 'https://zenodo.org/records/21346580/files/enveda-180.parquet.gz?download=1'
TEST_ADDUCTS = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M-H2O-H]-',
                '[M+CH2O2-H]-', '[M+Cl]-']
import glob, gzip, json, os, shutil, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, requests

T0 = time.time()
GZ, PQ = '/tmp/e180/enveda-180.parquet.gz', '/tmp/e180/enveda-180.parquet'
os.makedirs('/tmp/e180', exist_ok=True)
if not os.path.exists(PQ):
    with requests.get(URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(GZ, 'wb') as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    print(f'downloaded {os.path.getsize(GZ) / 1e9:.2f} GB in {time.time() - T0:.0f}s', flush=True)
    with gzip.open(GZ, 'rb') as fi, open(PQ, 'wb') as fo:
        shutil.copyfileobj(fi, fo, 1 << 24)
    os.remove(GZ)
schema = pq.read_schema(PQ)
print('schema:', schema, flush=True)
names = schema.names


def col(*must, any_of=(), exclude=()):
    for n in names:
        l = n.lower()
        if all(m in l for m in must) and (not any_of or any(a in l for a in any_of)) and not any(e in l for e in exclude):
            return n
    return None


C = dict(ik=col('inchikey'), smiles=col('smiles'), inchi=col('inchi', exclude=('key',)),
         mzs=col('mz', any_of=('ms2', 'peak', 'mzs', 'fragment'), exclude=('precursor', 'prec')) or ('mz' if 'mz' in names else None),
         its=col('intens', exclude=('precursor', 'base')), peaks=col('peaks'),
         prec=col('prec', any_of=('mz',)), adduct=col('adduct') or col('precursor_type') or col('ion_type'),
         mode=col('ion', any_of=('mode', 'polar')) or col('polarity'), ce=col('collision') or col('energy'),
         formula=col('formula'))
# Enveda-180 release schema (seen 2026-10-18: MGF-style, every field a string): override the guesses
KNOWN = dict(ik='inchikey', smiles='smiles', prec='pepmass', mzs='mzs', its='intensities', adduct='adduct', mode='ionmode',
             ce='collision_energies', formula='formula')
for k, v in KNOWN.items():
    if v in names:
        C[k] = v
if C['mzs'] and C['its']:
    C['peaks'] = None
print('detected columns:', C, flush=True)
need = ['ik', 'prec', 'adduct']
if any(C[k] is None for k in need) or (C['mzs'] is None or C['its'] is None) and C['peaks'] is None:
    print('ENVEDA180 SCHEMA UNSUPPORTED -> fix the column detection; schema printed above')
    sys.exit(0)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    return hits[0]


train_ik = set(pq.read_table(find('train.parquet'), columns=['inchikey14']).column(0).to_pylist())
ik = pq.read_table(PQ, columns=[C['ik']]).column(0).to_pylist()
ik14 = np.array([str(k)[:14] if k is not None else '' for k in ik])
unseen = sorted(set(ik14) - train_ik - {''})
print(f'compounds: {len(set(ik14)):,}; not in train by InChIKey-14: {len(unseen):,}', flush=True)
keep_cols = [c for c in {C[k] for k in C} if c]
_us = set(unseen)
full_keys = sorted({str(k) for k, k14 in zip(ik, ik14) if k14 in _us})
del ik, ik14
import pyarrow.compute as pc
_want = pa.array(full_keys)
parts = []                                   # stream: big nested list columns can't go to pandas in one piece
for b in pq.ParquetFile(PQ).iter_batches(columns=keep_cols, batch_size=100_000):
    b = b.filter(pc.is_in(b.column(C['ik']), value_set=_want))
    if b.num_rows:
        parts.append(b.to_pandas())
t = pd.concat(parts, ignore_index=True)       # unseen rows only
print(f'unseen-compound spectra read: {len(t):,}', flush=True)
t['inchikey14'] = [str(k)[:14] for k in t[C['ik']]]
t['adduct'] = t[C['adduct']].astype(str).str.replace(' ', '')
print('adducts among unseen-compound spectra:', t.adduct.value_counts().head(15).to_dict(), flush=True)
t = t[t.adduct.isin(TEST_ADDUCTS)].copy()
if C['peaks'] and not (C['mzs'] and C['its']):
    pk = t[C['peaks']].map(lambda p: np.asarray(p, float).reshape(-1, 2) if p is not None else np.zeros((0, 2)))
    t['ms2_mzs'] = pk.map(lambda a: a[:, 0].tolist()); t['ms2_normalized_intensities'] = pk.map(lambda a: a[:, 1].tolist())
else:
    t['ms2_mzs'] = t[C['mzs']].map(lambda v: [float(x) for x in v]); t['ms2_normalized_intensities'] = t[C['its']].map(lambda v: [float(x) for x in v])
t['ms2_normalized_intensities'] = [list(np.asarray(i) / max(i)) if len(i) and max(i) > 0 else list(i) for i in t.ms2_normalized_intensities]
import re as _re
_num = lambda s: [float(x) for x in _re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', str(s))]
t['precursor_mz'] = t[C['prec']].map(lambda s: (_num(s) or [np.nan])[0]).astype(float)   # 'mz [intensity]'
if C['mode']:
    m = t[C['mode']].astype(str).str.lower()
    t['ionization_mode'] = np.where(m.str.contains('neg') | m.str.contains('-'), 'negative', 'positive')
else:
    t['ionization_mode'] = np.where(t.adduct.str.endswith('-'), 'negative', 'positive')
t['instrument_type'] = 'timsTOF'
ce = t[C['ce']] if C['ce'] else pd.Series([None] * len(t), index=t.index)
t['collision_energy_ev'] = ce.map(lambda v: _num(v) if v is not None and str(v) not in ('nan', 'None', '') else [])
t['collision_energy_orig'] = ce.astype(str); t['collision_energy_orig_units'] = 'eV'
t['base_peak_intensity'] = None
t['smiles'] = t[C['smiles']] if C['smiles'] else None
t['inchi'] = t[C['inchi']] if C['inchi'] else None
t['formula'] = t[C['formula']] if C['formula'] else None
out = t[['inchikey14', 'smiles', 'inchi', 'formula', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
         'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_ev', 'collision_energy_orig',
         'collision_energy_orig_units']].reset_index(drop=True)
out.to_parquet('/kaggle/working/enveda180_questions.parquet', index=False)
n_ce = out.collision_energy_ev.map(len)
rep = dict(spectra=len(out), compounds=int(out.inchikey14.nunique()),
           compounds_by_mode=out.groupby('ionization_mode').inchikey14.nunique().to_dict(),
           adducts=out.adduct.value_counts().to_dict(), multi_ce_share=round(float((n_ce > 1).mean()), 4),
           has_smiles=bool(C['smiles']), has_inchi=bool(C['inchi']), secs=round(time.time() - T0))
print('ENVEDA180 QUESTIONS ' + json.dumps(rep, default=str))
