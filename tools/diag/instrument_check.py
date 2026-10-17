# Instrument check (CPU, ~10 min): which instrument do the test spectra come from?
# 1. Reads test.parquet's columns and any instrument / ion-mobility fields directly.
# 2. Trains a spectrum -> instrument classifier on a train sample (GroupKFold by compound, so the reported accuracy
#    is on unseen compounds) and predicts the instrument of every test spectrum.
# The visible test.parquet is a stand-in for the hidden one: treat its counts as a hint, never score anything on it.
SAMPLE_ROWS, SEED = 200_000, 0
import glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def inst(x):
    return str(x).lower().replace('-', '').replace(' ', '') if x is not None else 'unk'


TR, TE = find('train.parquet'), find('test.parquet')
report = {}

# ---- 1. what the test file says about itself ----
te_schema = pq.read_schema(TE)
report['test_columns'] = te_schema.names
print('test columns:', te_schema.names)
hints = [c for c in te_schema.names if any(k in c.lower() for k in ('instrument', 'mobility', 'ccs', 'tims', 'k0', 'source'))]
te_meta_cols = [c for c in te_schema.names if not c.startswith('ms2_')]
te = pq.read_table(TE, columns=te_meta_cols).to_pandas()
report['test_rows'] = len(te)
print(f'test rows: {len(te):,}; instrument/mobility-like columns: {hints or "none"}')
for c in hints + [c for c in ('ionization_mode', 'adduct') if c in te.columns]:
    vc = te[c].astype(str).value_counts().head(10)
    report[f'test_{c}_counts'] = vc.to_dict()
    print(f'\n{c}:\n{vc.to_string()}')

# ---- 2. spectrum features that differ by instrument ----
def features(mzs, its, prec):
    out = []
    for mz, it, p in zip(mzs, its, prec):
        mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]
        if len(mz) == 0:
            out.append([0] + [np.nan] * 9)
            continue
        itn = it / it.max()
        frac = np.abs(mz - np.round(mz, 4))                     # m/z precision as stored
        pr = itn / itn.sum()
        gaps = np.diff(np.sort(mz))
        out.append([len(mz), np.log10(itn.min() + 1e-9), np.median(itn), -(pr * np.log(pr)).sum(),
                    float((frac < 1e-9).mean()), mz.max() / max(float(p or 0), 1e-6), mz.min(),
                    np.median(gaps) if len(gaps) else np.nan, np.min(gaps) if len(gaps) else np.nan, float(p or np.nan)])
    return np.array(out, np.float32)


FEATS = ['n_peaks', 'log_min_int', 'median_int', 'int_entropy', 'frac_4dp', 'max_mz_over_prec', 'min_mz',
         'median_gap', 'min_gap', 'precursor_mz']

pf = pq.ParquetFile(TR)
rng = np.random.default_rng(SEED)
parts, n = [], 0
for g in rng.permutation(pf.num_row_groups):
    t = pf.read_row_group(int(g), columns=['inchikey14', 'instrument_type', 'ionization_mode', 'precursor_mz',
                                           'ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
    parts.append(t.sample(min(len(t), 20_000), random_state=SEED))
    n += len(parts[-1])
    if n >= SAMPLE_ROWS:
        break
tr = pd.concat(parts, ignore_index=True)
tr['inst'] = [inst(x) for x in tr.instrument_type]
report['train_sample_instrument_counts'] = tr.inst.value_counts().to_dict()
print('\ntrain sample instruments:\n' + tr.inst.value_counts().to_string())
X = np.c_[features(tr.ms2_mzs, tr.ms2_normalized_intensities, tr.precursor_mz),
          (tr.ionization_mode.astype(str).str.lower() == 'positive').values]
classes = sorted(tr.inst.unique())
y = tr.inst.map({c: i for i, c in enumerate(classes)}).values
params = dict(objective='multiclass', num_class=len(classes), learning_rate=0.05, num_leaves=63, verbose=-1, seed=SEED)

# held-out accuracy on unseen compounds
oof = np.zeros((len(y), len(classes)))
for trn, val in GroupKFold(5).split(X, y, tr.inchikey14.astype(str).values):
    m = lgb.train(params, lgb.Dataset(X[trn], y[trn]), 300)
    oof[val] = m.predict(X[val])
acc = float((oof.argmax(1) == y).mean())
per = {c: float((oof.argmax(1)[y == i] == i).mean()) for i, c in enumerate(classes)}
report['classifier_heldout_accuracy'] = acc
report['classifier_recall_per_instrument'] = per
print(f'\nclassifier accuracy on unseen compounds: {acc:.3f}; recall per instrument: '
      + ', '.join(f'{c} {v:.2f}' for c, v in per.items()))

# ---- 3. predict the test spectra ----
MZC = next(c for c in te_schema.names if c.startswith('ms2_') and 'mz' in c)
ITC = next(c for c in te_schema.names if c.startswith('ms2_') and 'int' in c)
tes = pq.read_table(TE, columns=[c for c in ('precursor_mz', 'ionization_mode') if c in te_schema.names] + [MZC, ITC]).to_pandas()
prec = tes.precursor_mz if 'precursor_mz' in tes else pd.Series(np.nan, index=tes.index)
Xt = np.c_[features(tes[MZC], tes[ITC], prec),
           (tes.ionization_mode.astype(str).str.lower() == 'positive').values if 'ionization_mode' in tes else np.ones(len(tes))]
m = lgb.train(params, lgb.Dataset(X, y), 300)
pt = m.predict(Xt)
pred = pd.Series([classes[i] for i in pt.argmax(1)]).value_counts(normalize=True).round(3)
report['test_predicted_instrument_share'] = pred.to_dict()
report['test_mean_probability'] = dict(zip(classes, pt.mean(0).round(3).tolist()))
print('\npredicted instrument share of test spectra:\n' + pred.to_string())
if 'instrument_type' in te.columns:
    report['test_says_instrument'] = te.instrument_type.map(inst).value_counts(normalize=True).round(3).to_dict()
json.dump(report, open('/kaggle/working/instrument_check.json', 'w'), indent=1, default=str)
print('\nINSTRUMENT CHECK ' + json.dumps({k: report[k] for k in ('test_columns', 'classifier_heldout_accuracy',
                                                                  'test_predicted_instrument_share')}, default=str))
