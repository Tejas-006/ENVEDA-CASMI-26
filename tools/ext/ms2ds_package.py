# Offline MS2DeepScore 2.x package for the internet-off competition notebooks (CPU, internet ON; plan step 9).
# Code: matchms/ms2deepscore (Apache-2.0). Model: "Dual mode MS2Deepscore model (MS2DeepScore >2.6)", Zenodo 17826815,
# CC BY 4.0, trained on public GNPS / MoNA / MassBank / MSnLib spectra (no NIST). One model embeds positive and negative
# spectra into the same space, so cross-ion-mode similarity is a cosine.
# Installs ms2deepscore + matchms with --no-deps into a separate target dir and adds only the modules their import path
# actually needs (torch / numpy / pandas / rdkit / numba stay the image's or E006's own), then checks the model on the
# competition train set: same-compound vs different-compound cosine within and across ion modes.
# Output (/kaggle/working/ms2ds): wheels/, model/, manifest.json -> attach this notebook's output as an input.
import glob, importlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, requests

T0 = time.time()
W = '/kaggle/working/ms2ds'; WH, MD, TGT = f'{W}/wheels', f'{W}/model', '/tmp/ms2ds_site'
for d in (WH, MD, TGT):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def sh(*a):
    r = subprocess.run(list(a), capture_output=True, text=True)
    if r.returncode:
        log('FAILED', ' '.join(a)[:160], (r.stdout + r.stderr)[-800:])
    return r.returncode


def pipget(spec):
    """Download one wheel (no deps) and install it into the private target dir."""
    if sh(sys.executable, '-m', 'pip', 'download', spec, '--no-deps', '--only-binary=:all:', '-d', WH):
        sh(sys.executable, '-m', 'pip', 'download', spec, '--no-deps', '-d', WH)          # sdist-only package
    name = spec.split('=')[0].split('<')[0].split('>')[0]
    return sh(sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--find-links', WH, '--target', TGT, name)


PIP_NAME = {'sklearn': 'scikit-learn', 'yaml': 'pyyaml', 'PIL': 'pillow', 'deprecated': 'Deprecated', 'pubchempy': 'PubChemPy',
            'dateutil': 'python-dateutil', 'google': 'protobuf'}
sys.path.insert(0, TGT)
pipget('ms2deepscore==2.11.0')
pipget('matchms')
added = []
for _ in range(40):
    importlib.invalidate_caches()
    for m in [m for m in sys.modules if m.split('.')[0] in ('ms2deepscore', 'matchms', 'chemap')]:
        del sys.modules[m]
    try:
        from ms2deepscore.models import load_model
        from ms2deepscore import MS2DeepScore
        from matchms import Spectrum
        break
    except ModuleNotFoundError as e:
        mod = (e.name or '').split('.')[0]
        log('missing module', mod)
        if not mod or mod in added or pipget(PIP_NAME.get(mod, mod)):
            raise
        added.append(mod)
else:
    raise RuntimeError('import chain did not resolve')
import torch, rdkit, numba
log('import ok; added from wheels:', added, '| torch', torch.__version__, 'numpy', np.__version__, 'rdkit', rdkit.__version__)

# ---- model (Zenodo 17826815, CC BY 4.0) ----
for f in ('ms2deepscore_model.pt', 'settings.json', 'embedding_model_settings.json', 'embedding_evaluator.pt'):
    p = os.path.join(MD, f)
    if not os.path.exists(p):
        with requests.get(f'https://zenodo.org/records/17826815/files/{f}?download=1', stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(p, 'wb') as fo:
                for ch in r.iter_content(1 << 22):
                    fo.write(ch)
    log('model file', f, f'{os.path.getsize(p) / 1e6:.1f} MB')
model = load_model(os.path.join(MD, 'ms2deepscore_model.pt'))
sim = MS2DeepScore(model)
log('model loaded; settings:', {k: v for k, v in vars(model.model_settings).items() if not isinstance(v, (list, dict))} if hasattr(model, 'model_settings') else '?')

# ---- check on the competition train set: same compound, both ion modes ----
CHECK = {}
try:
    TR = sorted(glob.glob('/kaggle/input/**/train.parquet', recursive=True), key=len)[0]
    cols = ['inchikey14', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'ionization_mode', 'adduct']
    pf = pq.ParquetFile(TR)
    parts, have = [], 0
    for b in pf.iter_batches(columns=cols, batch_size=200_000):
        parts.append(b.to_pandas()); have += len(parts[-1])
        if have >= 600_000:
            break
    t = pd.concat(parts, ignore_index=True)
    t = t[t.ms2_mzs.map(len) >= 3]
    both = t.groupby('inchikey14').ionization_mode.nunique()
    iks = np.random.default_rng(0).choice(both[both == 2].index.values, size=min(300, int((both == 2).sum())), replace=False)
    sel = t[t.inchikey14.isin(set(iks))].groupby(['inchikey14', 'ionization_mode']).head(1).reset_index(drop=True)
    log(f'check set: {len(iks)} compounds with spectra in both modes ({len(sel)} spectra)')


    def to_spec(r):
        mz = np.asarray(r.ms2_mzs, float); it = np.asarray(r.ms2_normalized_intensities, float)
        o = np.argsort(mz)
        return Spectrum(mz=mz[o], intensities=it[o], metadata={'precursor_mz': float(r.precursor_mz),
                                                              'ionmode': str(r.ionization_mode).lower(), 'adduct': r.adduct})


    specs = [to_spec(r) for r in sel.itertuples()]
    t1 = time.time(); emb = sim.get_embedding_array(specs); dt = time.time() - t1
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    log(f'embeddings {emb.shape} in {dt:.1f}s ({len(specs) / max(dt, 1e-9):.0f} spectra/s on CPU)')
    pos = sel.index[sel.ionization_mode.str.lower() == 'positive'].values
    neg = sel.index[sel.ionization_mode.str.lower() == 'negative'].values
    ik = sel.inchikey14.values
    S = emb[pos] @ emb[neg].T
    same = ik[pos][:, None] == ik[neg][None, :]
    from sklearn.metrics import roc_auc_score
    auc_x = float(roc_auc_score(same.ravel(), S.ravel()))
    rank = [int((S[i] > S[i, same[i]].max()).sum()) + 1 for i in range(len(pos)) if same[i].any()]
    Spp = emb[pos] @ emb[pos].T
    log(f'cross-mode (pos query -> neg spectra): same-compound AUC {auc_x:.3f}; rank of the same compound among {len(neg)}: '
        f'median {np.median(rank):.0f}, top-1 {np.mean(np.array(rank) == 1):.2f}, top-10 {np.mean(np.array(rank) <= 10):.2f}')
    CHECK = dict(compounds=len(iks), cross_mode_auc=round(auc_x, 4), cross_mode_top1=round(float(np.mean(np.array(rank) == 1)), 3),
                 cross_mode_top10=round(float(np.mean(np.array(rank) <= 10)), 3), cpu_spectra_per_s=round(len(specs) / max(dt, 1e-9)))
except Exception as e:
    CHECK = dict(error=repr(e)[:300])
    log('check failed', repr(e)[:300])
# keep only the wheels that were installed
keep = {n.lower().replace('-', '_') for n in ['ms2deepscore', 'matchms'] + [PIP_NAME.get(m, m) for m in added]}
for f in glob.glob(f'{WH}/*.whl'):
    if os.path.basename(f).split('-')[0].lower() not in keep:
        os.remove(f)
man = dict(ms2deepscore='2.11.0', wheels=sorted(os.path.basename(f) for f in glob.glob(f'{WH}/*.whl')), added_modules=added,
           model='Zenodo 17826815 (CC BY 4.0), dual mode, MS2DeepScore >= 2.6', code_licence='Apache-2.0',
           image=dict(torch=torch.__version__, numpy=np.__version__, rdkit=rdkit.__version__, python=sys.version.split()[0]),
           check=CHECK,
           install='pip install --no-deps --no-index --find-links <wheels> --target /tmp/ms2ds_site ms2deepscore matchms <added>; '
                   'sys.path.insert(0, "/tmp/ms2ds_site")')
json.dump(man, open(f'{W}/manifest.json', 'w'), indent=1)
print('MS2DS PACKAGE ' + json.dumps(man))
