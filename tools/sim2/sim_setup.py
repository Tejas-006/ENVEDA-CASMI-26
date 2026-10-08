# sim2 setup (replaces the baseline's smoke switch): a pseudo-competition folder built from held-out TRAINING compounds.
# Queries are natural products (compounds also in COCONUT/ChEBI/LIPID MAPS), so the true structure stays in the
# candidate pool after its spectra are removed from the library. timsTOF spectra are preferred (like the hidden test).
#   class 1 = query spectra from one instrument; the compound's other spectra stay in the library
#   class 2 = every spectrum of the compound is removed from the library
SHARD, SHARD_SIZE = 0, 500
SOURCE = 'train'   # 'train' = held-out training compounds; 'external' = public spectra of compounds NOT in train
CLASS1_FRAC, MAX_SPEC, SEED, SIM_TOPK, FILTER_TRAIN_STRUCTS = 0.3, 10, 2026, 300, False   # True breaks the v1 Library: its spectrum cache indexes train_structs rows by position
import zlib, pickle, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()


def metric_key(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToInchiKey(_TE.Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def scaffold_of(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
    except Exception:
        return ''


def _inst(x):
    return str(x).lower().replace('-', '').replace(' ', '') if x is not None else 'unk'


IS_RERUN = True
REAL_COMP = COMP
SIM_OUT = '/kaggle/working/sim_out'; os.makedirs(SIM_OUT, exist_ok=True)
SM = '/tmp/sim_comp'; os.makedirs(SM, exist_ok=True)   # outside /kaggle/working: not saved as output
TEST_SCHEMA = pq.read_schema(os.path.join(REAL_COMP, 'test.parquet'))
TR_PATH = os.path.join(REAL_COMP, 'train.parquet')


def _k14(k):
    if isinstance(k, (bytes, np.bytes_)):
        k = k.decode('ascii', 'ignore')
    return str(k).strip().upper()[:14]


# @@SELECT@@

# ICEBERG/GLACIER budgets are totals for the whole run: scale the submission's budgets to this shard's size
# (5400 s / 4000 s were sized for a full test set; a 20-query test must not inherit them)
ICE_BUDGET = int(max(120, 5400 * SHARD_SIZE / 500))
SIM_GL_BUDGET = int(max(120, 4000 * SHARD_SIZE / 500))

# Stream the helper processes (engine 2, PubChem channel, ICEBERG/GLACIER runners) into the log live, with
# timestamps, instead of holding their output until they finish.
import subprocess as _sp, threading as _th
_orig_sp_run = _sp.run


def _sim2_run(cmd, *a, capture_output=False, timeout=None, check=False, **kw):
    is_py = isinstance(cmd, (list, tuple)) and len(cmd) > 1 and 'python' in os.path.basename(str(cmd[0]))
    if not (capture_output and is_py):
        return _orig_sp_run(cmd, *a, capture_output=capture_output, timeout=timeout, check=check, **kw)
    text = kw.pop('text', False) or kw.pop('universal_newlines', False)
    name = os.path.basename(str(cmd[1] if isinstance(cmd, (list, tuple)) and len(cmd) > 1 else cmd))[:60]
    t0 = time.time()
    print(f'[sim2] {time.time() - T0:7.0f}s START {name}', flush=True)
    p = _sp.Popen(cmd, *a, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1, **kw)
    killer = _th.Timer(timeout, p.kill) if timeout else None
    if killer:
        killer.start()
    out = []
    try:
        for line in p.stdout:
            out.append(line)
            print('    | ' + line, end='', flush=True)
        p.wait()
    finally:
        if killer:
            killer.cancel()
    secs = time.time() - t0
    print(f'[sim2] {time.time() - T0:7.0f}s END {name} after {secs:.0f}s, exit code {p.returncode}', flush=True)
    if timeout and secs >= timeout:
        raise _sp.TimeoutExpired(cmd, timeout)
    if check and p.returncode:
        raise _sp.CalledProcessError(p.returncode, cmd)
    txt = ''.join(out)
    return _sp.CompletedProcess(cmd, p.returncode, txt if text else txt.encode(), '' if text else b'')


_sp.run = _sim2_run


def sim2_mark(label):
    print(f'[sim2] {time.time() - T0:7.0f}s STEP {label}', flush=True)


SIMD, SIM_NAMES = {}, None


def _simdump(mid, C, XF, fnames, score):
    global SIM_NAMES
    if SIM_NAMES is None:
        SIM_NAMES = list(fnames)
    keys = np.asarray(C.key, dtype=object)
    t = TRUTH.get(mid, set())
    pos = np.array([k in t for k in keys], bool)
    o = np.union1d(np.argsort(-score, kind='stable')[:SIM_TOPK], np.where(pos)[0])
    SIMD[mid] = dict(keys=keys[o].astype(str), smiles=np.asarray(C.smiles, dtype=object)[o].astype(str),
                     X=np.asarray(XF[o], np.float32), score=np.asarray(score[o], np.float32), y=pos[o],
                     n_all=len(keys))


COMP = SM
print(f'SIM2 {SOURCE} shard {SHARD}: {len(QUERIES)} queries, classes {QUERIES.sim_class.value_counts().to_dict()}, '
      f'{len(q_rows)} spectra, {len(drop_rows)} library rows removed; train_structs: {STRUCT_FILTER} | ICE_BUDGET {ICE_BUDGET}')
