# Winner obligations (Rules 2.8): record exactly which input files this notebook version loads. Writes
# /kaggle/working/input_manifest.json (path, bytes, sha256 per file under /kaggle/input, plus per-dataset totals).
# Runs on the push run only: the hidden rerun uses the same attached dataset versions, so it skips the hashing.
import hashlib as _mh, json as _mj, os as _mo, time as _mt
from concurrent.futures import ThreadPoolExecutor as _MTP


def _msha(p, _b=1 << 23):
    h = _mh.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(_b), b''):
            h.update(ch)
    return h.hexdigest()


try:
    if not globals().get('IS_RERUN', False):
        _mt0 = _mt.time()
        _mfiles = sorted(_mo.path.join(d, f) for d, _, fs in _mo.walk('/kaggle/input', followlinks=True) for f in fs)
        with _MTP(8) as _mx:
            _mh_ = list(_mx.map(_msha, _mfiles))
        _mrows = [dict(path=p[len('/kaggle/input/'):], bytes=_mo.path.getsize(p), sha256=h) for p, h in zip(_mfiles, _mh_)]
        _mds = {}
        for r in _mrows:
            k = '/'.join(r['path'].split('/')[:2]) if r['path'].startswith('datasets/') else r['path'].split('/')[0]
            _mds.setdefault(k, dict(files=0, bytes=0)); _mds[k]['files'] += 1; _mds[k]['bytes'] += r['bytes']
        _mj.dump(dict(created=_mt.strftime('%Y-%m-%d %H:%M:%S UTC', _mt.gmtime()), n_files=len(_mrows), datasets=_mds, files=_mrows),
                 open('/kaggle/working/input_manifest.json', 'w'), indent=0)
        print(f'INPUT MANIFEST {len(_mrows)} files, {sum(r["bytes"] for r in _mrows) / 1e9:.1f} GB, {len(_mds)} inputs, '
              f'{_mt.time() - _mt0:.0f}s', flush=True)
    else:
        print('INPUT MANIFEST skipped on the hidden rerun (same attached versions as the push run)', flush=True)
except Exception as _me:          # never let bookkeeping fail a submission
    print('INPUT MANIFEST failed:', repr(_me)[:200], flush=True)
