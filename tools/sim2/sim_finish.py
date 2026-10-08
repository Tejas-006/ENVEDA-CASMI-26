# sim2: save everything needed to re-rank offline, then print a quick MRR@25 (the full protocol is in casmi26-sim2-score)
import json, shutil


def _js(obj, path):
    json.dump(obj, open(path, 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))


_mids = [m for m in QUERIES.molecule_id if m in SIMD]
if _mids:
    np.savez_compressed(os.path.join(SIM_OUT, 'cands.npz'),
                        molecule_id=np.concatenate([np.repeat(str(m), len(SIMD[m]['y'])) for m in _mids]),
                        key=np.concatenate([SIMD[m]['keys'] for m in _mids]),
                        smiles=np.concatenate([SIMD[m]['smiles'] for m in _mids]),
                        X=np.concatenate([SIMD[m]['X'] for m in _mids]),
                        score=np.concatenate([SIMD[m]['score'] for m in _mids]),
                        y=np.concatenate([SIMD[m]['y'] for m in _mids]),
                        n_all=np.array([SIMD[m]['n_all'] for m in _mids]), mols=np.array([str(m) for m in _mids]),
                        names=np.array(SIM_NAMES or []))
_js({str(k): v for k, v in BASE0.items()}, os.path.join(SIM_OUT, 'base_pre_ice.json'))
_js({str(k): v for k, v in BASE_POST_ICE.items()}, os.path.join(SIM_OUT, 'base_post_ice.json'))
_js({str(k): v for k, v in (ICE_SCORES or {}).items()}, os.path.join(SIM_OUT, 'ice_scores.json'))
_js({str(k): v for k, v in (GL_SCORES or {}).items()}, os.path.join(SIM_OUT, 'gl_scores.json'))
for _f in ['eng_lists.json', 'pc_lists.json']:
    if os.path.exists(f'/kaggle/working/{_f}'):
        shutil.copy(f'/kaggle/working/{_f}', os.path.join(SIM_OUT, _f))
shutil.copy('submission.csv', os.path.join(SIM_OUT, 'sub_final.csv'))


def _rr(sub_path):
    s = pd.read_csv(sub_path)
    out = {}
    for mid, smi in zip(s.molecule_id, s.smiles):
        t, r = TRUTH_KEY.get(mid), 0.0
        for i, x in enumerate(str(smi).split(';')[:25], 1):
            if t and metric_key(x) == t:
                r = 1.0 / i; break
        out[mid] = r
    return out


_q = QUERIES.set_index('molecule_id')
summary = dict(shard=SHARD, n_queries=len(QUERIES), secs=round(time.time() - T0), n_errors=n_err,
               train_structs=STRUCT_FILTER, np_filter=NP_FILTER, ice_stats=ice_stats, gl_stats=gl_stats,
               cand_recall=float(np.mean([SIMD[m]['y'].any() for m in _mids])) if _mids else None)
for _name, _p in [('v4_pre_fusion', 'sub_v4.csv'), ('final', 'sub_final.csv')]:
    rr = pd.Series(_rr(os.path.join(SIM_OUT, _p)))
    cls = _q.sim_class.reindex(rr.index)
    summary[_name] = dict(mrr=round(rr.mean(), 4), top1=round((rr == 1).mean(), 4),
                          top5=round((rr >= 0.2).mean(), 4), recall25=round((rr > 0).mean(), 4),
                          by_class={int(c): round(rr[cls == c].mean(), 4) for c in sorted(cls.dropna().unique())})
    if 'in_pool' in _q.columns:
        ip = _q.in_pool.reindex(rr.index).astype(bool)
        summary[_name]['by_in_pool'] = {'in_pool': round(rr[ip].mean(), 4) if ip.any() else None,
                                        'not_in_pool': round(rr[~ip].mean(), 4) if (~ip).any() else None}
summary['source'] = SOURCE
if 'in_pool' in _q.columns:
    summary['truth_in_pool'] = round(float(_q.in_pool.mean()), 4)
_js(summary, os.path.join(SIM_OUT, 'summary.json'))
print(json.dumps(summary, indent=1, default=str))
print('sim_out files:', sorted(os.listdir(SIM_OUT)))

# keep the saved output small: drop helper installs/caches the pipeline leaves in /kaggle/working
for _d in ['ice_site', 'eng', 'gl_work', 'ice_work', '__pycache__', 'numba_cache', 'work', 'smoke_comp']:
    shutil.rmtree(os.path.join('/kaggle/working', _d), ignore_errors=True)
print('kept outputs:', sorted(os.listdir('/kaggle/working')))
