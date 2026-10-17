"""Offline replay of the v6 final step (tools/lb/v6_final.py) on a saved sim2 output folder, CPU only.

Re-runs identity fusion -> GLACIER/ICEBERG re-scoring -> metric-safe tail from the files sim2 saves (sub_v4.csv,
eng_lists.json, pc_lists.json, base_pre_ice.json, gl_scores.json, ice_scores.json, queries.parquet) and scores the
result with the official metric key, so final-step variants are compared in seconds instead of a 1.7 GPU-hour sim2 run.
Only final-step changes can be judged here: anything that needs new GLACIER/ICEBERG scores or new candidates needs a
sim2 run. The default config must reproduce the run's own sub_final.csv (checked on every call).

usage: python tools/replay/replay_final.py <sim_out dir> <dir with gl_fuse.py> <dir with fuse.py> [--cache keys.json]
       [--variants]     (also run the built-in ablations)
gl_fuse.py / fuse.py come from the public Kaggle datasets ahmedberatozer/casmi26-glacier and casmi26-iceberg; they
are not committed here.
"""
import argparse, collections, json, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()

# v6 constants (baseline v4n: ALPHA, KRR from the fusion cell; ICE_LAM = GL_LAM = 1.0; v6_final.V6_WINDOW)
V6 = dict(window=40, alpha=0.6, krr=3.0, ice_lam=1.0, gl_lam=1.0, rescore=True, fusion=True,
          tail=('overflow', 'prefusion', 'ranker', 'eng2', 'pubchem'), pc_into_window=0, sim='shipped')


class Keys:
    """identity = v6_identity (official metric key + heavy-atom counts); key = official metric key. Disk-cached."""

    def __init__(self, path=None):
        self.path, self.d = path, {}
        if path and os.path.exists(path):
            self.d = json.load(open(path))
        self.n_new = 0

    def _compute(self, s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s and s != 'CCO' else None
        if m is None:
            return None
        try:
            heavy = sorted(collections.Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() != 'H').items())
            Chem.RemoveStereochemistry(m)
            return [Chem.MolToInchiKey(_TE.Canonicalize(m))[:14], [list(x) for x in heavy], CalcMolFormula(m)]
        except (ValueError, RuntimeError, TypeError):
            return None

    def get(self, s):
        s = str(s).strip()
        if s not in self.d:
            self.d[s] = self._compute(s); self.n_new += 1
        return self.d[s]

    def identity(self, s):
        v = self.get(s)
        return (v[0], tuple(tuple(x) for x in v[1])) if v else None

    def key(self, s):
        v = self.get(s)
        return v[0] if v else None

    def formula(self, s):
        v = self.get(s)
        return v[2] if v else s

    def save(self):
        if self.path and self.n_new:
            json.dump(self.d, open(self.path, 'w'))


def unique(K, seq, limit=25):
    out, seen = [], set()
    for s in seq:
        k = K.identity(s)
        if k is not None and k not in seen:
            seen.add(k); out.append(str(s).strip())
            if len(out) == limit:
                break
    return out


def fusion(K, first, second, alpha, offset, limit):
    score, rep = {}, {}
    for cands, w in ((first, 1.0), (second, alpha)):
        seen = set()
        for s in cands:
            k = K.identity(s)
            if k is None or k in seen:
                continue
            seen.add(k)
            score[k] = score.get(k, 0.0) + w / (offset + len(seen))
            rep.setdefault(k, s)
    order = sorted(score, key=lambda k: -score[k])[:limit]
    return [rep[k] for k in order], [score[k] for k in order]


def nonempty(d, m):
    v = d.get(str(m), {}) if isinstance(d, dict) else {}
    return v if isinstance(v, dict) and any(x is not None for x in v.values()) else {}


def final_lists(run, K, gl_fuse, ice_fuse, cfg):
    out, stats = {}, collections.Counter()
    for mid, pre_s in run['pre'].items():
        first = [s for s in str(pre_s).split(';') if s and s != 'CCO']
        e_smis = list((run['eng'].get(mid) or {}).get('smiles', []))
        pc = list((run['pc'].get(mid) or {}).get('pc', []))
        if cfg['fusion'] and e_smis:
            cands, scs = fusion(K, first, e_smis, cfg['alpha'], cfg['krr'], cfg['window'])
        else:
            cands = unique(K, first, cfg['window']); scs = [1.0 / (cfg['krr'] + r) for r in range(1, len(cands) + 1)]
        if cfg['pc_into_window']:
            have = {K.identity(s) for s in cands}
            add = [s for s in unique(K, pc, cfg['pc_into_window'] * 3) if K.identity(s) not in have][:cfg['pc_into_window']]
            lo = min(scs) if scs else 0.0
            cands += add; scs += [lo - 1e-6 * (i + 1) for i in range(len(add))]
        keys = [K.key(s) or s for s in cands]
        sk = '_mce' if cfg.get('sim') == 'mce' else ''        # --both-ce runs: multi-energy simulator scores
        gl, ice = nonempty(run['gl' + sk], mid), nonempty(run['ice' + sk], mid)
        if cfg['rescore'] and gl and len(cands) > 1:
            o = gl_fuse.rerank_multi(cands, keys, scs, [K.formula(s) for s in cands], [{}, gl],
                                     [cfg['ice_lam'], cfg['gl_lam']], top_n=len(cands))
            cands = [cands[i] for i in o]; stats['glacier'] += 1
        elif cfg['rescore'] and ice and len(cands) > 1:
            o = ice_fuse.rerank(cands, keys, scs, [K.formula(s) for s in cands], ice, lam=cfg['ice_lam'], top_n=len(cands))
            cands = [cands[i] for i in o]; stats['iceberg_fallback'] += 1
        ranker = list((run['base'].get(mid) or [[]])[0])
        parts = dict(overflow=cands[25:], prefusion=first, ranker=ranker, eng2=e_smis, pubchem=pc)
        tail = [s for p in cfg['tail'] for s in parts[p]]
        out[mid] = unique(K, cands[:25] + tail, 25)
    return out, stats


def score(K, lists, truth):
    rr = {}
    for mid, t in truth.items():
        r = 0.0
        for i, s in enumerate(lists.get(mid, [])[:25], 1):
            if t and K.key(s) == t:
                r = 1.0 / i; break
        rr[mid] = r
    return pd.Series(rr)


def summary(rr):
    return dict(mrr=round(rr.mean(), 4), top1=round((rr == 1).mean(), 4), top5=round((rr >= 0.2).mean(), 4),
                recall25=round((rr > 0).mean(), 4))


def paired(a, b, n=5000, seed=0):
    d = (b - a.reindex(b.index)).values
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return dict(delta=round(d.mean(), 4), ci95=[round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)],
                better=int((d > 0).sum()), worse=int((d < 0).sum()))


def load_run(d):
    j = lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}
    csv = lambda f: dict(zip(*[pd.read_csv(os.path.join(d, f))[c].astype(str) for c in ('molecule_id', 'smiles')]))
    # pc_lists_v6.json (sim2 runs from 2026-10-17) = the PubChem lists after the notebook re-sorted gated ones
    return dict(pre=csv('sub_v4.csv'), final=csv('sub_final.csv'), eng=j('eng_lists.json'),
                pc=j('pc_lists_v6.json') or j('pc_lists.json'),
                base=j('base_pre_ice.json'), gl=j('gl_scores.json'), ice=j('ice_scores.json'),
                gl_mce=j('gl_scores_mce.json'), ice_mce=j('ice_scores_mce.json'),   # --both-ce runs
                q=pd.read_parquet(os.path.join(d, 'queries.parquet')))


VARIANTS = {
    'no_rescore': dict(rescore=False),
    'no_fusion': dict(fusion=False),
    'no_fusion_no_rescore': dict(fusion=False, rescore=False),
    'gl_lam_0.5': dict(gl_lam=0.5),
    'window_25': dict(window=25),
    'tail_prefusion_first': dict(tail=('prefusion', 'overflow', 'ranker', 'eng2', 'pubchem')),
    'pc_into_window_10': dict(pc_into_window=10),
    'v6.1 (gl_lam 0.5, window 25)': dict(gl_lam=0.5, window=25),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim_out'); ap.add_argument('gl_dir'); ap.add_argument('ice_dir')
    ap.add_argument('--cache', default=None); ap.add_argument('--variants', action='store_true')
    a = ap.parse_args()
    sys.path[:0] = [a.gl_dir, a.ice_dir]
    import gl_fuse, fuse as ice_fuse
    run, K = load_run(a.sim_out), Keys(a.cache)
    truth = {str(m): K.key(s) for m, s in zip(run['q'].molecule_id, run['q'].smiles)}
    try:
        base, st = final_lists(run, K, gl_fuse, ice_fuse, V6)
        same = sum(base[m] == run['final'][m].split(';') for m in base)
        rr_run = score(K, {m: s.split(';') for m, s in run['final'].items()}, truth)
        rr_pre = score(K, {m: s.split(';') for m, s in run['pre'].items()}, truth)
        rr_v6 = score(K, base, truth)
        print(f'reproduction: {same}/{len(base)} molecules identical to sub_final.csv; stats {dict(st)}')
        print('run final   ', summary(rr_run)); print('replay v6   ', summary(rr_v6)); print('pre-fusion  ', summary(rr_pre))
        if a.variants:
            for name, over in VARIANTS.items():
                lists, _ = final_lists(run, K, gl_fuse, ice_fuse, {**V6, **over})
                rr = score(K, lists, truth)
                print(f'{name:22s}', summary(rr), 'vs v6', paired(rr_v6, rr))
    finally:
        K.save()


if __name__ == '__main__':
    main()
