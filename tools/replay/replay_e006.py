"""Offline replay of the E006 fork's final stage on a saved sim2 run (CPU, minutes instead of a 3.5 GPU-hour A/B).

E006's output is fully determined by files the run already keeps: the window-40 pass (cell 31) rebuilds the final
list from the merged lists, engine 2's list and the ICEBERG/GLACIER scores, fills the tail from the metric-safe list
(cell 27) and the engine-2 rank-2 hedge (cell 32) runs last. This script re-implements cells 26/28/31/32 line for line
and checks that the default config reproduces the run's own outputs before reporting any variant.

usage: python tools/replay/replay_e006.py <run output dir> --gl-fuse DIR --ice-fuse DIR --chem DIR [--cache keys.json]
       [--b-gl-ice W] [--variants]
  <run output dir> = `kaggle kernels output` folder: e004_candidate_inputs.json, submission_reference.csv, sim_out/
  (ice_scores.json, gl_scores.json, queries.parquet, sub_final.csv, sub_final_A.csv when the run was an A/B).
gl_fuse.py / fuse.py / chem.py come from the public Kaggle datasets ahmedberatozer/casmi26-glacier, casmi26-iceberg and
casmi26-v4b-models (code/casmi/chem.py); they are not committed here. Not replayable: frag_gap (needs the spectra).
"""
import argparse, importlib.util, json, math, os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()

E006 = dict(window=40, alpha=0.6, krr=3.0, ice_lam=1.0, gl_lam=1.0, gl_ice=None, ice_fallback=True, hedge='rank2')


VARIANTS = [('gl_ice 0.25', dict(gl_ice=0.25)), ('gl_ice 0.5 (submitted)', dict(gl_ice=0.5)), ('gl_ice 0.75', dict(gl_ice=0.75)),
            ('gl_ice 1.0', dict(gl_ice=1.0)), ('gl_ice 0.5, GL 0.75', dict(gl_ice=0.5, gl_lam=0.75)),
            ('window 80', dict(window=80)), ('window 80 + gl_ice 0.5', dict(window=80, gl_ice=0.5)),
            ('alpha 0.4', dict(alpha=0.4)), ('alpha 0.8', dict(alpha=0.8)), ('krr 1', dict(krr=1.0)), ('krr 6', dict(krr=6.0)),
            ('no ICE fallback', dict(ice_fallback=False)), ('ICE fallback lam 0.5', dict(ice_lam=0.5)),
            ('no engine-2 hedge', dict(hedge=None)), ('engine-2 swap (x1 first)', dict(hedge='swap'))]


def _load(name, d):
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Cache:
    """Disk-cached metric identity (cell 26), score key (casmi.chem.score_key), formula (cell 24 _form2), official key."""

    def __init__(self, path, chem):
        self.path, self.chem = path, chem
        self.d = json.load(open(path)) if path and os.path.exists(path) else {}
        self.n_new = 0

    def _get(self, s):
        if s not in self.d:
            self.n_new += 1
            ident = ik = form = None
            try:
                m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
                if m is not None:
                    form = CalcMolFormula(m)
                    heavy = sorted((a, b) for a, b in __import__('collections').Counter(
                        x.GetSymbol() for x in m.GetAtoms() if x.GetSymbol() != 'H').items())
                    m2 = Chem.Mol(m); Chem.RemoveStereochemistry(m2)
                    ik = Chem.MolToInchiKey(_TE.Canonicalize(m2))[:14] or None
                    ident = [ik, heavy] if ik else None
            except (ValueError, RuntimeError, TypeError):
                ident = ik = None
            self.d[s] = [ident, self.chem.score_key(s) if isinstance(s, str) else None, form, ik]
        return self.d[s]

    def identity(self, s):
        v = self._get(str(s).strip())[0]
        return (v[0], tuple(tuple(x) for x in v[1])) if v else None

    def score_key(self, s):
        return self._get(s)[1]

    def form2(self, s):
        return self._get(s)[2] or s

    def official(self, s):
        return self._get(str(s).strip())[3]

    def save(self):
        if self.path and self.n_new:
            json.dump(self.d, open(self.path, 'w'))


def safe_extend(K, original, extra, limit=25):
    out, seen = [], set()
    for smi in list(original) + list(extra):
        smi = str(smi).strip()
        ident = K.identity(smi)
        if ident is not None and ident not in seen:
            seen.add(ident); out.append(smi)
            if len(out) == limit:
                break
    return out


def metric_fusion(K, first, second, alpha, offset, limit):
    scores, rep = {}, {}
    for cands, w in ((first, 1.0), (second, alpha)):
        seen = set()
        for c in cands:
            k = K.identity(c)
            if k is None or k in seen:
                continue
            seen.add(k)
            scores[k] = scores.get(k, 0.0) + w / (offset + len(seen))
            rep.setdefault(k, c)
    order = sorted(scores, key=lambda k: -scores[k])[:limit]
    return [rep[k] for k in order], [scores[k] for k in order]


def final_lists(R, K, gl_fuse, ice_fuse, cfg):
    """cell 31 (rerank_row, processed molecules) then cell 32 (engine-2 hedge)."""
    out, branches = {}, {}
    for mid, original in R['reference'].items():
        res, br = original, 'placeholder'
        eng = R['eng'].get(mid)
        if mid in R['prefusion'] and eng and eng.get('smiles'):
            first = [s for s in R['prefusion'][mid] if s and s != 'CCO']
            cands, scores = metric_fusion(K, first, eng['smiles'], cfg['alpha'], cfg['krr'], cfg['window'])
            keys = [K.score_key(s) or s for s in cands]
            ice = R['ice'].get(mid, {}) or {}
            g = R['gl'].get(mid, {}) or {}
            gl = g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}
            br = 'unrescored'
            if ice and gl:
                dicts, lams = ([{}, gl], [cfg['ice_lam'], cfg['gl_lam']]) if cfg['gl_ice'] is None else \
                              ([ice, gl], [cfg['gl_ice'], cfg['gl_lam']])
                o = gl_fuse.rerank_multi(cands, keys, scores, [K.form2(s) for s in cands], dicts, lams, top_n=len(cands))
                cands, br = [cands[i] for i in o], 'glacier'
            elif cfg['ice_fallback'] and ice and not gl:
                o = ice_fuse.rerank(cands, keys, scores, [K.form2(s) for s in cands], ice, lam=cfg['ice_lam'], top_n=len(cands))
                cands, br = [cands[i] for i in o], 'ice_fallback'
            res = safe_extend(K, cands[:25], cands[25:] + original) if cands else original
        branches[mid] = br
        out[mid] = hedge(K, res, R['eng'].get(mid), cfg['hedge'])
    return out, branches


def hedge(K, toks, e, mode):
    toks = [t for t in toks if t]
    if mode is None:
        return toks
    keyp = lambda s: (K.score_key(s) or None) if isinstance(s, str) and s and s != 'CCO' and Chem.MolFromSmiles(s) is not None else None
    r1 = toks[0] if toks else 'CCO'
    l = [x for x in (e.get('smiles') or []) if x and x != 'CCO'] if e else []
    if not l:
        return toks
    kx = keyp(l[0])
    if kx is None or kx == keyp(r1):
        return toks
    new = toks[:1] + [l[0]] + toks[1:] if mode == 'rank2' else [l[0]] + toks
    seen, out = set(), []
    for t in new:
        k = keyp(t) or t
        if k not in seen:
            seen.add(k); out.append(t)
    return out[:25]


def rr(K, lists, truth):
    out = {}
    for mid, t in truth.items():
        r = 0.0
        for i, s in enumerate(lists.get(mid, [])[:25], 1):
            if t and K.official(s) == t:
                r = 1.0 / i; break
        out[mid] = r
    return pd.Series(out)


def boot(d, n=2000):
    d = np.asarray(d)
    bs = [d[np.random.default_rng(i).integers(0, len(d), len(d))].mean() for i in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_run(run):
    inp = json.load(open(os.path.join(run, 'e004_candidate_inputs.json')))
    so = os.path.join(run, 'sim_out')
    ref = pd.read_csv(os.path.join(run, 'submission_reference.csv'), dtype=str, keep_default_na=False)
    read = lambda f: pd.read_csv(os.path.join(so, f), dtype=str, keep_default_na=False) if os.path.exists(os.path.join(so, f)) else None
    sub = lambda df: None if df is None else dict(zip(df.molecule_id, df.smiles.str.split(';')))
    return dict(prefusion=inp['prefusion'], eng=inp['engine2'], reference=dict(zip(ref.molecule_id, ref.smiles.str.split(';'))),
                ice=json.load(open(os.path.join(so, 'ice_scores.json'))), gl=json.load(open(os.path.join(so, 'gl_scores.json'))),
                q=pd.read_parquet(os.path.join(so, 'queries.parquet')),
                final=sub(read('sub_final.csv')), final_A=sub(read('sub_final_A.csv')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--gl-fuse', required=True); ap.add_argument('--ice-fuse', required=True)
    ap.add_argument('--chem', required=True); ap.add_argument('--cache', default=None)
    ap.add_argument('--b-gl-ice', type=float, default=None, help='the A/B arm B weight to check (gl_ice05 runs: 0.5)')
    ap.add_argument('--variants', action='store_true', help='also score the built-in final-stage variants (paired vs E006)')
    a = ap.parse_args()
    gl_fuse, ice_fuse, chem = _load('gl_fuse', a.gl_fuse), _load('fuse', a.ice_fuse), _load('chem', a.chem)
    R = load_run(a.run)
    K = Cache(a.cache, chem)
    truth = {str(m): K.official(s) for m, s in zip(R['q'].molecule_id, R['q'].smiles)}
    try:
        checks = [('E006 (arm A)', E006, R['final_A'] or R['final'])]
        if a.b_gl_ice is not None:
            checks.append((f'gl_ice {a.b_gl_ice:g} (arm B)', dict(E006, gl_ice=a.b_gl_ice), R['final']))
        base = None
        for name, cfg, saved in checks:
            got, br = final_lists(R, K, gl_fuse, ice_fuse, cfg)
            same = sum(got[m] == saved.get(m) for m in got)
            print(f'reproduce {name}: {same}/{len(got)} lists identical; branches {pd.Series(br).value_counts().to_dict()}')
            s = rr(K, got, truth)
            print(f'   MRR {s.mean():.4f} (saved file: {rr(K, saved, truth).mean():.4f})')
            base = s if base is None else base
        if a.variants:
            print('\n' + f'{"variant":34s} MRR     delta   95% CI             better/worse')
            for name, cfg in VARIANTS:
                s = rr(K, final_lists(R, K, gl_fuse, ice_fuse, dict(E006, **cfg))[0], truth).reindex(base.index)
                d = (s - base).values; lo, hi = boot(d)
                print(f'{name:34s} {s.mean():.4f} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {int((d > 0).sum())}/{int((d < 0).sum())}')
        return R, K, gl_fuse, ice_fuse, truth, base
    finally:
        K.save()


if __name__ == '__main__':
    main()
