"""Replay of E006 from the gated PubChem merge onwards (cells 22 -> 24 -> 27 -> 31 -> 32), for PubChem-gate experiments.

Builds on replay_e006.py (cells 31/32) and adds the merge (cell 22), the two-engine fusion + GLACIER pass (cell 24) and the
metric-safe tail (cell 27), each checked against the run's own saved intermediate (e004 'prefusion',
submission_reference.csv, sub_final_A.csv) before any variant is scored. Extra inputs: sim_out/base_post_ice.json
(BASE as the merge saw it) and e004_candidate_inputs.json 'pubchem' / 'base_tail'.

usage: python tools/replay/replay_e006_merge.py <run dir> [<run dir> ...] --gl-fuse DIR --ice-fuse DIR --chem DIR [--cache keys.json]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_e006 as RE

# cell 6 / cell 22 constants
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]
MERGE = dict(lib_tau=0.9, rel_th=600.0, s_tau=6.0, pop_tau=5.0, rel_promote=250.0, slots_agg=SLOTS_AGG, slots_gentle=SLOTS_GENTLE,
             promote_all=False, pc_hedge=None, strong_lib='untouched', strong_slots=None, fuse_n=40, tail_pc_early=False)
# strong_lib: untouched | gentle (strong_slots or slots_gentle, no promotion); fuse_n: cell-24 fusion length;
# tail_pc_early: cell-27 tail fill takes PubChem right after the fused overflow (E006: last)


def merge(base, base_keys, pc, pc_keys, slots, n=25):
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out


def promote(base, base_keys, pc, pc_keys, slots, n=25):
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]


def merged_lists(R, m):
    out = {}
    for mid in R['prefusion']:
        smis, keys, lib_max = (R['base_post'].get(mid) or ([], [], 0.0))[:3]
        p = R['pc'].get(mid)
        if p is None or not p.get('pc'):
            final = smis
        elif lib_max >= m['lib_tau'] and m['strong_lib'] == 'untouched':
            final = smis
        elif lib_max >= m['lib_tau'] and m['strong_lib'] == 'gentle':
            final = merge(smis, keys, p['pc'], p['pc_keys'], m['strong_slots'] or m['slots_gentle'])
        else:
            bp = p.get('best_pool_fz')
            top_fz = p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]
            rel = top_fz - bp if bp is not None and np.isfinite(bp) else 1e9
            slots = m['slots_agg'] if rel > m['rel_th'] else m['slots_gentle']
            S, pop_top = p.get('S'), float(p.get('top_pop') or 0.0)
            if m['promote_all'] or (S is not None and float(S) > m['s_tau'] and pop_top >= m['pop_tau']) or rel > m['rel_promote']:
                final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
            else:
                final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
        out[mid] = list(final[:25]) if final else ['CCO']
    return out


def fuse2(K, v_smis, e_smis, e_keys, alpha=0.6, krr=3.0, n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = K.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (krr + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + alpha / (krr + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


def cell24(R, K, gl_fuse, ice_fuse, pre, fuse_n=40):
    base_sub, overflow = {}, {}
    for mid, s in pre.items():
        vs = [x for x in s if x and x != 'CCO']
        e = R['eng'].get(mid)
        if e and e['smiles']:
            fsm, fk, fsc = fuse2(K, vs, e['smiles'], e['keys'], n=fuse_n)
            ice = R['ice'].get(mid, {}) or {}
            if ice:
                try:
                    forms = [K.form2(x) for x in fsm]
                    o = ice_fuse.rerank(fsm, fk, fsc, forms, {}, lam=1.0, top_n=len(fsm))       # CELL19_ICE = False
                    g = R['gl'].get(mid, {}) or {}
                    g = g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}
                    if g:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [{}, g], [1.0, 1.0], top_n=len(fsm))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    print('fused ICE rerank failed', mid, repr(ex))
            overflow[mid] = list(fsm[25:])
            vs = fsm[:25]
        base_sub[mid] = vs[:25] if vs else ['CCO']
    return base_sub, overflow


def cell27(R, K, base_sub, overflow, pc_early=False):
    ref = {}
    for mid, original in base_sub.items():
        original = [s.strip() for s in original if s.strip()]
        clean = RE.safe_extend(K, original, [])
        ov, bt = list(overflow.get(mid, [])), list(R['base_tail'].get(mid, []))
        en, pcl = list((R['eng'].get(mid) or {}).get('smiles', [])), list((R['pc'].get(mid) or {}).get('pc', []))
        extras = ov + pcl + bt + en if pc_early else ov + bt + en + pcl
        ref[mid] = RE.safe_extend(K, clean, extras)
    return ref


def pc_hedge(K, toks, p, rank):
    """PubChem's #1 inserted at `rank` when it is not already in the top `rank` (metric identity dedup)."""
    if not p or not p.get('pc'):
        return toks
    x = p['pc'][0]; kx = K.identity(x)
    if kx is None or kx in {K.identity(t) for t in toks[:rank]}:
        return toks
    new = toks[:rank - 1] + [x] + [t for t in toks[rank - 1:] if K.identity(t) != kx]
    return new[:25]


def run_chain(R, K, gl_fuse, ice_fuse, m, final_cfg=None):
    pre = merged_lists(R, m)
    base_sub, overflow = cell24(R, K, gl_fuse, ice_fuse, pre, m['fuse_n'])
    ref = cell27(R, K, base_sub, overflow, m['tail_pc_early'])
    R2 = dict(R, prefusion=pre, reference=ref)
    fin, _ = RE.final_lists(R2, K, gl_fuse, ice_fuse, final_cfg or RE.E006)
    if m['pc_hedge']:
        fin = {mid: pc_hedge(K, l, R['pc'].get(mid), m['pc_hedge']) for mid, l in fin.items()}
    return pre, ref, fin


VARIANTS = [('promote PubChem #1 whenever merged', dict(promote_all=True)),
            ('promote when rel > 100', dict(rel_promote=100.0)), ('promote when rel > 0', dict(rel_promote=0.0)),
            ('promote when S > 4 & pop >= 2', dict(s_tau=4.0, pop_tau=2.0)),
            ('aggressive slots always', dict(rel_th=-1e18)), ('gentle slots always', dict(rel_th=1e18)),
            ('merge even when lib_max >= 0.9', dict(lib_tau=1.01)),
            ('lib_max >= 0.9: gentle slots only', dict(strong_lib='gentle')),
            ('lib >= 0.9 gentle + PC hedge rank 5', dict(strong_lib='gentle', pc_hedge=5)),
            ('merge always + PC hedge rank 5', dict(lib_tau=1.01, pc_hedge=5)),
            ('PubChem #1 hedge at rank 2', dict(pc_hedge=2)), ('PubChem #1 hedge at rank 3', dict(pc_hedge=3)),
            ('PubChem #1 hedge at rank 5', dict(pc_hedge=5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--gl-fuse', required=True); ap.add_argument('--ice-fuse', required=True)
    ap.add_argument('--chem', required=True); ap.add_argument('--cache', default=None)
    a = ap.parse_args()
    gl_fuse, ice_fuse, chem = RE._load('gl_fuse', a.gl_fuse), RE._load('fuse', a.ice_fuse), RE._load('chem', a.chem)
    K = RE.Cache(a.cache, chem)
    table = {}
    try:
        for run in a.runs:
            R = RE.load_run(run)
            inp = json.load(open(os.path.join(run, 'e004_candidate_inputs.json')))
            R.update(base_post=json.load(open(os.path.join(run, 'sim_out', 'base_post_ice.json'))), pc=inp['pubchem'], base_tail=inp['base_tail'],
                     saved_prefusion=inp['prefusion'], saved_reference=R['reference'])
            truth = {str(m): K.official(s) for m, s in zip(R['q'].molecule_id, R['q'].smiles)}
            pre, ref, fin = run_chain(R, K, gl_fuse, ice_fuse, MERGE)
            saved_final = R['final_A'] or R['final']
            ok = (sum(pre[m] == R['saved_prefusion'][m] for m in pre), sum(ref[m] == R['saved_reference'][m] for m in ref),
                  sum(fin[m] == saved_final.get(m) for m in fin))
            print(f'{run}: reproduce merge {ok[0]}/{len(pre)}, tail {ok[1]}/{len(ref)}, final {ok[2]}/{len(fin)}')
            base = RE.rr(K, fin, truth)
            for name, v in VARIANTS:
                s = RE.rr(K, run_chain(R, K, gl_fuse, ice_fuse, dict(MERGE, **v))[2], truth).reindex(base.index)
                d = (s - base).values; lo, hi = RE.boot(d, 1000)
                table.setdefault(name, {})[run] = f'{d.mean():+.4f} [{lo:+.3f},{hi:+.3f}] {int((d > 0).sum())}/{int((d < 0).sum())}'
        print(f'\n{"vs E006":38s}' + ''.join(f'{r:>32s}' for r in a.runs))
        for name, row in table.items():
            print(f'{name:38s}' + ''.join(f'{row.get(r, ""):>32s}' for r in a.runs))
    finally:
        K.save()


if __name__ == '__main__':
    main()
