# Phase 2 P3: merge / tail variants vs E006 + strong-lib gentle; choose on Enveda-180 s0 + MassSpecGym, confirm on s1.
# usage: python tools/replay/search_phase2.py <scratch dir with repo/, casmi_src/ (chem.py) and e006_keys.json>; run from kaggle-ml
import sys, json, os, numpy as np
S = sys.argv[1]; sys.path.insert(0, S + '/repo/tools/replay')
import replay_e006 as RE, replay_e006_merge as RM
gl, ice, chem = RE._load('gl_fuse', 'ext_glacier'), RE._load('fuse', 'ext_iceberg'), RE._load('chem', S + '/casmi_src')
K = RE.Cache(S + '/e006_keys.json', chem)
BASE = dict(RM.MERGE, strong_lib='gentle')
G7, A7 = [4, 7, 10, 13, 16, 19, 22], [2, 4, 6, 8, 10, 12, 14]
G10, A10 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
V = [('strong-lib: aggressive slots', dict(strong_slots=RM.SLOTS_AGG)), ('strong-lib: 7 gentle slots', dict(strong_slots=G7)),
     ('7 PubChem slots everywhere', dict(slots_gentle=G7, slots_agg=A7)), ('10 PubChem slots everywhere', dict(slots_gentle=G10, slots_agg=A10)),
     ('PubChem early in tail fill', dict(tail_pc_early=True)), ('cell-24 fusion 25', dict(fuse_n=25)), ('cell-24 fusion 60', dict(fuse_n=60)),
     ('7 slots + PubChem early in tail', dict(slots_gentle=G7, slots_agg=A7, tail_pc_early=True))]
TEST_LIKE = ['[M+H]+', '[M-H]-', '[M+H]+;[M-H]-']
runs = {}
for run in ['ab_out_gl-ice05', 'ab_out_msg', 'ab_out_e180s1']:            # choose on s0 + MSG, confirm on s1
    R = RE.load_run(run); inp = json.load(open(os.path.join(run, 'e004_candidate_inputs.json')))
    R.update(base_post=json.load(open(os.path.join(run, 'sim_out', 'base_post_ice.json'))), pc=inp['pubchem'], base_tail=inp['base_tail'])
    truth = {str(m): K.official(s) for m, s in zip(R['q'].molecule_id, R['q'].smiles)}
    chk = RM.run_chain(R, K, gl, ice, RM.MERGE)[2]; saved = R['final_A'] or R['final']
    assert all(chk[m] == saved.get(m) for m in chk), f'{run}: default config no longer reproduces the saved run'
    tl = sorted(set(R['q'].molecule_id.astype(str)[R['q'].adducts.isin(TEST_LIKE)]))
    runs[run] = (R, truth, tl, RE.rr(K, RM.run_chain(R, K, gl, ice, BASE)[2], truth))
print('default config still reproduces all three saved runs exactly; base below = E006 + strong-lib gentle\n')
hdr = ['s0 all', 's0 test-like', 'MSG all', 'MSG test-like', 's1 all (confirm)', 's1 test-like (confirm)']
print(f'{"variant vs E006 + strong-lib":32s}' + ''.join(f'{h:>27s}' for h in hdr))
for name, v in V:
    row = f'{name:32s}'
    for run, (R, truth, tl, base) in runs.items():
        s = RE.rr(K, RM.run_chain(R, K, gl, ice, dict(BASE, **v))[2], truth).reindex(base.index)
        for ix in (list(base.index), tl):
            d = (s.reindex(ix) - base.reindex(ix)).values; lo, hi = RE.boot(d, 1000)
            row += f'  {d.mean():+.4f}[{lo:+.3f},{hi:+.3f}]{int((d>0).sum()):2d}/{int((d<0).sum()):<2d}'
    print(row)
K.save()
