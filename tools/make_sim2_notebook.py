"""Build the sim2 notebook from the submitted baseline notebook.

usage: python tools/make_sim2_notebook.py <baseline.ipynb> <out.ipynb> [--shard N] [--size N]

The output contains the baseline's code, so it must never be committed (this repo is public).
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(HERE, 'sim2')

SMOKE_MARK = '# SMOKE switch'
ENG_MARK = '# Our engine (two-ranker'
CELL6_MARK = '# all molecules -> base lists'
CELL6_SCORE = 'score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))'
CELL7_MARK = '# ICEBERG post-ranker'
CELL8_MARK = '# gated PubChem merge'
CELL9_MARK = 'ALPHA, KRR ='
GL_BUDGET_LIT = ', 1.0, 4000'


def code_cell(src):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=src.splitlines(keepends=True))


def read(name):
    return open(os.path.join(SIM_DIR, name)).read()


def build_setup(source, shard=0, size=500):
    assert source in ('train', 'external', 'enveda180')
    s = read('sim_setup.py').replace('SHARD, SHARD_SIZE = 0, 500', f'SHARD, SHARD_SIZE = {shard}, {size}')
    s = s.replace("SOURCE = 'train'", f"SOURCE = '{source}'")
    assert s.count('# @@SELECT@@\n') == 1
    return s.replace('# @@SELECT@@\n', read(f'select_{source}.py'))


def find_cell(cells, mark):
    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and mark in ''.join(c['source'])]
    assert len(hits) == 1, f'expected exactly one cell containing {mark!r}, found {len(hits)}'
    return hits[0]


def patch_cell6(src):
    lines = src.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if l.strip() == CELL6_SCORE]
    assert len(hits) == 1, 'ranker scoring line not found in the base-list cell'
    i = hits[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i + 1, f'{indent}_simdump(mid, C, np.concatenate([X, F], 1), list(FEATURES) + list(names), score)\n')
    out = ''.join(lines)
    assert out.count('(gi + 1) % 25 == 0') == 1, 'progress-print line not found in the base-list cell'
    return out.replace('(gi + 1) % 25 == 0', '(gi + 1) % 5 == 0')


ENG_HOOK = 'E.find = _find\\n'
ENG_HOOK_NEW = ENG_HOOK + '    if os.environ.get("SIM2_ENG_PATCH"): exec(open(os.environ["SIM2_ENG_PATCH"]).read())\\n'


def patch_engine2(src):
    # the engine-2 runner is a string literal in this cell: hook a patch file in right after its find() override
    assert src.count(ENG_HOOK) == 1, 'engine-2 runner hook point not found'
    return src.replace(ENG_HOOK, ENG_HOOK_NEW)


V6_SNAPSHOT = ("import copy\nV6_BASE_PRE = copy.deepcopy(BASE)   # v6: ranker lists (top-60) before ICEBERG/GLACIER\n")
V6_RESTORE = ("# v6: keep the base lists in ranker order; GLACIER/ICEBERG act on the fused list in the v6 final pass\n"
              "for _m in list(BASE):\n"
              "    _p = V6_BASE_PRE.get(_m)\n"
              "    if _p is not None:\n"
              "        BASE[_m] = (list(_p[0])[:25], list(_p[1])[:25], _p[2])\n")
V6_PREFUSION = "import shutil\nshutil.copy('submission.csv', 'sub_v4_prefusion.csv')   # v6: lists before the final fusion\n"


def apply_v6(cells):
    """GLACIER-first re-scoring, identity fusion and metric-safe tail (tools/lb/v6_final.py)."""
    i7 = find_cell(cells, CELL7_MARK)
    cells.insert(i7, code_cell(V6_SNAPSHOT))
    cells.insert(i7 + 2, code_cell(V6_RESTORE))
    i9 = find_cell(cells, CELL9_MARK)
    cells.insert(i9, code_cell(V6_PREFUSION))
    cells.insert(i9 + 2, code_cell(open(os.path.join(HERE, 'lb', 'v6_final.py')).read()))


PC_SIM_HOOK = '    cands = {m: c for m, c in cands.items() if c}\n'
PC_SIM_ADD = (
    "    if PC_SIM_K and isinstance(PC, dict):   # EXECUTION_PLAN step 5: PubChem top-K of EVERY molecule through ICE/GL\n"
    "        n_pc = 0\n"
    "        for m in mids:\n"
    "            p = PC.get(m)\n"
    "            if p and p.get('pc'):\n"
    "                before = len(cands.get(m, []))\n"
    "                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + [str(x) for x in p['pc'][:PC_SIM_K]]))\n"
    "                n_pc += len(cands[m]) - before\n"
    "        print('PC_SIM: PubChem candidates added to ICE/GL input:', n_pc, flush=True)\n")


PC_MARK = '# PubChem-only channel'
PC_K_LIT = '\\nK = 25\\n'          # the PubChem core is a string literal in this cell: its K line, escaped
PC_DEEP_SAVE = (
    "# sim2 diagnostic (--pc-depth): keep the deep PubChem lists for analysis, give the pipeline its usual top 25\n"
    "import json as _j, shutil as _sh\n"
    "if isinstance(PC, dict) and PC:\n"
    "    _sh.copy('/kaggle/working/pc_lists.json', os.path.join(SIM_OUT, 'pc_lists_deep.json'))\n"
    "    for _p in PC.values():\n"
    "        if isinstance(_p, dict):\n"
    "            for _k in ('pc', 'pc_fz', 'pc_keys'):\n"
    "                if isinstance(_p.get(_k), list):\n"
    "                    _p[_k] = _p[_k][:25]\n"
    "    _j.dump(PC, open('/kaggle/working/pc_lists.json', 'w'))\n")


def apply_pc_depth(cells, depth):
    """Diagnostic only: the PubChem channel lists `depth` candidates instead of 25; the deep lists are saved to
    sim_out/pc_lists_deep.json and the pipeline still sees the top 25 (same results as a normal run, more runtime)."""
    i3 = find_cell(cells, PC_MARK)
    src3 = ''.join(cells[i3]['source'])
    assert src3.count(PC_K_LIT) == 1, 'PubChem core K literal not found'
    cells[i3] = code_cell(src3.replace(PC_K_LIT, f'\\nK = {int(depth)}\\n'))
    cells.insert(i3 + 1, code_cell(PC_DEEP_SAVE))


MCE_HOOKS = ('budget_s=ICE_BUDGET,', 'budget_s=GL_BUDGET,')


def apply_multi_ce(cells):
    """Merged multi-energy spectra are scored against merged predictions at each listed energy (tools/lb/multi_ce.py):
    a wrapper passed to run_ice / run_gl as `python=`."""
    i7 = find_cell(cells, CELL7_MARK)
    src7 = ''.join(cells[i7]['source'])
    for h in MCE_HOOKS:
        assert src7.count(h) == 1, f'runner call {h!r} not found in the ICEBERG cell'
        src7 = src7.replace(h, h + ' python=MCE_PYTHON,')
    cells[i7] = code_cell(src7)
    cells.insert(i7, code_cell(open(os.path.join(HERE, 'lb', 'multi_ce.py')).read()))


BOTH_CE_HOOK = 'def gl_of(m):\n'
BOTH_CE_ADD = (
    "# sim2 (--both-ce): the same runner input scored again with the multi-energy wrapper; saved for offline replay only\n"
    "ICE_SCORES_MCE, GL_SCORES_MCE = {}, {}\n"
    "try:\n"
    "    ICE_SCORES_MCE = ice_fuse.run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work_mce', device='cuda',\n"
    "                                      budget_s=ICE_BUDGET, site='/kaggle/working/ice_site', python=MCE_PYTHON)\n"
    "    gc.collect(); torch.cuda.empty_cache()\n"
    "    GL_SCORES_MCE = gl_fuse.run_gl(GL_PKG, items, workdir='/kaggle/working/gl_work_mce', device='cuda',\n"
    "                                   budget_s=GL_BUDGET, site='/kaggle/working/ice_site',\n"
    "                                   wheels=os.path.join(ICE_PKG, 'wheels'), python=MCE_PYTHON)\n"
    "    print('MCE scores (sim2 extra): ICE', len(ICE_SCORES_MCE), 'GL', len(GL_SCORES_MCE), f'{time.time() - T0:.0f}s')\n"
    "except Exception as e:\n"
    "    print('MCE scoring failed (sim2 extra only):', repr(e))\n")


def apply_both_ce(cells):
    """sim2 only: score the simulator input twice (runners as shipped -> used by the pipeline; multi-energy wrapper ->
    saved as ice/gl_scores_mce.json), so the replay harness can compare both on the same run."""
    i7 = find_cell(cells, CELL7_MARK)
    src7 = ''.join(cells[i7]['source'])
    assert src7.count(BOTH_CE_HOOK) == 1, 'gl_of() definition not found in the ICEBERG cell'
    cells[i7] = code_cell(src7.replace(BOTH_CE_HOOK, BOTH_CE_ADD + BOTH_CE_HOOK))
    cells.insert(i7, code_cell(open(os.path.join(HERE, 'lb', 'multi_ce.py')).read()))


def apply_pc_sim(cells, k):
    """Step 5: score each molecule's PubChem top-k with ICEBERG/GLACIER (adds scores only; the baseline's own
    re-ranking is unchanged) and let those k candidates into the v6 final re-score window (V6_PC_WINDOW)."""
    i7 = find_cell(cells, CELL7_MARK)
    src7 = ''.join(cells[i7]['source'])
    assert src7.count(PC_SIM_HOOK) == 1, 'ICE candidate-dict line not found in the ICEBERG cell'
    cells[i7] = code_cell(src7.replace(PC_SIM_HOOK, PC_SIM_ADD + PC_SIM_HOOK))
    cells.insert(i7, code_cell(f'PC_SIM_K = V6_PC_WINDOW = {int(k)}   # EXECUTION_PLAN step 5\n'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('out')
    ap.add_argument('--pc-sim', type=int, default=0, help='step 5: PubChem top-K through ICE/GL and into the v6 window')
    ap.add_argument('--pc-depth', type=int, default=0, help='diagnostic: save PubChem lists this deep (pipeline keeps 25)')
    ap.add_argument('--multi-ce', action='store_true', help='score merged multi-energy spectra against merged predictions')
    ap.add_argument('--both-ce', action='store_true', help='also save multi-energy simulator scores for offline replay')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--size', type=int, default=500)
    ap.add_argument('--source', choices=['train', 'external', 'enveda180'], default='external')
    ap.add_argument('--cmatch', action='store_true', help='insert the CMatch re-rank after BASE0 (needs casmi26-cmatch)')
    ap.add_argument('--v6', action='store_true', help='GLACIER-first re-scoring + identity fusion + metric-safe tail')
    a = ap.parse_args()

    nb = json.load(open(a.baseline))
    cells = nb['cells']
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None

    setup = build_setup(a.source, a.shard, a.size)
    cells[find_cell(cells, SMOKE_MARK)] = code_cell(setup)

    i2 = find_cell(cells, ENG_MARK)
    cells[i2] = code_cell(patch_engine2(''.join(cells[i2]['source'])))

    i6 = find_cell(cells, CELL6_MARK)
    cells[i6] = code_cell(patch_cell6(''.join(cells[i6]['source'])))
    cells.insert(i6 + 1, code_cell('import copy\nBASE0 = copy.deepcopy(BASE)   # sim2: ranker lists before ICEBERG/GLACIER\n'))
    if a.cmatch:
        cells.insert(i6 + 2, code_cell(open(os.path.join(HERE, 'lb', 'cmatch_rerank.py')).read()))

    i7 = find_cell(cells, CELL7_MARK)
    cells.insert(i7 + 1, code_cell('BASE_POST_ICE = copy.deepcopy(BASE)   # sim2: lists after ICEBERG/GLACIER, before the PubChem merge\n'))

    i8 = find_cell(cells, CELL8_MARK)
    cells.insert(i8 + 1, code_cell("import shutil\nshutil.copy('submission.csv', os.path.join(SIM_OUT, 'sub_v4.csv'))   # sim2: before final fusion\n"))

    if a.v6:
        apply_v6(cells)
    if a.pc_sim:
        assert a.v6, '--pc-sim needs --v6 (the PubChem candidates join the v6 final window)'
        apply_pc_sim(cells, a.pc_sim)
    if a.pc_depth:
        apply_pc_depth(cells, a.pc_depth)
    if a.multi_ce:
        apply_multi_ce(cells)
    if a.both_ce:
        assert not a.multi_ce, '--both-ce keeps the shipped runners for the pipeline; do not combine with --multi-ce'
        apply_both_ce(cells)
    find_cell(cells, CELL9_MARK)
    i7 = find_cell(cells, CELL7_MARK)
    src7 = ''.join(cells[i7]['source'])
    assert src7.count(GL_BUDGET_LIT) == 1, 'GLACIER budget literal not found'
    cells[i7] = code_cell(src7.replace(GL_BUDGET_LIT, ', 1.0, SIM_GL_BUDGET'))
    cells.append(code_cell(read('sim_finish.py')))

    # a timestamped progress line before every step after the setup cell, so a stalled run shows where it is
    i_setup = find_cell(cells, '# sim2 setup')
    out = cells[:i_setup + 1]
    for c in cells[i_setup + 1:]:
        if c['cell_type'] == 'code':
            first = ''.join(c['source']).strip().splitlines()[0][:70].replace("'", '')
            out.append(code_cell(f"sim2_mark('{first}')\n"))
        out.append(c)
    nb['cells'] = cells = out

    cells.insert(0, dict(cell_type='markdown', metadata={}, source=[
        f'# casmi26-sim2-build ({a.source} questions, shard {a.shard}, {a.size} queries)\n',
        'Generated by tools/make_sim2_notebook.py from the submitted baseline. Runs the unchanged pipeline on held-out\n',
        'training compounds and saves labelled outputs to /kaggle/working/sim_out. Save the output as a PRIVATE dataset.\n']))
    json.dump(nb, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
