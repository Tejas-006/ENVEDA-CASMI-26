"""Build a leaderboard-track submission notebook = the 0.401 baseline + selected upgrades, and its push folder.

usage: python tools/make_lb_notebook.py <baseline.ipynb> <baseline kernel-metadata.json> <out_dir> [--cmatch] [--no-fusion] [--v6]
       [--gl-ice LAM] [--frag-gap] [--slug S]

The output contains the baseline's code: never commit it (this repo is public).
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_sim2_notebook import (code_cell, find_cell, apply_v6, apply_pc_sim, apply_multi_ce, is_e006,  # noqa: E402
                                _arm_gl_ice, _arm_frag_gap, apply_strong_lib_gentle, CELL6_MARK, CELL9_MARK)
import make_kaggle_push  # noqa: E402

NO_FUSION = "ENG = {}   # v5i: skip the final engine-2 fusion (sim2 external, 103 q: final 0.319 vs pre-fusion 0.362)\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('meta')
    ap.add_argument('out_dir')
    ap.add_argument('--cmatch', action='store_true')
    ap.add_argument('--no-fusion', action='store_true')
    ap.add_argument('--v6', action='store_true')
    ap.add_argument('--pc-sim', type=int, default=0, help='step 5: PubChem top-K through ICE/GL and into the v6 window')
    ap.add_argument('--multi-ce', action='store_true', help='score merged multi-energy spectra against merged predictions')
    ap.add_argument('--gl-ice', type=float, default=0.0, help='E006: add ICEBERG at this weight to the GLACIER-only re-scores')
    ap.add_argument('--frag-gap', action='store_true', help='E006: MetFrag-lite re-rank where no simulator has scores')
    ap.add_argument('--strong-lib-gentle', action='store_true', help='E006: PubChem gentle slots also for lib_max >= 0.9')
    ap.add_argument('--slug', default='casmi26-v5-lb')
    a = ap.parse_args()
    nb = json.load(open(a.baseline))
    cells = nb['cells']
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    extra, notes = [], []
    if a.cmatch:
        i6 = find_cell(cells, CELL6_MARK)
        cells.insert(i6 + 1, code_cell(open(os.path.join(HERE, 'lb', 'cmatch_rerank.py')).read()))
        extra.append('t3jas06/casmi26-cmatch')
        notes.append('v5b CMatch within-formula re-rank (CM_LAM=1.0)')
    if a.no_fusion:
        i9 = find_cell(cells, CELL9_MARK)
        cells.insert(i9, code_cell(NO_FUSION))
        notes.append('v5i no final engine-2 fusion')
    if a.v6:
        assert not a.no_fusion, '--v6 rebuilds the final fusion; it cannot be combined with --no-fusion'
        apply_v6(cells)
        notes.append('v6 GLACIER-first re-scoring + identity fusion + metric-safe tail')
    if a.pc_sim:
        assert a.v6, '--pc-sim needs --v6'
        apply_pc_sim(cells, a.pc_sim)
        notes.append(f'PubChem top-{a.pc_sim} through ICEBERG/GLACIER and into the v6 window')
    if a.multi_ce:
        apply_multi_ce(cells)
        notes.append('merged multi-energy spectra scored against merged ICEBERG/GLACIER predictions')
    if a.gl_ice or a.frag_gap or a.strong_lib_gentle:
        assert is_e006(cells), '--gl-ice / --frag-gap / --strong-lib-gentle edit E006 cells'
    if a.gl_ice:
        _arm_gl_ice(cells, a.gl_ice)
        notes.append(f'ICEBERG at weight {a.gl_ice:g} next to GLACIER in the final re-scores')
    if a.frag_gap:
        _arm_frag_gap(cells)
        notes.append('MetFrag-lite re-rank for molecules without simulator scores (v7c)')
    if a.strong_lib_gentle:
        apply_strong_lib_gentle(cells)
        notes.append('PubChem gentle slots also for strong library matches (lib_max >= 0.9)')
    assert notes, 'choose at least one upgrade'
    base_label = ('E006 fork (antriksharya/casmi26-e006-e2rank2, LB 0.442)'
                  if any('E006: window=40' in ''.join(c['source']) for c in cells) else 'Baseline v4n (LB 0.401)')
    if base_label.startswith('E006'):
        assert not (a.v6 or a.no_fusion or a.cmatch or a.pc_sim), 'these options target the v4n baseline'
    cells.insert(0, dict(cell_type='markdown', metadata={}, source=[
        f'# {a.slug}\n', base_label + ' + ' + '; '.join(notes) + '\n']))
    nbp = os.path.join(tempfile.mkdtemp(), f'{a.slug}.ipynb')
    json.dump(nb, open(nbp, 'w'), indent=1)
    make_kaggle_push.main(a.meta, nbp, a.out_dir, a.slug, a.slug, ','.join(extra))


if __name__ == '__main__':
    main()
