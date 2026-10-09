"""Build a leaderboard-track submission notebook = the 0.401 baseline + selected upgrades, and its push folder.

usage: python tools/make_lb_notebook.py <baseline.ipynb> <baseline kernel-metadata.json> <out_dir> --cmatch

The output contains the baseline's code: never commit it (this repo is public).
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_sim2_notebook import code_cell, find_cell, CELL6_MARK  # noqa: E402
import make_kaggle_push  # noqa: E402

SLUG = 'casmi26-v5-lb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('meta')
    ap.add_argument('out_dir')
    ap.add_argument('--cmatch', action='store_true')
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
    assert notes, 'choose at least one upgrade'
    cells.insert(0, dict(cell_type='markdown', metadata={}, source=[
        '# casmi26-v5-lb\n', 'Baseline v4n (LB 0.401) + ' + '; '.join(notes) + '\n']))
    nbp = os.path.join(tempfile.mkdtemp(), f'{SLUG}.ipynb')
    json.dump(nb, open(nbp, 'w'), indent=1)
    make_kaggle_push.main(a.meta, nbp, a.out_dir, SLUG, SLUG, ','.join(extra))


if __name__ == '__main__':
    main()
