"""Build the one-time MassSpecGym download notebook + kaggle push folder (internet ON, CPU, private).

usage: python tools/make_msg_download_notebook.py <owner> <out_dir>
After it runs: Output -> New Dataset 'casmi26-massspecgym' (Private).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = 'casmi26-massspecgym-download'


def main(owner, out_dir):
    src = open(os.path.join(HERE, 'ext', 'download_msg.py')).read()
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
              cells=[dict(cell_type='markdown', metadata={}, source=[
                  '# casmi26-massspecgym-download\n', 'Downloads MassSpecGym (MIT) and converts it to the competition format.\n',
                  'Then: Output -> New Dataset `casmi26-massspecgym` (Private).\n']),
                  dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=src.splitlines(keepends=True))])
    meta = dict(id=f'{owner}/{SLUG}', title=SLUG, code_file=f'{SLUG}.ipynb', language='python', kernel_type='notebook',
                is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True, dataset_sources=[],
                competition_sources=[], kernel_sources=[], model_sources=[])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(nb, open(os.path.join(out_dir, f'{SLUG}.ipynb'), 'w'), indent=1)
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
