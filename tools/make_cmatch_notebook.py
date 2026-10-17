"""Build the CMatch training notebook + kaggle push folder.

usage: python tools/make_cmatch_notebook.py <baseline kernel-metadata.json> <out_dir>

Inputs: the competition, the RDKit wheel the submission uses, and the COCONUT/ChEBI key files (only to reproduce
the sim2 hold-out exactly). Same Docker image as the submission so the saved model loads there.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = 'casmi26-cmatch-train'
DATASETS = ['metric/rdkit-2026-3-3-wheel', 'prvsiyan/coconut-casmi26-candidates', 'prvsiyan/chebi-lipidmaps-casmi26']

SETUP = '''# CMatch training: setup (RDKit wheel as in the submission, shared cmatch module)
import os, sys, glob, subprocess
def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w]
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
print('rdkit wheel', whl[0], r.returncode, r.stderr[-300:])
os.makedirs('/kaggle/working/cmatch_src', exist_ok=True)
open('/kaggle/working/cmatch_src/cmatch.py', 'w').write(CMATCH_SRC)
sys.path.insert(0, '/kaggle/working/cmatch_src')
import cmatch, torch, rdkit
print('rdkit', rdkit.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
'''


def code_cell(src):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=src.splitlines(keepends=True))


def main(src_meta, out_dir):
    cm = open(os.path.join(HERE, 'cmatch', 'cmatch.py')).read()
    train = open(os.path.join(HERE, 'cmatch', 'train_cmatch.py')).read()
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')),
              cells=[dict(cell_type='markdown', metadata={}, source=['# casmi26-cmatch-train\n',
                          'Contrastive spectrum-molecule matcher trained on competition train data (sim2 hold-out excluded).\n',
                          'Save /kaggle/working/cmatch_out as a PRIVATE dataset named casmi26-cmatch.\n']),
                     code_cell('CMATCH_SRC = ' + repr(cm) + '\n'),
                     code_cell(SETUP),
                     code_cell(train)])
    src = json.load(open(src_meta))
    meta = {k: src[k] for k in ['language', 'kernel_type', 'docker_image', 'machine_shape'] if k in src}
    meta.update(id=f"{src['id'].split('/')[0]}/{SLUG}", title=SLUG, code_file=f'{SLUG}.ipynb', is_private=True,
                enable_gpu=True, enable_tpu=False, enable_internet=False, dataset_sources=DATASETS,
                competition_sources=src['competition_sources'], kernel_sources=[], model_sources=[])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(nb, open(os.path.join(out_dir, f'{SLUG}.ipynb'), 'w'), indent=1)
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
