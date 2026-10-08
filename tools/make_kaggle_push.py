"""Make a `kaggle kernels push` folder: a notebook + kernel-metadata.json that copies every input of a source kernel.

usage: python tools/make_kaggle_push.py <source kernel-metadata.json> <notebook.ipynb> <out_dir> <slug> <title> [extra,datasets]

The pushed kernel is always private with internet off; inputs, GPU and environment pinning come from the source.
"""
import json
import os
import shutil
import sys

KEEP = ['language', 'kernel_type', 'enable_gpu', 'enable_tpu', 'dataset_sources', 'competition_sources',
        'kernel_sources', 'model_sources', 'docker_image', 'docker_image_pinning_type', 'machine_shape']


def main(src_meta, notebook, out_dir, slug, title, extra=''):
    src = json.load(open(src_meta))
    owner = src['id'].split('/')[0]
    meta = {k: src[k] for k in KEEP if k in src}
    meta.update(id=f'{owner}/{slug}', title=title, code_file=os.path.basename(notebook),
                is_private=True, enable_internet=False)
    meta.setdefault('enable_gpu', True)
    meta['dataset_sources'] = list(meta.get('dataset_sources', [])) + [d for d in extra.split(',') if d]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(notebook, os.path.join(out_dir, os.path.basename(notebook)))
    json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:7])
