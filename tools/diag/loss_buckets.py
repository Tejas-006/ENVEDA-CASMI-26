# Where the pipeline loses on a sim2 run: rank of the truth in each saved list (usage: loss_buckets.py <sim_out dir>/)
import json, sys, collections
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); TE = rdMolStandardize.TautomerEnumerator()
d = sys.argv[1]
q = pd.read_parquet(d + 'queries.parquet')
def form(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s and s != 'CCO' else None
    return CalcMolFormula(m) if m is not None else None
def okey(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.RemoveStereochemistry(m)
    try: return Chem.MolToInchiKey(TE.Canonicalize(m))[:14]
    except Exception: return None
def rank_of(lst, tf, tk):
    for i, s in enumerate(lst, 1):
        if form(s) == tf and okey(s) == tk: return i, len(lst)
    return None, len(lst)
L = {n: json.load(open(d + n + '.json')) for n in ['pc_lists', 'eng_lists', 'base_pre_ice', 'gl_scores', 'ice_scores']}
subA = pd.read_csv(d + 'sub_final_A.csv').set_index('molecule_id').smiles
rows = []
for mid, smi, mode in zip(q.molecule_id, q.smiles, q.ion_mode):
    tf, tk = form(smi), okey(smi); m = str(mid)
    fin, _ = rank_of(str(subA.get(mid, '')).split(';')[:25], tf, tk)
    pcr, pcn = rank_of(L['pc_lists'].get(m, {}).get('pc', []), tf, tk)
    b = L['base_pre_ice'].get(m, []); b = b[0] if b and isinstance(b[0], list) else b
    br, bn = rank_of(b, tf, tk)
    er, en = rank_of(L['eng_lists'].get(m, {}).get('smiles', []), tf, tk)
    gl = L['gl_scores'].get(m) or {}; ice = L['ice_scores'].get(m) or {}
    in_gl = any(form(s) == tf and okey(s) == tk for s in gl); in_ice = any(form(s) == tf and okey(s) == tk for s in ice)
    tops = [s for s in b[:5]] + L['pc_lists'].get(m, {}).get('pc', [])[:5]
    rows.append(dict(mid=m, mode=mode, final=fin, pc=pcr, pc_n=pcn, base=br, base_n=bn, eng=er, eng_n=en,
                     gl_cov=bool(gl), in_gl=in_gl, in_ice=in_ice,
                     top1_formula_ok=(form(str(subA.get(mid, '')).split(';')[0]) == tf)))
r = pd.DataFrame(rows); r.to_csv(d + 'buckets.csv', index=False)
n = len(r); hit = r.final.notna()
print(f'n={n}  final hit@25 {hit.mean():.3f}  top1 {(r.final == 1).mean():.3f}  top1 formula right {r.top1_formula_ok.mean():.3f}')
print(f'GLACIER covers {r.gl_cov.mean():.3f} of molecules; truth among GL-scored cands {r.in_gl.mean():.3f}, ICE-scored {r.in_ice.mean():.3f}')
anyl = r.pc.notna() | r.base.notna() | r.eng.notna()
print(f'truth in any saved list {anyl.mean():.3f}: PubChem list {r.pc.notna().mean():.3f} (mean len {r.pc_n.mean():.0f}), base {r.base.notna().mean():.3f} (len {r.base_n.mean():.0f}), engine-2 {r.eng.notna().mean():.3f} (len {r.eng_n.mean():.0f})')
miss = ~hit
print(f'misses {miss.sum()}: truth in some list but not final top 25: {(miss & anyl).sum()}; in no list: {(miss & ~anyl).sum()}')
print('PubChem-list rank of truth (all q):', pd.cut(r.pc.dropna(), [0, 1, 5, 10, 25, 50, 100, 1e9]).value_counts().sort_index().to_dict())
print('PubChem rank when final missed:', pd.cut(r.pc[miss].dropna(), [0, 1, 5, 10, 25, 50, 100, 1e9]).value_counts().sort_index().to_dict())
print('final rank of hits:', pd.cut(r.final.dropna(), [0, 1, 3, 5, 10, 25]).value_counts().sort_index().to_dict())
print('max PubChem list length:', r.pc_n.max(), ' share with list len >= 25:', (r.pc_n >= 25).mean().round(3))
