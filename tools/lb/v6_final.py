# v6 final pass (LB track), inserted after the baseline's fusion cell; it rebuilds submission.csv from the same run.
# Ideas follow public CASMI26 notebooks (credited in EXPERIMENTS.md); this code is our own.
#  1. Identity fusion: one vote per metric identity per engine in the reciprocal-rank fusion (no duplicate votes).
#  2. GLACIER-first re-scoring of the fused top V6_WINDOW: z(fusion) + GL_LAM z(GLACIER) where GLACIER scored the
#     molecule; ICEBERG only where GLACIER has no scores (GLACIER runs on [M+H]+ spectra only). Base lists stay in
#     ranker order (see the restore cell after the ICEBERG/GLACIER cell).
#  3. Metric-safe tail: drop candidates the official metric treats as the same molecule, then fill empty slots
#     from the fused overflow, the pre-fusion list, the ranker's top-60, engine 2 and the PubChem list.
# Any failure for a molecule keeps the baseline fusion's row for it.
# v6.1 (2026-10-15): GLACIER weight 1.0 -> 0.5 and window 40 -> 25 in this final pass only. Offline replay on sim2 v11
# (tools/replay/replay_final.py, accurate precursors): weights 0.25-0.75 all beat 1.0 by +0.02-0.03 with CI > 0; the
# full weight let GLACIER swap truths from the top 25 to slots 26-40. v6 = V6_WINDOW 40, V6_GL_LAM = GL_LAM (1.0).
V6_WINDOW, V6_GL_LAM = 25, 0.5
import collections, json
from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _v6_CMF
RDLogger.DisableLog('rdApp.*')
_v6_taut = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=200_000)
def v6_identity(smiles):
    """The official metric's key (stereo removed, then canonical tautomer, InChIKey first block) + heavy atoms."""
    try:
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles and smiles != 'CCO' else None
        if m is None:
            return None
        heavy = tuple(sorted(collections.Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() != 'H').items()))
        Chem.RemoveStereochemistry(m)
        return (Chem.MolToInchiKey(_v6_taut.Canonicalize(m))[:14], heavy)
    except (ValueError, RuntimeError, TypeError):
        return None


def v6_unique(seq, limit=25):
    out, seen = [], set()
    for s in seq:
        k = v6_identity(str(s).strip())
        if k is not None and k not in seen:
            seen.add(k); out.append(str(s).strip())
            if len(out) == limit:
                break
    return out


def v6_fusion(first, second, alpha, offset, limit):
    score, rep = {}, {}
    for cands, w in ((first, 1.0), (second, alpha)):
        seen = set()
        for s in cands:
            k = v6_identity(s)
            if k is None or k in seen:
                continue
            seen.add(k)
            score[k] = score.get(k, 0.0) + w / (offset + len(seen))
            rep.setdefault(k, s)
    order = sorted(score, key=lambda k: -score[k])[:limit]
    return [rep[k] for k in order], [score[k] for k in order]


def v6_form(s):
    try:
        return _v6_CMF(Chem.MolFromSmiles(s))
    except Exception:
        return s


def v6_scores(d, m):
    v = d.get(str(m), {}) if isinstance(d, dict) else {}
    return v if isinstance(v, dict) and any(x is not None for x in v.values()) else {}


_v6_base = pd.read_csv('submission.csv')                       # baseline fusion output = per-molecule fallback
_v6_pre = pd.read_csv('sub_v4_prefusion.csv')
_v6_pre = dict(zip(_v6_pre.molecule_id.astype(str), _v6_pre.smiles))
_v6_stats = collections.Counter()
_v6_rows = []
for _mid0, _fallback in zip(_v6_base.molecule_id, _v6_base.smiles):
    mid = str(_mid0)
    try:
        first = [s for s in str(_v6_pre.get(mid, _fallback)).split(';') if s and s != 'CCO']
        e = ENG.get(mid) if isinstance(ENG, dict) else None
        e_smis = list(e.get('smiles', [])) if e else []
        if e_smis:
            cands, scs = v6_fusion(first, e_smis, ALPHA, KRR, V6_WINDOW)
        else:
            cands = v6_unique(first, V6_WINDOW); scs = [1.0 / (KRR + r) for r in range(1, len(cands) + 1)]
        keys = [chem.score_key(s) or s for s in cands]
        gl, ice = v6_scores(GL_SCORES, mid), v6_scores(ICE_SCORES, mid)
        if gl and len(cands) > 1:
            o = gl_fuse.rerank_multi(cands, keys, scs, [v6_form(s) for s in cands], [{}, gl], [ICE_LAM, V6_GL_LAM], top_n=len(cands))
            cands = [cands[i] for i in o]; _v6_stats['glacier'] += 1
        elif ice and len(cands) > 1:
            o = ice_fuse.rerank(cands, keys, scs, [v6_form(s) for s in cands], ice, lam=ICE_LAM, top_n=len(cands))
            cands = [cands[i] for i in o]; _v6_stats['iceberg_fallback'] += 1
        else:
            _v6_stats['no_simulator'] += 1
        pre = V6_BASE_PRE.get(_mid0, V6_BASE_PRE.get(mid)) if isinstance(V6_BASE_PRE, dict) else None
        p = PC.get(_mid0, PC.get(mid, {})) if isinstance(PC, dict) else {}
        tail = cands[25:] + first + (list(pre[0]) if pre else []) + e_smis + list((p or {}).get('pc', []))
        out = v6_unique(cands[:25] + tail, 25)
        _v6_stats['filled_slots'] += max(0, len(out) - len(v6_unique(cands[:25], 25)))
        if not out:
            raise ValueError('no valid candidates')
        _v6_stats['changed_top1'] += int(out[0] != str(_fallback).split(';')[0])
        _v6_rows.append((_mid0, ';'.join(out)))
    except Exception as ex:
        _v6_stats['fallback_rows'] += 1
        print('v6 kept baseline row', mid, repr(ex))
        _v6_rows.append((_mid0, _fallback))
_v6_out = pd.DataFrame(_v6_rows, columns=['molecule_id', 'smiles'])
assert len(_v6_out) == len(_v6_base) and _v6_out.molecule_id.is_unique and _v6_out.smiles.notna().all()
assert _v6_out.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
_v6_out.to_csv('submission.csv', index=False)
_v6_stats.update(molecules=len(_v6_out), mols_with_glacier_scores=sum(bool(v6_scores(GL_SCORES, m)) for m in _v6_pre),
                 mols_with_iceberg_scores=sum(bool(v6_scores(ICE_SCORES, m)) for m in _v6_pre))
print('V6 FINAL ' + json.dumps(dict(_v6_stats)))
