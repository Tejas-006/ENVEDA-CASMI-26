# Multi-energy fix for the ICEBERG / GLACIER runners (2026-10-17), inserted before the ICEBERG cell.
# Test spectra listing several collision energies ([20, 40, 60, 80]) are MERGED acquisitions (Data tab). The runners
# compare such a spectrum with ONE prediction at mean(collision_energy_ev). This wrapper keeps the runners unchanged
# but swaps three functions: a multi-energy spectrum is compared with the runner's own merge() of predictions at each
# listed energy (the same merge it already uses when the energy is missing). Single-energy spectra score exactly as
# before. Hooked in through run_ice / run_gl's `python=` argument; any wrapper failure leaves ICE/GL empty, which the
# pipeline already handles as "keep the ranker order".
import os, stat, sys as _sys
MCE_WRAP = '/kaggle/working/mce_wrap.py'
MCE_PYTHON = '/kaggle/working/mce_python.sh'
_MCE_SRC = r'''
import sys, importlib.util
runner_path = sys.argv[1]
spec = importlib.util.spec_from_file_location('_mce_runner', runner_path)
R = importlib.util.module_from_spec(spec); sys.modules['_mce_runner'] = R
spec.loader.exec_module(R)
np = R.np


def _ces(sp):
    """Distinct bucketed energies of a merged eV spectrum (>= 2), else None."""
    if sp.get('ce_units') == 'NCE':
        return None
    try:
        v = [float(x) for x in (sp.get('ce_ev') or [])]
    except (TypeError, ValueError):
        return None
    b = sorted({R.ce_bucket(x) for x in v if np.isfinite(x)})
    return tuple(b) if len(b) > 1 else None


def molecule_plan(mol):
    out = []
    for sp in mol.get('spectra') or []:
        if not R.is_covered(sp):
            continue
        qm, qp = R.prep_spectrum(sp)
        multi = _ces(sp)
        out.append((qm, qp, sp['adduct'], R.instr_token(sp), multi if multi else R.ce_bucket(R.spectrum_ce(sp))))
    return out


def _ce_list(ceb):
    if isinstance(ceb, tuple):
        return ceb
    return (ceb,) if np.isfinite(ceb) else tuple(R.SUM_CE)


def molecule_jobs(plan, cands):
    J = set()
    for _, _, ad, ins, ceb in plan:
        for c in cands:
            for ce in _ce_list(ceb):
                J.add((c, ad, ins, ce))
    return J


def score_molecule(plan, cands, cache):
    res, merged = {}, {}
    for c in cands:
        sims = []
        for qm, qp, ad, ins, ceb in plan:
            if not isinstance(ceb, tuple) and np.isfinite(ceb):
                pr = cache.get((c, ad, ins, ceb))
            else:
                ces = _ce_list(ceb); sk = (c, ad, ins, ces)
                if sk not in merged:
                    parts = [cache.get((c, ad, ins, ce)) for ce in ces]
                    parts = [p for p in parts if p is not None]
                    merged[sk] = R.merge(parts) if len(parts) == len(ces) else None
                pr = merged[sk]
            if pr is None:
                continue
            pm, pp = R.prep_query(pr[0], pr[1], R.FLOOR, R.TOPK, R.POWER, R.ENTW)
            s = R.entropy_sim(qm, qp, pm, pp, R.TOL, R.PPM)
            if np.isfinite(s):
                sims.append(s)
        res[c] = float(np.mean(sims)) if sims else None
    return res


R.molecule_plan, R.molecule_jobs, R.score_molecule = molecule_plan, molecule_jobs, score_molecule
print('[mce] multi-energy plan/jobs/score active for', runner_path, flush=True)
sys.argv = [runner_path] + sys.argv[2:]
sys.exit(R.main())
'''
open(MCE_WRAP, 'w').write(_MCE_SRC)
open(MCE_PYTHON, 'w').write(f'#!/bin/sh\nexec "{_sys.executable}" "{MCE_WRAP}" "$@"\n')
os.chmod(MCE_PYTHON, os.stat(MCE_PYTHON).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
print('multi-CE runner wrapper ready:', MCE_PYTHON)
# merged spectra need up to 4 predictions instead of 1 (many energies are shared with the molecule's single-energy
# spectra and reused): give ICEBERG more time on the real rerun only (sim2 keeps its scaled budget)
if IS_RERUN and 'SIM_OUT' not in globals():
    ICE_BUDGET = max(ICE_BUDGET, 6600)
print('ICE_BUDGET', ICE_BUDGET)
