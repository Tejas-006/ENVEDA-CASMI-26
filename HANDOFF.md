# Handoff: ENVEDA CASMI 2026, state as of 2026-10-20

Everything a new session needs to continue. Read `CLAUDE.md` first: its competition rules and validation gate
apply to every change. Results live in `EXPERIMENTS.md` (one row per experiment, newest last), the impact-ordered
plan in `EXECUTION_PLAN.md`, the Kaggle workflow in `NOTEBOOKS.md`, external resources and licences in
`EXTERNAL_RESOURCES.md`.

## 1. Where things stand

| Submission | Public LB | Notes |
|---|---|---|
| **E006 fork** (`t3jas06/casmi26-e006-base` v1) | **0.442 (best)** | public antriksharya/casmi26-e006-e2rank2, unused hengck23 input removed. **The baseline since 2026-10-18** |
| E006 + GLACIER/ICEBERG 0.5 + frag_gap (`casmi26-e006-glice-frag` v1) | 0.435 | practice: +0.011 / +0.016 (all Enveda-180 q), +0.020 / +0.027 on test-like q; MassSpecGym neutral |
| E006 + strong-library gentle merge (`casmi26-e006-strong-lib` v1, submission 57404158) | pending | practice (exact replay): +0.003 / +0.003 / +0.010; test-like +0.006 / +0.005 / +0.008, all CIs > 0 |
| v6 (our own lineage), v6.1, v6.2 | 0.425, 0.399, 0.420 | superseded by the E006 rebase |

- **Final picks (deadline 14 Dec 2026), current thinking:** pick 1 = the LB-safe line (E006 fork, or E006 +
  strong-lib if it scores ≥ ~0.43); pick 2 = the practice-best line (E006 + gl_ice 0.5 + frag_gap + strong-lib,
  replay +0.026 / +0.032 / +0.015 on test-like questions). The public LB is ~130 of ~400 molecules (±0.035), so every
  LB difference so far is inside noise.
- **Branch:** `claude/gracious-maxwell-04dmc2` (PR https://github.com/Tejas-006/ENVEDA-CASMI-26/pull/2).
- **Repo is public.** Never commit notebooks (`*.ipynb` gitignored), forked notebook code, Competition Data or
  anything derived from it (practice questions, spectrum caches, notebook outputs). Third-party modules used by the
  replay (`gl_fuse.py`, `fuse.py`, `casmi/chem.py`) stay local.

## 2. How we work

- **A local Claude Code session on Tejas's PC drives Kaggle** with the CLI in `%USERPROFILE%\kaggle-ml\.venv`
  (OAuth as `t3jas06`): push notebooks, poll status, download logs/outputs, submit. Push freely; **every competition
  submission needs Tejas's explicit OK.** Batch GPU limit: 2 sessions (queued ones count). CPU notebooks don't.
- Push folders live in `kaggle-ml\<name>` (`kernel-metadata.json` + `.ipynb`). Always copy E006's pinned docker
  image (`e006base/kernel-metadata.json`): the ICEBERG/GLACIER subprocesses need its Python 3.12 + RDKit 2025.03.6.
- `kaggle kernels output` sometimes returns a 502 HTML page instead of a file: check the first byte and retry.
- A push run uses the visible stand-in test: 12 molecules processed (all strong library matches, they overlap
  train), 388 placeholder rows. Health lines to check: `errors 0`, `ICE meta` / `GL meta` status ok,
  `output_rows 400`, `invalid_removed 0`, `E006 ENGINE-2 CELL APPLIED`.
- Hidden-test runtime of E006 is ~3 h of the 9 h limit; ICEBERG and GLACIER budgets are not binding (ICE ~1,150 s,
  GL ~400 s per 350 molecules vs budgets 5,400 / 4,000 s).

## 3. Validation toolkit (what to use for which question)

| Question | Tool | Cost |
|---|---|---|
| Does a change of E006's merge / fusion / re-score / tail help? | `tools/replay/replay_e006_merge.py <run dirs>` (cells 22 → 24 → 27 → 31 → 32) and `replay_e006.py` (cells 31/32 only, `--variants`). **Exact**: reproduce every saved list on all three runs | ~1 min, CPU |
| Change that needs new simulator scores, candidates or spectra | `tools/make_sim2_notebook.py e006base/casmi26-e006-base.ipynb <out.ipynb> --source enveda180 --shard S --size 400 --ab ARM` (arms: multi_ce, gl_ice05, frag_gap, gl_frag; add new ones in `AB_ARMS`) or `--source external` (MassSpecGym) | ~3 GPU h per A/B |
| Where are we losing? | `tools/diag/loss_buckets.py`, `tools/diag/gate_study.py` on a run's `sim_out/` | minutes |
| LB notebook | `tools/make_lb_notebook.py e006base/casmi26-e006-base.ipynb e006base/kernel-metadata.json <out_dir> --slug S [--gl-ice 0.5] [--frag-gap] [--strong-lib-gentle]` | 20 min push |

Saved practice runs (download with `kaggle kernels output`; local copies in `kaggle-ml\ab_out_*`):

| Run | Questions | E006 MRR | Notes |
|---|---|---|---|
| `casmi26-sim2-e180-gl-ice05` (→ `ab_out_gl-ice05`) | Enveda-180 shard 0, 400 | 0.267 (test-like 216 q: 0.472) | timsTOF, class 2: 0% truth in pool, reached only via PubChem |
| `casmi26-sim2-e180s1-gl-frag` (→ `ab_out_e180s1`) | Enveda-180 shard 1, 350 | 0.250 (test-like 195 q: 0.419) | same |
| `casmi26-sim2-msg-e006-gl-frag` (→ `ab_out_msg`) | MassSpecGym unseen, 103 | 0.538 | Orbitrap/QTOF, 71% truth in pool: **the in-pool guard** |

Rules learnt the hard way:
- **Judge every change on Enveda-180 AND MassSpecGym.** Anything that trusts PubChem more at the top gains on
  Enveda-180 and loses on MassSpecGym (e.g. PubChem #1 at rank 2: +0.03 / −0.025).
- **Engine 2 never holds an Enveda-180 truth**, so down-weighting it always "wins" there (v5i: practice +0.043,
  LB −0.024). Don't tune engine-2 weights on practice sets.
- **35% of Enveda-180 questions are [M+NH4]+ and E006 solves ~0 of them** (PubChem ranking and GLACIER fail on
  non-protonated adducts; masses are correct). The test is ~79% [M+H]+, 16% [M-H]-, 0.3% [M+NH4]+ by spectra, so
  report Enveda-180 on test-like questions ([M+H]+ / [M-H]-) as well; an NH4 fix would not move the LB.
- Simulator re-weighting: three LB tests away from GLACIER-dominant (v6.1, v6.2, gl_ice05) all came out −0.005 to
  −0.026, each within noise. gl_ice 0.5 still has the best timsTOF evidence (two shards, 61/5); keep it for the
  practice-best final pick, not the LB-safe one.

## 4. Kaggle assets

| Asset | What |
|---|---|
| `t3jas06/casmi26-e006-base` | E006 fork (LB 0.442). `e006base/` holds its notebook + metadata: the input to every builder |
| `t3jas06/casmi26-e006-glice-frag`, `casmi26-e006-strong-lib` | LB variants (0.435, pending) |
| `t3jas06/casmi26-enveda180-questions` | Enveda-180 timsTOF practice questions (750 compounds not in train); output used as kernel source |
| `t3jas06/casmi26-massspecgym` (private dataset) | MassSpecGym (HF d2e86d0) for the MassSpecGym practice set |
| `t3jas06/casmi26-newer-libraries` | counts of newer GNPS / MassBank / MoNA spectra not in train (2,248 usable compounds, 0 timsTOF); private parquet |
| `t3jas06/casmi26-ms2ds-package`, `casmi26-ms2ds-eval` | MS2DeepScore 2.11 offline package + its evaluation (parked) |
| `t3jas06/casmi26-ce-eval` | timsTOF isomer bench (GLACIER + 0.5 ICEBERG +0.025 on natural products) |
| older | v4n baseline, v6/v6.1/v6.2 LB notebooks, CMatch datasets, sim2-build runs (see EXPERIMENTS.md) |

## 5. History (details in EXPERIMENTS.md)

| Date | Step | Result |
|---|---|---|
| 09-16→09-30 | Fork of a public notebook → v4n | LB 0.401 |
| 10-02→10-11 | sim2 practice test; CMatch v1/v2; no-fusion | CMatch LB 0.372, no-fusion LB 0.377 (sim2 said +0.043): dropped |
| 10-13 | v6 (GLACIER-first, identity fusion, metric-safe tail) | LB 0.425 |
| 10-15→10-17 | Official scorer + precursor fix in sim2, replay harness, v6.1 / v6.2 | LB 0.399 / 0.420: simulator / PubChem changes failed |
| 10-17→10-18 | Enveda-180 timsTOF practice set; licence check; **rebase onto public E006** | 750 questions; fork LB 0.442 |
| 10-18→10-19 | A/Bs on Enveda-180: multi-energy (neutral), gl_ice05 (+0.011), frag_gap (+0.004), shard-1 replication (+0.016) | gl_ice05 + frag_gap LB 0.435 |
| 10-19 | Desk checks: DiffMS usable but ~+0.004 expected (parked), MS-BART no licence, FlowMS no code; newer libraries low value | — |
| 10-19 | Exact E006 replay (final stage, then from the merge); PubChem-gate variants | strong-lib gentle merge: only change positive on all three sets |
| 10-19 | MS2DeepScore 2.11 offline + analog re-rank | negative-mode +0.004 / +0.007, unguarded: parked |
| 10-20 | Diagnostics: per-adduct (NH4 artefact), budgets not binding, all 7 lost PubChem-#1 [M+H]+ answers are strong-library molecules | strong-lib recovers all 7 |

## 6. Next steps

See `EXECUTION_PLAN.md` → "Phase 2 (from 2026-10-20)" for the scoped list with gates. Immediate:
1. Read the score of submission 57404158 (strong-lib). Keep it as the LB-safe pick if ≥ ~0.43.
2. Before the next push, re-push the unchanged E006 fork once to see whether the 2026-10-19 popularity-prior update
   changes its output (EXTERNAL_RESOURCES.md).

## 7. Known issues and open questions

- **Popularity-prior dataset updated 2026-10-19 18:47 UTC:** all submissions so far used the previous version; the
  next push attaches the new one. Compare like with like.
- Public LB ±0.035: use it for yes/no breakage checks, not for small effects.
- MANIFEST-style hashes of every file the scoring notebook loads are not kept yet (winner obligations, Rules 2.8).
- MVP licence question for the host still open (only matters if MVP is ever used).

## 8. Links

- Repo branch: https://github.com/Tejas-006/ENVEDA-CASMI-26/tree/claude/gracious-maxwell-04dmc2
- Pull request: https://github.com/Tejas-006/ENVEDA-CASMI-26/pull/2
- E006 base notebook: https://www.kaggle.com/code/antriksharya/casmi26-e006-e2rank2
- Project-steps doc: https://claude.ai/code/artifact/9263a6f1-2769-4837-9805-dd9472d01ba2
