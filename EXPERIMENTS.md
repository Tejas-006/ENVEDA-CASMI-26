# Experiments log (sim2 validation gate; see CLAUDE.md)

## Ground rules

- Comply with competition rules sections 4 and 6 and the code-competition limits (CLAUDE.md): internet off, <= 9 h run,
  keep the pipeline under ~8.5 h (current estimate ~6.5 h on ~400 hidden molecules).
- sim2 = practice test on public MassSpecGym spectra of compounds NOT in competition train (`--source external`).
  The old train-holdout sim scored 0.93-0.97 vs LB 0.401 (models had memorised those compounds) and is not used for decisions.
- Only ~125 unseen questions exist (232 of 28,929 MassSpecGym compounds, minus tautomers of train compounds):
  sim2 noise ~ +/-0.035, about the same as the public LB (~130 molecules). Treat both as independent checks.
- Keep a change if: sim2 dMRR >= +0.01 AND public LB drop <= 0.02 AND runtime fits. Model-vs-model choices
  (CMatch v1 vs v1.1 vs FLARE-style) are decided on `val isomer MRR` (~2,000 held-out train compounds; far less noisy).
- Two attempts per direction, then move on. No tuning constants against the public LB.
- Final picks: one public-LB-best submission + one sim2-best submission.

## Where the score is lost (sim2 external baseline, 103 q, pre-fusion)

| Situation | Share | Fixed by |
|---|---|---|
| Truth not in the candidate pool at all | ~29% | Stage C (bigger pool), Stage D (generative proposals) |
| In pool, but the v1 engine never retrieves it (pool 71% vs engine recall 52%) | ~19% | **Step 4 dense retrieval (moved up)** |
| Retrieved, not first | ~23% | Stages A/B (CMatch, FLARE-style), ICEBERG/GLACIER |
| First | ~29% | — |

Also: the final engine-2 fusion (cell 9) lowers sim2 MRR 0.362 -> 0.319 (top1 29% -> 21%): test v5i = no fusion.

## Target model (what the final submission should look like)

1. Candidates: baseline library/pool/PubChem channel (v4n) **+ generated natural-product library (C)** **+ generative
   proposals for weak-library-match molecules only (D)**.
2. Ranking: baseline v1 engine + LightGBM ranker (unchanged; retraining needs >= 500 unseen questions).
3. Look-alike re-scoring: ICEBERG + GLACIER **+ CMatch (A; instrument-robust v1.1 from B; FLARE-style v2 or a fusion of
   both if it wins head-to-head)**.
4. Fusion: hand-set weights and PubChem gate today; **learned stacker (E)** once enough unseen questions exist.
5. Engine-2 reciprocal-rank fusion (unchanged) -> submission.csv.

## Consolidated plan

Projections are rough (public benchmarks, not measurements) and overlap. Path: 0.401 -> ~0.41 (A/B) -> ~0.42-0.43 (C)
-> ~0.44-0.47 (D), if each stage passes its gate.

| Stage | Version | What | Keep gate | If it fails | Status |
|---|---|---|---|---|---|
| 0 | v5a | sim2 baseline on all unseen MassSpecGym questions | — | — | done: 103 q, final 0.319 / pre-fusion 0.362 |
| 0b | v5i | Drop the final engine-2 fusion (cell 9) | LB >= -0.02 (sim2 already +0.043) | keep fusion | notebook ready: make_lb_notebook.py --no-fusion |
| A | v5b | CMatch v1: own contrastive matcher (spectrum transformer + fingerprint MLP, trained vs same-formula decoys), within-formula re-rank before ICEBERG/GLACIER at fixed weight 1.0 | val isomer MRR >= 0.30; sim2 >= +0.01; LB >= -0.02 | A2 | training running; submission notebook ready (tools/make_lb_notebook.py --cmatch) |
| A2 | v5b2 | FLARE-style CMatch v2: peak-to-atom late-interaction scoring, same data/split; also try fusing v1+v2 | beats v1 on val isomer MRR, then same gates as A | drop the contrastive line | planned (build when A finishes, or earlier for a head-to-head) |
| B | v5b1 | CMatch v1.1 instrument robustness: instrument-label dropout, balanced timsTOF sampling, peak/intensity/m-z augmentation, cross-instrument same-molecule pairing (arXiv 2602.00547), per-instrument val scores | timsTOF val isomer MRR >= +0.02 vs v1 without hurting others; LB >= -0.02 | drop | planned; first check whether test.parquet has `instrument_type` (CMatch v1 assumes non-timsTOF if it's missing) |
| C | v5h | Bigger candidate pool: generated natural-product library (MassKG-style; ours if theirs isn't public) | truth-in-pool +5 points; sim2 not worse; runtime fits | 2nd try: different library/generator | planned |
| D | v5f | Generative proposals + GLMR-style re-rank for weak-library-match molecules only | START if after C truth-not-in-pool >= 20%, or two ranking upgrades in a row each < +0.01. KEEP if >= +0.015 and <= 1.5 h extra | drop | candidates need public code + weights: MARLIN has none (checked 2026-10-09); check DiffMS, FlowMS, MS-GPT |
| E | v5c | Learned stacker replacing ICE_LAM/GL_LAM/ALPHA/KRR and the PubChem gate | needs >= 500 unseen questions (add CASMI 2016/2017/2022 answer sets etc.); sim2 >= +0.01 | keep fixed weights | waiting on data |
| backlog | v5e | Retrain FPNet with a listwise/contrastive loss | needs ranker retraining on unseen questions (same data gate as E) | — | deprioritised |
| backlog | v5g | Formula-confidence feature (SIRIUS/BUDDY-style) | as a re-rank term: sim2 >= +0.01 | — | deprioritised |
| on hold | v5d | ICEBERG -> MARASON | — | — | code public (coleygroup/ms-pred, MIT), no released weights |

Sources: FLARE (bioRxiv 2026.01.27.702086), MSAlign (arXiv 2605.19752), GLMR (arXiv 2511.06259),
cross-instrument contrastive DG (arXiv 2602.00547), MS-GPT (arXiv 2607.23607), MARLIN (arXiv 2607.04774),
MassKG (PMC11415640), MassSpecGym in the Wild (arXiv 2606.19624).

## Post-CMatch plan: per-case experts, architecture changes, branching steps (2026-10-09)

### Experts per failure case, stitched by a router
| Case (pilot share) | Expert (research) | Availability |
|---|---|---|
| 1. Truth not in pool (~27%) | Generative proposals (MS-GPT formula-given, MARLIN formula-free, DiffMS/FlowMS) + generated NP library (MassKG-style) | MARLIN no code; others to check |
| 2. In pool, outside top 25 (~18%) | Dense retrieval with a contrastive matcher over the whole mass window (FLARE, MSAlign, JESTR/MVP) | our CMatch / FLARE-style build |
| 3. In top 25, not first (~28%) | Isomer experts: ICEBERG, GLACIER, FLARE peak-to-atom, new cross-encoder (MARASON: no weights) | partly in pipeline |

Router (heuristic first, learned in step 9), generalising the existing PubChem gate:
strong library match -> keep ranker order | big ranker margin -> light re-rank | close same-formula race -> full isomer
experts | weak everything (low lib_max, low best CMatch cosine) -> generative + expanded-pool proposals in fixed slots.

### Architecture changes
| Change | Type | Basis |
|---|---|---|
| Fingerprint MLP -> GNN (GIN / graph transformer) molecule encoder | replace | JESTR, MVP, FLARE |
| Global cosine -> late interaction (peak<->atom max-sim) | replace | FLARE |
| Peak subformula-annotation tokens | add | MIST / MIST-CF |
| Pairwise mass-difference attention bias | add | DreaMS, MassFormer |
| Masked-peak self-supervised pretraining on 2.5M train spectra | add | DreaMS (own, dependency-free) |
| Instrument embedding -> gradient-reversal instrument-adversarial layer | replace/remove | cross-instrument DG (arXiv 2602.00547) |
| Cross-attention to nearest library spectra + structures | add | MARASON, MS2Query |
| Auxiliary fingerprint/formula heads | add | MIST, multi-task |
| Hard negatives mined from the pipeline's own top-25 lists | replace | retrieval practice |
| Cross-encoder re-ranker (spectrum tokens x candidate graph tokens) on top 25 only | new | two-stage retrieval (GLMR-style) |

### Steps
| Step | Do | Decide by | If yes | If no | Cost |
|---|---|---|---|---|---|
| 1 | CMatch v1 results | val isomer MRR >= 0.30 | 2 | 1b | — |
| 1b | Rebuild: GNN molecule encoder + masked-peak pretraining | val >= 0.30 | 2 | drop contrastive line -> 5 | ~12 GPU h |
| 2 | Submit CMatch (v5-lb) | sim2 >= +0.01 and LB >= -0.02 | keep -> 3 | 2b | 1 submission |
| 2b | CMatch as a whole-list term (judged on sim2 only, no LB tuning) | sim2 >= +0.01 | keep -> 3 | drop re-rank use; keep model for 4 | — |
| 3 | Head-to-head v1.1 (instrument tricks + adversarial) vs v2 (FLARE-style) vs fusion | val isomer MRR, overall + timsTOF | submit winner if >= +0.02 val over v1 | keep v1 | ~9-12 GPU h each |
| 4 | Dense retrieval: matcher top-K over whole mass window, union into candidates | sim2 recall@25 +3 pts; runtime <= +20 min | keep -> 5 | drop | ~1 day code |
| 5 | Pool expansion: generated NP library | truth-in-pool +5 pts; sim2 not worse | keep -> 6 | 2nd try other generator -> 6 | GPU h |
| 6 | Router v1 (heuristic) | sim2 >= +0.01 or same score, less runtime | keep -> 7 | always-on flow | small |
| 7 | Generative proposals for likely-not-in-pool molecules | START if not-in-pool >= 20% after 5, or 3 and 4 each < +0.01. KEEP if >= +0.015 and <= 1.5 h | keep -> 8 | drop | public-weights model needed |
| 8 | Cross-encoder re-ranker on top 25 | sim2 >= +0.01; runtime <= +45 min | keep -> 9 | drop | ~12 GPU h |
| 9 | Learned stacker + router (E) | >= 500 unseen questions, then sim2 >= +0.01 | keep | heuristic router | data |
| 10 | Final picks: LB-best + sim2-best | — | — | — | — |

Runtime rule: total <= ~8.5 h (now ~6.5 h); if a step doesn't fit, the router skips ICEBERG/GLACIER on solved molecules first.

## Results

Metric: MRR@25 (InChIKey-14), GroupKFold by molecule, paired bootstrap 95% CI vs the baseline.

| Date | Change | Commit | sim2 MRR (Δ [95% CI]) | Class-1 Δ | Class-2 Δ | Runtime Δ / mol | Public LB | Decision |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | Baseline v4n (current submission) | n/a | see v5a row (2026-10-10) | | | | 0.401 | reference |
| 2026-10-09 | sim2 external pilot (MassSpecGym, 11 unseen q), baseline v4n unchanged | n/a | final MRR 0.322 (pre-fusion 0.238); top1 0.27, recall@25 0.55; truth in pool 73% | — | — | ~ | 0.401 | realistic (train-holdout gave 0.93); only 232 of 28,929 MassSpecGym compounds unseen → ~125 usable questions; running all next |
| 2026-10-10 | v5a sim2 external baseline (103 unseen MassSpecGym q), v4n unchanged | n/a | final MRR 0.319 (top1 0.21, top5 0.44, recall@25 0.55); pre-fusion 0.362 (top1 0.29); in-pool 0.42 final / 0.50 pre; truth in pool 71%; engine candidate recall 52% | — | — | ~36 s/mol + ~20 min fixed | 0.401 | baseline set. Final engine-2 fusion costs -0.043 on sim2 -> test v5i (no fusion) on LB |
