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
| 0b | v5i | Drop the final engine-2 fusion (cell 9) | LB >= -0.02 (sim2 already +0.043) | keep fusion | done: LB 0.377 (-0.024) ✗; fusion kept |
| 0c | v6 | Port of public-notebook fixes onto v4n (tools/lb/v6_final.py, `--v6`): base lists in ranker order; fused top-40 re-scored by GLACIER where it has scores, ICEBERG only where GLACIER is blind (non-[M+H]+); one vote per metric identity in the fusion; metric-safe dedupe + tail fill (overflow, pre-fusion, ranker top-60, engine 2, PubChem) | sim2 >= +0.01 (or ~neutral: dedupe/tail can't demote a truth); LB >= -0.02 | keep v4n | done: sim2 0.379 (+0.060), LB **0.425** (+0.024) ✓; **v6 is the new baseline** |
| A | v5b | CMatch v1 (val isomer MRR 0.642 on random-compound val; sim2 -0.015 ✗; LB 0.372 ✗): own contrastive matcher (spectrum transformer + fingerprint MLP, trained vs same-formula decoys), within-formula re-rank before ICEBERG/GLACIER at fixed weight 1.0 | val isomer MRR >= 0.30; sim2 >= +0.01; LB >= -0.02 | A2 | done: sim2 -0.015 ✗, LB 0.372 (-0.029) ✗; v1 re-rank dropped, retrain with scaffold val |
| A2 | v5b2 | FLARE-style CMatch v2: peak-to-atom late-interaction scoring, same data/split; also try fusing v1+v2 | beats v1 on val isomer MRR, then same gates as A | drop the contrastive line | **dropped 2026-10-15**: attempt 2 passed val (0.665) but dense retrieval has no headroom inside +-10 ppm (offline test); two attempts without gain |
| B | v5b1 | CMatch v1.1 instrument robustness: instrument-label dropout, balanced timsTOF sampling, peak/intensity/m-z augmentation, cross-instrument same-molecule pairing (arXiv 2602.00547), per-instrument val scores | timsTOF val isomer MRR >= +0.02 vs v1 without hurting others; LB >= -0.02 | drop | **dropped 2026-10-15** with the CMatch line. Instrument check (2026-10-14): test.parquet has `instrument_type`, all rows `timsTOF`, so the CMatch timsTOF flag is set correctly. But sim2 (MassSpecGym: Orbitrap/QTOF) never tests the timsTOF path |
| C | v5h | Bigger candidate pool: generated natural-product library (MassKG-style; ours if theirs isn't public) | truth-in-pool +5 points; sim2 not worse; runtime fits | 2nd try: different library/generator | planned |
| D | v5f | Generative proposals + GLMR-style re-rank for weak-library-match molecules only | START if after C truth-not-in-pool >= 20%, or two ranking upgrades in a row each < +0.01. KEEP if >= +0.015 and <= 1.5 h extra | drop | candidates need public code + weights: MARLIN has none (checked 2026-10-09); check DiffMS, FlowMS, MS-GPT |
| E | v5c | Learned stacker replacing ICE_LAM/GL_LAM/ALPHA/KRR and the PubChem gate | needs >= 500 unseen questions (add CASMI 2016/2017/2022 answer sets etc.); sim2 >= +0.01 | keep fixed weights | waiting on data |
| backlog | v5e | Retrain FPNet with a listwise/contrastive loss | needs ranker retraining on unseen questions (same data gate as E) | — | deprioritised |
| backlog | v5g | Formula-confidence feature (SIRIUS/BUDDY-style) | as a re-rank term: sim2 >= +0.01 | — | deprioritised |
| dropped | v5d | ICEBERG -> MARASON | — | — | dropped 2026-10-11 by decision: no released weights; training it ourselves isn't worth the GPU days |

v6 sources (ideas only; our code): public Kaggle notebooks obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail (metric dedupe + tail, reported LB 0.425 -> 0.433), flexonafft/casmi26-public-baseline-adaptation-experiments v80 (GLACIER-only final re-rank, identity fusion; reported 0.436; its notes: ICE+GL 0.402 vs GLACIER-only 0.417), samanyu1808/casmi26-iceberg-where-glacier-is-blind (reported 0.438), canqiang/casmi26-e006-window-40-ice-fallback-true (reported ~0.44). Reported scores are theirs, public LB only.

Sources: FLARE (bioRxiv 2026.01.27.702086), MSAlign (arXiv 2605.19752), GLMR (arXiv 2511.06259),
cross-instrument contrastive DG (arXiv 2602.00547), MS-GPT (arXiv 2607.23607), MARLIN (arXiv 2607.04774),
MassKG (PMC11415640), MassSpecGym in the Wild (arXiv 2606.19624).

## Post-CMatch plan: per-case experts, architecture changes, branching steps (2026-10-09)

### Experts per failure case, stitched by a router
| Case (pilot share) | Expert (research) | Availability |
|---|---|---|
| 1. Truth not in pool (~27%) | Generative proposals (MS-GPT formula-given, MARLIN formula-free, DiffMS/FlowMS) + generated NP library (MassKG-style) | MARLIN no code; others to check |
| 2. In pool, outside top 25 (~18%) | Dense retrieval with a contrastive matcher over the whole mass window (FLARE, MSAlign, JESTR/MVP) | our CMatch / FLARE-style build |
| 3. In top 25, not first (~28%) | Isomer experts: ICEBERG, GLACIER, FLARE peak-to-atom, new cross-encoder | partly in pipeline |

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
| 4 | ~~Dense retrieval~~ dropped 2026-10-15: inside +-10 ppm only 2 of 19 pool-covered misses are in the window (CMatch ranks 29, 36); wider windows only helped MassSpecGym's off-mass precursors | — | — | — | — |
| 5 | Pool expansion: generated NP library | truth-in-pool +5 pts; sim2 not worse | keep -> 6 | 2nd try other generator -> 6 | GPU h |
| 6 | Router v1 (heuristic) | sim2 >= +0.01 or same score, less runtime | keep -> 7 | always-on flow | small |
| 7 | Generative proposals for likely-not-in-pool molecules | START if not-in-pool >= 20% after 5, or 3 and 4 each < +0.01. KEEP if >= +0.015 and <= 1.5 h | keep -> 8 | drop | public-weights model needed |
| 8 | Cross-encoder re-ranker on top 25 | sim2 >= +0.01; runtime <= +45 min | keep -> 9 | drop | ~12 GPU h |
| 9 | Learned stacker + router (E) | >= 500 unseen questions, then sim2 >= +0.01 | keep | heuristic router | data |
| 10 | Final picks: LB-best + sim2-best | — | — | — | — |

Runtime rule: total <= ~8.5 h (now ~6.5 h); if a step doesn't fit, the router skips ICEBERG/GLACIER on solved molecules first.

## Plan review against the literature (2026-10-15)

What changed since the plan was written: v6 is the baseline (LB 0.425; sim2 official-key MRR 0.420 vs v4n 0.341,
+0.079 [95% CI +0.032, +0.129]); the hidden test is timsTOF with precursors within ~4 ppm; CMatch retrieval has no
headroom inside the +-10 ppm window. Where v6 loses questions on sim2 (103 q, before the precursor fix): truth in no
candidate source ~37%; truth only in the PubChem list (ranks 4-22) ~8%; truth found but outside the top 25 ~2%.

### Add (in priority order)
| # | What | Evidence | Gate / first test | Cost |
|---|---|---|---|---|
| 1 | **Offline replay of the final step.** Re-run `v6_final.py` locally on saved sim2 outputs (prefusion, engine 2, PubChem lists, GLACIER/ICEBERG scores) with GLACIER's public `fuse.py`, so final-step variants are scored in minutes without GPU | Our traces: every final-step question so far was answerable from saved files | Reproduces sim2 v6 final MRR exactly | ~0.5 day, no GPU |
| 2 | **PubChem candidates into the GLACIER re-score window** (top-K PubChem not already present join the fused top-40 before re-scoring, instead of only filling empty tail slots) | 8 truths sit in the PubChem list at ranks 4-22 but never reach the final list; GLACIER already scores PubChem candidates for ~26 molecules | Replay (#1) first, then sim2 >= +0.01, LB >= -0.02 | small |
| 3 | **Confidence router** (plan step 6): low top-1 confidence -> more PubChem/expanded slots; high -> keep the list | Selective prediction for MS/MS retrieval (arXiv 2603.10950): retrieval-level uncertainty and cheap first-order confidence are the best "trust the top-1?" signals | Replay (#1); sim2 >= +0.01 | small |
| 4 | **Coverage diagnostic for the ~37% not-in-any-source**: which public databases hold those truths (PubChem depth beyond the channel's window, COCONUT 2.0 release vs the pool's copy, LOTUS, PubChemLite) | Pool expansion is the largest remaining gap; COCONUT 2.0 (2024) was re-curated | Count truths recovered per database before building anything | CPU only |
| 5 | **Consensus spectrum** per molecule (merge all spectra across collision energies, max intensity per peak) as an extra library-match / look-alike term | MVP (bioRxiv 2025.11.12): consensus view rank@1 36.0% vs 26.4% single spectrum on MassSpecGym; test molecules have 1-16 spectra at several energies | As a re-rank z-term at weight 1.0 (ranker retraining is blocked on data, stage E) | ~1 day |
| 6 | **A published look-alike scorer instead of our own:** MVP (code + weights public, HassounLab/MVP) as a third re-scoring term beside GLACIER/ICEBERG | Reports 2.4x MIST and JESTR rank@1 on MassSpecGym | Licence + EXTERNAL_RESOURCES + ban check first. **sim2 caveat:** MVP is trained on MassSpecGym, which our sim2 questions come from -> only MassSpecGym *test-fold* questions are fair | ~1-2 days |
| 7 | **MS2DeepScore 2.0** (public weights, cross-ion-mode) as evidence in engine 2's analog search | Lets negative-mode test spectra (~19%) use positive-mode library analogs | Same external-resource checks; sim2 >= +0.01 | ~1-2 days |

### Update
- **sim2** (done 2026-10-15): official-key scorer (stereo removed before tautomer canonicalization) and precursor
  m/z recomputed from the truth with timsTOF-like noise. Re-baseline = v6 on sim2-build v11.
- **Generative proposals (stage D)**: candidate models now FlowMS (ICLR 2026; 9.2% top-1 on NPLIB1; code/weights availability not yet checked),
  DiffMS, MS-BART; DualLGD (2026) claims 34% NPLIB1 top-1, unverified. All need a formula: check formula accuracy
  (MIST-CF / BUDDY-style) on timsTOF first. MARLIN still has no code.
- **FPNet retraining (backlog v5e)**: optimise retrieval, not fingerprint accuracy; arXiv 2602.16507 shows the two
  trade off and the best choice depends on the candidate sets.
- **Evaluation hygiene**: "MassSpecGym in the Wild" (arXiv 2606.19624) found evaluation bugs or leakage in 17 of 26
  papers; any external model trained on MassSpecGym must be tested on questions it never saw (see #6).

### Remove / stop
- **CMatch line (stages A2 and B, step 4 dense retrieval)**: two attempts without gain (v1 re-rank LB 0.372;
  retrieval no headroom inside +-10 ppm). If a learned look-alike scorer is wanted, use a published one (#6).
- **MARASON, FraGNNet**: no public weights / code found. ICEBERG + GLACIER stay the forward simulators.

## Results

Metric: MRR@25 (InChIKey-14), GroupKFold by molecule, paired bootstrap 95% CI vs the baseline.

| Date | Change | Commit | sim2 MRR (Δ [95% CI]) | Class-1 Δ | Class-2 Δ | Runtime Δ / mol | Public LB | Decision |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | Baseline v4n (current submission) | n/a | see v5a row (2026-10-10) | | | | 0.401 | reference |
| 2026-10-09 | sim2 external pilot (MassSpecGym, 11 unseen q), baseline v4n unchanged | n/a | final MRR 0.322 (pre-fusion 0.238); top1 0.27, recall@25 0.55; truth in pool 73% | — | — | ~ | 0.401 | realistic (train-holdout gave 0.93); only 232 of 28,929 MassSpecGym compounds unseen → ~125 usable questions; running all next |
| 2026-10-10 | v5a sim2 external baseline (103 unseen MassSpecGym q), v4n unchanged | n/a | final MRR 0.319 (top1 0.21, top5 0.44, recall@25 0.55); pre-fusion 0.362 (top1 0.29); in-pool 0.42 final / 0.50 pre; truth in pool 71%; engine candidate recall 52% | — | — | ~36 s/mol + ~20 min fixed | 0.401 | baseline set. Final engine-2 fusion costs -0.043 on sim2 -> test v5i (no fusion) on LB |
| 2026-10-10 | v5b CMatch v1 training (casmi26-cmatch-train) | e6770fd | val isomer MRR 0.642 (n=1,792 held-out train compounds vs <=100 same-formula decoys); plateau from ~epoch 20; 1.50M spectra / 270,810 compounds; 46 min T4 | — | — | inference: seconds | pending | Step 1 passed (>= 0.30) -> sim2 + LB tests |
| 2026-10-11 | v5b CMatch v1 within-formula re-rank on sim2 external (103 q) | e6770fd | pre-fusion 0.347 (Δ -0.015 vs 0.362); final 0.302 (Δ -0.017 vs 0.319); top1 pre 28.2% (-0.9 pt), top5 42.7% (-2.9 pt) | — | — | ~2 s total | 0.372 | FAILS sim2 gate (>= +0.01). Random-compound val (0.642) overstated generalisation; switched CMatch val to scaffold hold-out. Keep v1 only if LB >= 0.421 |
| 2026-10-11 | v5i no final engine-2 fusion (casmi26-v5-lb --no-fusion) | 287b726 | sim2 had shown +0.043 (0.362 vs 0.319, 103 q) | — | — | slightly faster | 0.377 (Δ -0.024) | FAILS LB gate (>= -0.02): keep fusion. sim2 sign was wrong; +0.043 is ~1σ of sim2 noise and MassSpecGym questions differ from the hidden test, so don't trust sim2 alone on engine-2 changes |
| 2026-10-11 | v5b CMatch v1 within-formula re-rank, leaderboard (casmi26-v5-lb --cmatch) | 0ee8c8e | see sim2 row: -0.015 | — | — | ~2 s total | 0.372 (Δ -0.029) | FAILS (bar was LB >= 0.421). Drop the v1 re-rank; sim2 and LB agree. Next attempt: CMatch retrained with scaffold-hold-out val |
| 2026-10-11 | CMatch attempt 2 training (casmi26-cmatch-train, scaffold hold-out val, 2 variants) | 4ab9d53 | scaffold-val isomer MRR: v1s 0.665 (best epoch 28; seen-train 0.823), reg 0.656 (epoch 27); winner v1s; 1.50M spectra / 270,810 compounds; ~39 min T4 per variant | — | — | inference: seconds | — | passes gate (>= 0.30) with an honest split. Weights in private dataset `t3jas06/casmi26-cmatch-v2` (cmatch.pt = v1s; `casmi26-cmatch` can't take uploaded versions). Use for dense retrieval on top of v6 |
| 2026-10-13 | Simulator-confound check on the 0.372 / 0.377 runs | n/a | ICE meta and GL meta `status: ok` in both logs (RDKit 2025.03.6, CUDA, 11/11 smoke molecules scored) | — | — | — | — | confound ruled out: the pinned docker image avoids the Python 3.13 wheel failure. 0.372 and 0.377 were real losses |
| 2026-10-13 | v6 on sim2 external (103 q): GLACIER-first re-scoring, identity fusion, metric-safe tail | 75ce79a | final **0.379** (Δ +0.060 vs 0.319); top1 0.291 (+0.08), top5 0.476 (+0.04), recall@25 0.524 (**-0.03**); pre-fusion 0.378 (+0.016); V6 FINAL: GLACIER 100/103, 208 slots filled, 20 top-1 changes, 0 errors | — | — | sim2 run 5,863 s for 103 q (1.7 GPU h) | — | passes sim2 gate. Recall@25 drop: dedupe/tail may push truths out of the top 25 — investigate |
| 2026-10-13 | v6 leaderboard (casmi26-v6-lb v1, built from t3jas06/baseline) | 75ce79a | see sim2 row: +0.060 | — | — | finished inside the 9 h limit | **0.425** (Δ +0.024) | **KEEP: new best; v6 = new baseline.** sim2 and LB agree. Public notebooks report 0.43–0.44 for this direction; the DreaMS NP fill they use is still left out |
| 2026-10-14 | Instrument check (casmi26-instrument-check, CPU) | 99d6218 | visible test.parquet (1,213 spectra) has an `instrument_type` column: 100% `timsTOF`; 81% positive mode, 79% [M+H]+. Spectrum->instrument classifier (GroupKFold by compound, 60 classes) 0.395 accuracy, timsTOF recall 0.70; predicts 95.4% timsTOF on test | — | — | ~10 min CPU | — | test is timsTOF (confirmed two ways). `cmatch_rerank.py` lowercases `instrument_type`, so its timsTOF flag = 1 on test: no bug. Practice set (sim2) has no timsTOF -> HANDOFF step 'grow the practice set with timsTOF spectra' matters more |
| 2026-10-15 | v6 recall@25 drop investigated (sim2 v9 traces) | n/a | 2 of 3 'lost' truths were the v6 top-1 under the official key (stereo removed before tautomer canonicalization); the sim2 scorer lacked stereo removal | — | — | — | — | scorer artifact. Fixed: `official_key` in sim_setup.py / sim_finish.py (question selection unchanged; *_oldkey kept) |
| 2026-10-15 | v4n sim2 rerun with the official-key scorer (sim2-build v10) | this commit | final 0.341 (old key 0.319, reproduces the logged baseline); recall@25 0.573 | — | — | 5,017 s | 0.401 | v6 vs v4n (official key, same 103 q): **+0.079 [95% CI +0.032, +0.129]**, better 24 / worse 6; recall@25 equal (0.573). v6 kept with CI > 0 |
| 2026-10-15 | CMatch v2 dense retrieval, offline CPU test over v2-pool mass windows | n/a | +-10 ppm: only 2 of 19 pool-covered v6 misses in the window, CMatch ranks 29 / 36. +-50 ppm: 9 of 15 in CMatch top-5, but only because MassSpecGym precursors are 10-160 ppm off | — | — | — | — | **dropped**: no headroom on accurate precursors. Ends the CMatch line (two attempts) |
| 2026-10-15 | Precursor mass-error check (casmi26-mass-error-check, train sample 257k) | this commit | timsTOF: median 1.9 ppm, p99 4.1 ppm, 0% > 10 ppm (every adduct); Orbitrap p99 33 ppm; QTOF 24% > 10 ppm | — | — | ~10 min CPU | — | hidden test (timsTOF) has accurate precursors; sim2's MassSpecGym precursors don't -> precursor fix in sim2 |
| 2026-10-15 | **v6 sim2 with precursor fix + official key (sim2-build v11) = new sim2 reference** | this commit | final **0.574** (top1 0.456, top5 0.709, recall@25 0.748); pre-fusion 0.581; engine candidate recall 0.718 (was 0.524); same 103 q | — | — | 6,023 s | 0.425 | reference for all later steps. The v6 final step now costs -0.008 vs its own pre-fusion list (it gained on the off-mass data) -> first job for the replay harness (EXECUTION_PLAN step 1) |
