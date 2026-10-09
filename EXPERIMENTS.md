# Experiments log (sim2 validation gate; see CLAUDE.md)

## Ground rules

- Comply with competition rules sections 4 and 6 (see CLAUDE.md).
- No more tuning constants against the public LB; it's a sanity check only.
- Every change is scored on sim2: MRR@25, GroupKFold by molecule, and it ships only with a positive paired-bootstrap 95% CI.
- Final picks: one sim2-best submission + one public-LB-best submission.
- sim1x (the original v4 ranker's simulation) is lost; sim2 replaces it.

## Version plan

Projections are rough guesses from public benchmarks (MassSpecGym etc.), not measurements; gains overlap, so they don't add up.
Combined target: ~0.43–0.46 public LB.

| # | Ver | Change | Why / projected effect | Needs sim2? | Status |
|---|---|---|---|---|---|
| 0 | v5a | Build sim2 and score the unchanged pipeline on it (notebooks: NOTEBOOKS.md). Questions now come from public MassSpecGym spectra of compounds NOT in competition train (`--source external`); the train-holdout mode is kept but is far too easy | Replaces lost sim1x; baseline's sim score for all later comparisons; LB stays 0.401 | builds it | train-holdout test (20 q): MRR 0.975 pre-fusion / 0.933 final vs LB 0.401 → models memorised train compounds; switched to external questions, download notebook ready |
| 1 | v5c | Learned stacker replacing ICE_LAM/GL_LAM/ALPHA/KRR and the PubChem gate (LIB_TAU/REL_TH/slots) | Removes ~8 LB-tuned constants: biggest generalization fix; per-molecule weighting for later signals; +0.005–0.02 | yes | planned |
| 2 | v5b | CMatch: own contrastive spectrum↔molecule matcher (spectrum transformer + fingerprint MLP, trained on competition train vs same-formula decoys, sim2 hold-out excluded) as within-formula re-rank term (LB track), later a ranker feature (sim track). DreaMS/ChemBERTa dropped: DreaMS pins old torch/numpy/RDKit and can't run offline beside RDKit 2026 | Ranks candidates directly, should lift top-1 among same-formula isomers; +0.01–0.03 | LB: no; ranker feature: yes | training notebook ready (tools/make_cmatch_notebook.py) |
| 3 | v5d | Swap ICEBERG → MARASON (retrieval-augmented ICEBERG) | Stronger isomer re-scoring (18.7%→27.8% top-1 in paper); +0.01–0.02 | optional | on hold: code is public in coleygroup/ms-pred (MIT) but no released weights; training it ourselves is days of GPU |
| 4 | v5e | Retrain FPNet with listwise/contrastive loss over candidate sets | Optimizes ranking, not bit accuracy; low-risk swap; +0.005–0.015 | yes | planned |
| 5 | v5f | Generative re-rank feature (GLMR-style; MS-BART/FlowMS/MARLIN), only on weak-library-match molecules | Orthogonal signal + new candidates for out-of-pool compounds; +0.005–0.02, GPU cost | optional | planned |
| 6 | v5g | Formula-confidence feature (SIRIUS/BUDDY-style) | Picks the right formula group before isomer ranking; skip if explain_score covers it; 0–+0.01 | yes, if ranker feature | planned |

## Roadmap and decision gates (2026-10-09)

Where MRR is lost (sim2 external pilot, 11 q, final MRR 0.32; refresh with the ~125-question run):
truth not in candidate pool ~27% | in pool but outside top 25 ~18% | in top 25 but not first ~28% | first ~27%.

Rules for every attempt: judged on sim2-external AND public LB. Keep if sim2 ΔMRR >= +0.01, LB drop <= 0.02
(its noise band, ~130 molecules) and runtime fits. Two attempts per direction, then move on.

| Stage | What | Gate to keep | If it fails |
|---|---|---|---|
| A | CMatch within-formula re-rank (v5b) | training val isomer MRR >= 0.30; sim2 >= +0.01; LB >= -0.02 | 2nd try: FLARE-style peak-to-atom matcher; then drop contrastive line |
| B | Cross-instrument robustness for CMatch (hidden test looks timsTOF; MassSpecGym has none) | only after A passes; LB >= +0.01 (sim2 can't measure it) | drop |
| C | Bigger candidate pool: generated natural-product library (MassKG-style) | truth-in-pool +5 points, sim2 not worse | 2nd try: different library/generator |
| D | Generative models (MARLIN formula-free / MS-GPT), only for weak-library-match molecules, as candidate proposer + GLMR-style re-rank | START only if after C truth-not-in-pool >= 20%, OR two ranking upgrades in a row each < +0.01. KEEP if >= +0.015 and <= 1.5 h extra runtime | drop |
| E | Learned stacker (v5c) | needs >= 500 unseen sim2 questions (add CASMI 2016/2017/2022 answer sets etc.) | keep fixed weights |

Sources: FLARE (bioRxiv 2026.01.27.702086), MSAlign (arXiv 2605.19752), GLMR (arXiv 2511.06259),
cross-instrument contrastive DG (arXiv 2602.00547), MS-GPT (arXiv 2607.23607), MARLIN (arXiv 2607.04774),
MassKG (PMC11415640), MassSpecGym in the Wild (arXiv 2606.19624).

Availability check 2026-10-09: MARLIN (arXiv 2607.04774, Che/Du/Xu, UNC Charlotte, posted 2026-07-06) has no public
code or weights found (paper mirrors, web search, likely GitHub names). Stage D must use a model with released code + weights.

## Results

Metric: MRR@25 (InChIKey-14), GroupKFold by molecule, paired bootstrap 95% CI vs the baseline.

| Date | Change | Commit | sim2 MRR (Δ [95% CI]) | Class-1 Δ | Class-2 Δ | Runtime Δ / mol | Public LB | Decision |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | Baseline v4n (current submission) | n/a | TODO (v5a) | | | | 0.401 | reference |
| 2026-10-09 | sim2 external pilot (MassSpecGym, 11 unseen q), baseline v4n unchanged | n/a | final MRR 0.322 (pre-fusion 0.238); top1 0.27, recall@25 0.55; truth in pool 73% | — | — | ~ | 0.401 | realistic (train-holdout gave 0.93); only 232 of 28,929 MassSpecGym compounds unseen → ~125 usable questions; running all next |
