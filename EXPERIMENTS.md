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
| 0 | v5a | Build sim2 inside Kaggle (notebooks: see NOTEBOOKS.md) (scaffold hold-out, over-sample natural-product-like molecules, a few hundred queries to start) and score the current pipeline on it unchanged | Replaces lost sim1x; baseline's sim score for all later comparisons; LB stays 0.401 | builds it | next: harness notebook |
| 1 | v5c | Learned stacker replacing ICE_LAM/GL_LAM/ALPHA/KRR and the PubChem gate (LIB_TAU/REL_TH/slots) | Removes ~8 LB-tuned constants: biggest generalization fix; per-molecule weighting for later signals; +0.005–0.02 | yes | planned |
| 2 | v5b | MSAlign-style contrastive score (frozen DreaMS + ChemBERTa, trained MLP heads) as ranker feature + re-rank term | Ranks candidates directly, should lift top-1 among same-formula isomers; external pretraining helps generalization; +0.01–0.03 | yes | planned |
| 3 | v5d | Swap ICEBERG → MARASON (retrieval-augmented ICEBERG) | Stronger isomer re-scoring (18.7%→27.8% top-1 in paper); +0.01–0.02, if code/weights available | optional | planned |
| 4 | v5e | Retrain FPNet with listwise/contrastive loss over candidate sets | Optimizes ranking, not bit accuracy; low-risk swap; +0.005–0.015 | yes | planned |
| 5 | v5f | Generative re-rank feature (GLMR-style; MS-BART/FlowMS/MARLIN), only on weak-library-match molecules | Orthogonal signal + new candidates for out-of-pool compounds; +0.005–0.02, GPU cost | optional | planned |
| 6 | v5g | Formula-confidence feature (SIRIUS/BUDDY-style) | Picks the right formula group before isomer ranking; skip if explain_score covers it; 0–+0.01 | yes, if ranker feature | planned |

## Results

Metric: MRR@25 (InChIKey-14), GroupKFold by molecule, paired bootstrap 95% CI vs the baseline.

| Date | Change | Commit | sim2 MRR (Δ [95% CI]) | Class-1 Δ | Class-2 Δ | Runtime Δ / mol | Public LB | Decision |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | Baseline v4n (current submission) | n/a | TODO (v5a) | | | | 0.401 | reference |
