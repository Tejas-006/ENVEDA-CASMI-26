# Experiments log (sim1x validation gate; see CLAUDE.md)

## Version plan

Projections are rough guesses from public benchmarks (MassSpecGym etc.), not measurements; gains overlap, so they don't add up.
Combined target: ~0.43–0.46 public LB. Each version must pass the sim1x gate before it ships.

| Ver | Change | Why / projected effect | Status |
|---|---|---|---|
| v5a | Reproduce baseline v4n on sim1x, no changes | Reference score for all later comparisons; LB stays 0.401 | blocked: needs sim1x |
| v5b | MSAlign-style contrastive score (frozen DreaMS + ChemBERTa, trained MLP heads) as ranker feature + 4th re-rank z-term | Retrieval-aligned signal, should lift top-1 among same-formula isomers; +0.01–0.03 MRR | planned |
| v5c | Learned stacker replacing ICE_LAM/GL_LAM/ALPHA/KRR and the PubChem gate (LIB_TAU/REL_TH/slots), trained on sim1x folds | Removes ~8 LB-tuned constants, cuts private-LB risk, per-molecule weighting; +0.005–0.02 | planned |
| v5d | Swap ICEBERG → MARASON (retrieval-augmented ICEBERG) | Stronger isomer re-scoring (18.7%→27.8% top-1 in paper); +0.01–0.02, if code/weights available | planned |
| v5e | Retrain FPNet with listwise/contrastive loss over candidate sets | Optimizes ranking, not bit accuracy; low-risk swap; +0.005–0.015 | planned |
| v5f | Generative re-rank feature (GLMR-style; MS-BART/FlowMS/MARLIN samples), only on lib_max < LIB_TAU molecules | Orthogonal signal + novel candidates for out-of-pool compounds; +0.005–0.02, GPU cost | planned |
| v5g | Formula-confidence feature (SIRIUS/BUDDY-style) | Picks the right formula group before isomer ranking; skip if explain_score covers it; 0–+0.01 | planned |

## Results

Metric: MRR@25 (InChIKey-14), GroupKFold by molecule, paired bootstrap 95% CI vs the baseline.

| Date | Change | Commit | sim1x MRR (Δ [95% CI]) | Class-1 Δ | Class-2 Δ | Runtime Δ / mol | Public LB | Decision |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | Baseline v4n (current submission) | n/a | TODO (reproduce) | | | | 0.401 | reference |
