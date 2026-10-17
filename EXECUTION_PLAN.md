# Execution plan (2026-10-15)

Ordered by expected impact per unit of cost. Each step has a projected gain, a kill criterion and a fallback, so a
projection that doesn't pan out ends cleanly instead of being retried. Evidence and sources: EXPERIMENTS.md,
"Plan review against the literature". Rules from CLAUDE.md apply to every step (external-resource checks, no
Competition Data in git, sim2 before LB, one change at a time, two attempts per direction).

**Baseline:** v6 (LB 0.425). sim2 reference = v6 on sim2-build v11 (official-key scorer + precursor fix, 103 q).
**Decision rule:** keep if sim2 ΔMRR ≥ +0.01 with paired-bootstrap CI lower bound ≥ −0.005 (103 q can't resolve
smaller gains), LB drop ≤ 0.02, runtime ≤ ~8.5 h total. Submissions need Tejas's OK.
**Measurement limit:** with 103 questions a paired 95% CI is about ±0.045 MRR. Steps projected below +0.02 need the
bigger practice set (step 3) before they can be judged; until then they are judged on recall/top-k counts and
"no question got worse".

## Where v6 loses (sim2, 103 q, official key)

v11 (precursor fix): final 0.574, pre-fusion 0.581, recall@25 0.748, engine candidate recall 0.718. The table below
is from v9 (before the fix); the replay harness (step 1) re-derives it on v11 first. New: **the v6 final step
costs -0.008 on v11** (0.581 -> 0.574), so step 1 also checks which final-step part loses.
| Bucket | Share | Steps that target it |
|---|---|---|
| Truth in no candidate source | ~37% | 4 (coverage) → 7 (pool) → 11 (generative) |
| Truth only in the PubChem list (ranks 4–22), never reaches top 25 | ~8% | 5, 6 |
| Truth outside the ±10 ppm window (MassSpecGym precursor error) | ~16% before v11 | fixed in sim2 (not a hidden-test problem) |
| Truth retrieved, not first | rest of the misses | 8, 9, 10 |

## Steps

| # | Step | Implementation | Projected gain | Cost | Kill / fallback |
|---|---|---|---|---|---|
| 0 | **Re-baseline + housekeeping** | Commit scorer fix, precursor fix, mass-check script, plan review; log v11 as the v6 sim2 reference; mark CMatch A2/B/dense retrieval dropped (done in this commit) | — (measurement correctness) | done / ~1.7 GPU h | — |
| 1 | **Final-step replay harness** | `tools/replay/replay_final.py`: loads a sim2 `sim_out/` (sub_v4, eng/pc lists, base lists, GL/ICE scores, queries), re-implements `v6_final` with the public `fuse.py` of `ahmedberatozer/casmi26-glacier` (downloaded locally, never committed), scores with `official_key`. Must reproduce v11's final MRR exactly | Enabler: final-step ideas tested in minutes, no GPU | ~0.5 day, CPU | If it can't reproduce v11 within 0.001 → find the mismatch before using it; never tune on it what it can't reproduce |
| 2 | **Confidence features from saved runs** | In the harness: per-question top-1 confidence signals (ranker margin z1−z2, lib_max, GLACIER margin, agreement between ranker/engine-2/PubChem top-1). Report how well each predicts "top-1 correct" (AUC) | Enabler for 6 | ~2 h, CPU | AUC < 0.65 for all → step 6 becomes a fixed rule only |
| 3 | **Bigger practice set (≥ 300 unseen questions)** | Add MoNA / MassBank / GNPS public spectra of compounds not in train (dedupe by InChIKey-14 + tautomer, licence + ban checks, EXTERNAL_RESOURCES entries), prefer timsTOF/QTOF; precursor fix applies. New sim2 `--source external2`. Runs in parallel with 4–6 | Not a score gain: cuts the CI from ±0.045 to ~±0.025, so +0.01–0.02 steps become measurable | ~1 day + ~4 GPU h | If < 200 usable new questions → keep 103 q and judge small steps on counts only |
| 4 | **Coverage diagnostic for "truth in no source"** | CPU notebook: for every sim2 miss of that bucket, look up the truth in (a) PubChem beyond the channel's depth (`casmi26-pubchem-tier`, full ±10 ppm window), (b) COCONUT 2.0 (2024 Zenodo dump) vs the pool's COCONUT copy, (c) LOTUS, (d) PubChemLite. Report truths recovered per source and window sizes | Decides the largest remaining gap (~37%) | ~0.5 day, CPU | — (diagnostic) |
| 5 | **PubChem candidates through GLACIER** | v6 cell change: add PubChem top-K (K=10, not already in the fused list) to the GLACIER/ICEBERG input for every molecule (today only gated molecules), and to the v6 re-score window (fused top-40 ∪ PubChem top-K). ~4k extra GLACIER predictions ≈ 1–2 min | Up to 8 more truths reachable; projected sim2 +0.01–0.03 MRR, recall@25 +3–8 pts | ~0.5 day + 1.7 GPU h | Fail gate → **fallback 5b:** reserve slots 21–25 for PubChem top-5 not already present (recall-only, no MRR risk to top 20), tested on the harness. Fail again → drop |
| 6 | **Confidence router** | Using step-2 signals: high confidence → keep list; low → more PubChem / expanded-pool slots (generalises the existing PubChem gate). Thresholds chosen on harness folds, never on LB | +0.005–0.02 | ~0.5 day, CPU | No gain on harness → keep the existing gate |
| 7 | **Pool expansion (only if step 4 finds ≥ 5 pts truth coverage)** | Add the winning source's structures to the candidate pool with exact masses and the pool's fingerprint features; candidates enter the ranker like pool entries. Runtime check on 400 molecules | +0.02–0.05 if ≥ 5 pts coverage (largest ceiling) | 1–2 days + GPU | Coverage < 5 pts → skip to 11's start gate. Runtime > budget → add only to the PubChem channel |
| 8 | **Consensus spectrum term** | Merge each molecule's spectra across collision energies (max intensity per peak, ppm-binned); cosine of the consensus vs each candidate's library consensus (where the library has the structure) and vs GLACIER's predicted spectrum; add as a z-term at weight 1.0 in the v6 re-score | +0.005–0.02 (MVP: big gain for learned matchers; smaller expected for library cosine) | ~1 day | Fail gate → drop |
| 9 | **MS2DeepScore 2.0 for negative mode (~19% of test spectra)** | External-resource checks (licence, forum/Data-tab bans, EXTERNAL_RESOURCES, offline install beside RDKit 2026). Use cross-ion-mode similarity in engine 2's analog search for negative-mode molecules | +0.005–0.015 overall (negative-mode only) | 1–2 days | Install conflict or no gain → drop |
| 10 | **MVP as a third look-alike scorer** | Same checks; sim2 restricted to MassSpecGym **test-fold** compounds (MVP trained on the rest). Add z(MVP) beside GLACIER/ICEBERG in the v6 re-score | +0.01–0.03 if it transfers to timsTOF; uncertain | 1–2 days + GPU | Test-fold subset too small (< 50 q) → judge on step-3 set only; no gain → drop (ends learned look-alike scorers) |
| 11 | **Generative proposals (stage D), gated** | Start only if after 7 truth-in-no-source ≥ 20%. First: formula accuracy on timsTOF train (top-1 formula from MIST-CF/BUDDY-style scoring). Then FlowMS / DiffMS / MS-BART (check code + weights + licence) for weak-library molecules only, fixed slots | +0.01–0.04 if formula accuracy is high | days + GPU | Formula top-1 < 70% or no public weights → drop |

## Update (folded into the steps above)
- sim2: official-key scorer and precursor fix (done, step 0); larger practice set (step 3).
- Generative shortlist: FlowMS, DiffMS, MS-BART; formula accuracy first (step 11).
- FPNet retraining (backlog v5e): when it happens, optimise retrieval, not fingerprint accuracy.
- External models trained on MassSpecGym: only judged on questions they never saw (step 10).

## Remove (done in step 0)
- CMatch A2 (FLARE-style), B (instrument robustness) and dense retrieval: dropped after two attempts without gain.
- MARASON, FraGNNet: no public weights / code.

## Order of execution and parallelism
1. Step 0 → 1 → 2 (CPU, same day).
2. Steps 3 and 4 in parallel (CPU notebooks + data work) while 5 runs on GPU.
3. 6 after 2 and 5; 7 only if 4 says so; then 8, 9, 10 one at a time on the best list so far.
4. LB submissions: after 5 (if kept), after 7, and after the best of 8–10. Final picks: LB-best + sim2-best.

## Status
| Step | Status (2026-10-15) |
|---|---|
| 0 | done: v11 reference 0.574 (commit 04430da) |
| 1 | done: replay reproduces v11 exactly; found v6.1 (final-step GLACIER weight 0.5, window 25): +0.025 [+0.001, +0.052]; `casmi26-v61-lb` built, LB pending |
| 2 | done: no signal reaches AUC 0.65 -> step 6 reduced to a fixed rule, revisit after step 3 |
| 3 | **stopped by its kill rule** (2026-10-16): the competition train set already aggregates GNPS, MassBank, MoNA, RIKEN, MSnLib, Spectraverse and MS-DIAL (Data tab), so those libraries can't supply unseen questions (MassSpecGym gave 232 of 28,929). Rules section 6 allows free public external data unless the host bans it; GNPS community spectra are CC0, MassBank licences are per record (some CC BY-NC-SA), MoNA's licence unconfirmed. Fallback: keep 103 q and judge small steps on counts. sim2 realism still to add from the Data tab: drop peaks above precursor + 2 Da |
| 4 | done (2026-10-16): on v11 only **14/103** truths are in no source (not 37%: that was the off-mass precursors); **12 of the 14 are in PubChem** (PUG REST by InChIKey), i.e. inside the PubChem channel's +-10 ppm window but below its top 25; 2 are in no public database (generative only). So step 7 becomes 'deeper PubChem list', not a new database. Caveat: practice compounds come from public libraries, so they are likelier to be in PubChem than Enveda's hidden natural products. Also: 12 truths are in some source but not in the final top 25 |
| 5 | **passes sim2** (replay on v12 with v6 constants: +0.011 [+0.001, +0.024], recall +4.9 pts); `casmi26-v62-lb` = v6 + `--pc-sim 10` built for the LB |
| v6.1 | **failed on the LB** (0.399 vs 0.425): v6 constants restored; GLACIER weighting is not tuned on sim2 any more |
| 7 | **dropped** (2026-10-16): only 2 of 12 PubChem-held truths rank in the top 100 (35, 48). Remaining lever for that bucket = better PubChem ranking (fingerprint model), see step 10 |
| 6, 8-11 | not started. Note: sim2 has no negative-mode questions (test ~19%) |
