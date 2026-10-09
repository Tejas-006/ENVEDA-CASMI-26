# ENVEDA-CASMI-26

Solution code for the Kaggle competition "Enveda CASMI 2026 - Molecule ID From Mass Spectra"
(rank up to 25 SMILES per `molecule_id`, scored by MRR@25 on InChIKey-14).

## Competition rules: hard constraints on everything built here

Every model, dataset, feature and commit must comply. If something is unclear, stop and ask; never assume it's allowed.

### Competition Data (Rules section 4)
- Non-commercial use only: the competition, Kaggle forums, academic research and education.
- Never commit, upload or publish Competition Data or anything that reproduces it to a place non-participants can reach.
  Covered: `train.parquet`, `test.parquet`, `sample_submission.csv`, and derived tables that contain their
  spectra, structures or labels (spectrum caches, `train_structs.parquet`, `train_fp_sel.npy`, simulation/ranker
  training rows such as sim1x/sim2, pools built from train).
- Kaggle Datasets holding such derived files must stay **private**. Only attach them to our own notebooks.
- Model weights trained on Competition Data are OK to keep in private datasets. Code is OK to commit.
- If unauthorized access or transmission happens, tell the user immediately (they must notify Kaggle).

### External Data and Tools (Rules section 6)
- Use only external data, models and tools that are **publicly available and free to all participants**
  (e.g. GNPS, MoNA, MassBank, MassSpecGym, PubChem, DreaMS, ChemBERTa, ICEBERG, RDKit, LightGBM).
- Do not use paid or proprietary resources (e.g. the NIST MS/MS library, commercial spectral databases, paid
  APIs) without the user's explicit approval after a Reasonableness check (section 6b).
- Check the competition's discussion forum and Data tab for host-specific bans before adding any new external
  resource; a host prohibition overrides everything else.
- AutoML tools only with a licence that permits compliance (section 6c).

### Code-competition limits (from public write-ups; confirm on the Overview → Code Requirements tab)
- Scored notebook: internet OFF, must finish in <= 9 h, writes `/kaggle/working/submission.csv`; submit via the notebook.
- Hidden test ~400 molecules, up to 25 SMILES each. 5 submissions/day; 2 final selections.
- Current pipeline estimate on 400 molecules: ~6.5 h (engine 2 ~1.9 h, PubChem ~1 h, v1 engine ~0.8 h, ICEBERG <= 1.5 h,
  GLACIER <= 1.1 h, fixed ~20 min). Any addition must report its runtime and keep the total under ~8.5 h.

### Winner obligations and reproducibility (section 2.8)
- Record every external resource in `EXTERNAL_RESOURCES.md`: name, source URL, version/record ID, licence,
  and what it's used for. Add the entry in the same change that introduces the resource.
- Everything we train must be rebuildable from code in this repo: no hand-edited artifacts and no
  undocumented weights.
- Keep MANIFEST.json-style hashes for every file the Kaggle notebook loads.

## Validation gate: every replacement or addition is tested on sim2 before it ships

sim2 = simulation built inside Kaggle (replaces the lost sim1x): training molecules held out by scaffold
(over-sampling natural-product-like ones) as pseudo-queries, run through the engine, candidates labelled by
InChIKey-14 match. Keep its rows in a private Kaggle output. It's derived Competition Data: keep it out of git (see .gitignore).

Protocol, for each change (new feature, model swap, fusion or gate change, constant change):
1. **Baseline first.** Reproduce the current pipeline's sim2 score before changing anything; record it.
2. **Same metric as Kaggle.** MRR@25 with RDKit tautomer canonicalization and InChIKey-14 matching. Also
   report top-1, top-5 and recall@25, overall and split by query class (class-1 = in library,
   class-2 = analog only, and any others present).
3. **No in-sample scores.** The shipped ranker was trained on train-derived simulations, so evaluate with GroupKFold by molecule
   (retrain ranker/stacker per fold). Tune constants on folds, never on the reported fold.
4. **One change at a time.** Ablate it: baseline vs baseline + change, same folds and seeds.
5. **Significance.** Paired bootstrap over molecules (≥1000 resamples); report ΔMRR with a 95% CI.
   Ship only if the CI lower bound is > 0, or if it's ~neutral on MRR and buys runtime or robustness
   (say which).
6. **Budget check.** Report added runtime per molecule and confirm the full hidden-test run stays inside
   the Kaggle time limit.
7. **Log it.** Append a row to `EXPERIMENTS.md`: date, change, commit, ΔMRR [CI], per-class Δ, runtime,
   decision. Public LB score goes there too, but only as a secondary check.

If sim2 isn't available in the session, don't ship the change: say so and ask the user for it.

### Leakage hygiene (validation integrity, not a rule, but required)
- Deduplicate external spectra (e.g. MassSpecGym) against Competition Data by InChIKey-14 before using them for
  validation.
- Never validate on the visible `test.parquet`; it's a dummy replaced at rerun.
- Never tune constants against the public leaderboard alone; validate on the held-out simulation first.
