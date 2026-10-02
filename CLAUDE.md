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
  training rows such as sim1x, pools built from train).
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

### Winner obligations and reproducibility (section 2.8)
- Record every external resource in `EXTERNAL_RESOURCES.md`: name, source URL, version/record ID, licence,
  and what it's used for. Add the entry in the same change that introduces the resource.
- Everything we train must be rebuildable from code in this repo: no hand-edited artifacts and no
  undocumented weights.
- Keep MANIFEST.json-style hashes for every file the Kaggle notebook loads.

### Leakage hygiene (validation integrity, not a rule, but required)
- Deduplicate external spectra (e.g. MassSpecGym) against Competition Data by InChIKey-14 before using them for
  validation.
- Never validate on the visible `test.parquet`; it's a dummy replaced at rerun.
- Never tune constants against the public leaderboard alone; validate on the held-out simulation first.
