# Notebook roadmap (Kaggle)

All notebooks live on Kaggle (this environment can't reach it): Claude writes them here, you create and run them there.
Code goes in `notebooks/` in this repo; data and outputs never do (rules section 4b).

## Kaggle basics used in every step

- **Create:** Kaggle → Code → New Notebook → File → Import Notebook (upload the `.ipynb` from this repo).
- **Inputs:** right panel → Add Input → attach the datasets/models listed for that notebook.
- **Settings:** Accelerator = GPU (T4 x2 or P100) unless noted; Internet = Off for the submission notebook
  (On is fine for training notebooks that download public weights).
- **Run in background:** Save Version → "Save & Run All (Commit)". It runs up to 12 h without your browser open.
- **Outputs to dataset:** open the finished version → Output → "New Dataset" (or "New Version" of an existing one).
  **Set visibility to Private** for anything derived from Competition Data.
- **GPU quota:** about 30 h/week, shared by every notebook below. Check Settings → Quotas.

## Notebooks

| ID | Name | Purpose | GPU | Output → private dataset |
|---|---|---|---|---|
| N1 | `casmi26-sim2-build` | Build sim2: held-out training queries run through the full pipeline; dumps labelled candidate rows | Yes | `casmi26-sim2` (one version per shard) |
| N2 | `casmi26-sim2-score` | MRR@25 (InChIKey-14), GroupKFold ranker retrain, paired bootstrap vs baseline | No (CPU) | none (results go to EXPERIMENTS.md) |
| N3 | `casmi26-msalign-train` | Train the MSAlign heads (frozen DreaMS + ChemBERTa) on train.parquet pairs | Yes | `casmi26-msalign` |
| N4 | `casmi26-marason-check` | Check MARASON code and weights; run it on a few molecules | Yes | `casmi26-marason` (if viable) |
| N5 | `casmi26-v5-lb` | Submission notebook = baseline + new re-rank terms (leaderboard track) | Yes | none (submit) |
| N6 | `casmi26-v5-sim` | Submission notebook = retrained ranker + stacker (sim track) | Yes | none (submit) |

## Step by step

### Step 0: prepare inputs (once)
1. Create private datasets for external weights (Internet On is allowed while preparing them):
   - `casmi26-dreams-weights` ← DreaMS checkpoints from Zenodo record 10997887.
   - `casmi26-chemberta` ← ChemBERTa from Hugging Face (seyonec).
2. Fill in the matching rows of `EXTERNAL_RESOURCES.md` (URL, version, licence).
3. Confirm `casmi26-v4b-models`, `casmi26-v3-models`, the pool and other input datasets are **Private**.

### Step 1: N1, sim2 shard 1 (start first, runs in the background)
1. Generate the notebook (Claude does this; it's never committed because it contains the baseline code):
   `python tools/make_sim2_notebook.py baseline.ipynb casmi26-sim2-build.ipynb --shard 0 --size 500`
2. Kaggle → open the 0.401 baseline notebook → **Copy & Edit** (keeps every input attached) →
   File → Import Notebook → upload `casmi26-sim2-build.ipynb`. GPU on.
3. **Timing test:** in the "sim2 setup" cell set `SHARD_SIZE = 20` and Run All interactively. Check the summary printed
   at the end and note `secs`.
4. **Real run:** set `SHARD_SIZE` back to 500 (lower it if the timing test projects more than ~11 h), then
   Save Version → Save & Run All (Commit).
5. When done: Output → New Dataset `casmi26-sim2` (**Private**). Paste `summary.json` into the chat.
6. Later shards: regenerate with `--shard 1, 2, ...` (or edit `SHARD`), commit, and add each output as a new version.

How it works: queries are training compounds that are also in COCONUT/ChEBI/LIPID MAPS (natural products, so the
true structure stays in the candidate pool), timsTOF spectra preferred. A fixed seed orders them, so shards never
overlap. Class 1 (~30% of eligible) holds out one instrument's spectra and keeps the rest in the library;
class 2 removes the compound from the library and from train_structs/train_fp_sel. The unchanged pipeline then
runs, and `sim_out/` gets queries, top-300 candidate feature rows with labels, pre/post ICE+GL lists, engine and
PubChem lists, both submissions and a quick MRR summary.
Known biases (absolute scores optimistic, comparisons fair-ish): FPNet, the fe_v4 priors and the engine-2 ranker
rows were built from training data that includes these compounds; the pool keeps truths via COCONUT, so the
PubChem-only case isn't simulated.

### Step 2: N3, MSAlign heads (while shard 1 runs)
1. Import `notebooks/casmi26-msalign-train.ipynb`. Attach competition data, `casmi26-dreams-weights`, `casmi26-chemberta`.
2. Commit. Output → New Dataset `casmi26-msalign` (Private).

### Step 3: N4, MARASON feasibility (short)
1. Run interactively on a few molecules. If code or weights aren't available, drop v5d and note it in EXPERIMENTS.md.

### Step 4: N5, leaderboard version 1 → submit
1. Copy of the 0.401 notebook + MSAlign score (+ MARASON if viable) as extra re-rank z-terms at **default weight 1.0**.
   No weight tuning against the leaderboard.
2. Internet Off. Commit, then Submit to competition. Log the LB score in EXPERIMENTS.md.

### Step 5: shard 1 done (500 queries) → N2
1. Import `notebooks/casmi26-sim2-score.ipynb`, attach `casmi26-sim2`. CPU is enough.
2. Record the **v5a baseline** sim2 MRR.
3. Score v5b as a **ranker feature** (retrained ranker, GroupKFold) and v5d, each vs baseline with CI. Log both.

### Step 6: keep going in parallel
- Sim track: run shards 2–5 → at ~1,000–2,000 queries do v5c (stacker), then v5e and v5g as features → N6 → submit.
- LB track: v5f generative re-rank term → new version of N5 → submit.

### Step 7: final selection
- Pick 2 final submissions on Kaggle: best N5 version (LB) + best N6 version (sim2).
