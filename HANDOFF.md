# Handoff: ENVEDA CASMI 2026, state as of 2026-10-12

Everything a new session needs to continue. Read `CLAUDE.md` first: its competition rules and validation gate
apply to every change. Results live in `EXPERIMENTS.md`, the Kaggle workflow in `NOTEBOOKS.md`, and external
resources in `EXTERNAL_RESOURCES.md`.

## 1. Where things stand

- **Best submission: 0.401 public LB** (v4n, a fork of a public notebook). Nothing we've tried has beaten it yet.
- **Next candidate: v6** (built, not run). It ports fixes from recent public notebooks that report 0.43–0.44.
- **CMatch attempt 2** (training notebook built, not yet confirmed running). Last attempt of the contrastive line.
- **Branch with all work:** `claude/gracious-maxwell-04dmc2` (it contains `claude/vibrant-edison-b7wd63` plus later
  commits). `main` only has the first 5 commits; nothing has been merged since. Open a PR when you want it in `main`.
- **Repo is public.** Never commit notebooks (`*.ipynb` is gitignored), baseline code, Competition Data or
  anything derived from it (sim2 rows, spectrum caches).

## 2. How we work

- Claude writes code and notebook generators in this repo. Kaggle can't be reached from a Claude session.
- Tejas runs everything on Kaggle (free T4, about 30 GPU h/week) from Windows cmd in `%USERPROFILE%\kaggle-ml`,
  using `python -m kaggle ...`. Kaggle user: `t3jas06`. Competition: `enveda-CASMI26-molecule-id-mass-spectra`.
- Typical loop: Claude builds a push folder (`kernel-metadata.json` + `.ipynb`), then Tejas runs
  `python -m kaggle kernels push -p <folder>` and `python -m kaggle kernels status t3jas06/<slug>`, and pastes
  log lines back. Submissions are made in the browser (notebook page → Submit to competition).
- Code-competition limits: internet off, ≤ 9 h, 5 submissions/day, 2 final picks. Pipeline is about 6.5 h on
  about 400 hidden molecules; keep any addition under about 8.5 h total.
- Keep rule (EXPERIMENTS.md): keep a change if sim2 gains ≥ 0.01 AND the public LB drops ≤ 0.02 AND runtime fits.
  Two attempts per direction, then move on. Never tune constants on the public LB. Final picks: one LB-best,
  one sim2-best.

## 3. Kaggle assets

| Asset | What | Status |
|---|---|---|
| 0.401 baseline notebook (owner t3jas06, slug not recorded here) | v4n pipeline: v1 engine + LightGBM ranker, engine 2 (analog propagation), PubChem channel, ICEBERG + GLACIER, final RRF fusion | Reference |
| `t3jas06/casmi26-sim2-build` | Practice test (sim2); `--source external` uses unseen MassSpecGym compounds | Last baseline run: 103 q |
| `t3jas06/casmi26-massspecgym` (private dataset) | MassSpecGym HF revision d2e86d0 | Done |
| `t3jas06/casmi26-cmatch-train` | CMatch training notebook | Attempt 2 notebook built |
| `t3jas06/casmi26-cmatch` (private dataset) | CMatch weights (`cmatch.pt`, `cmatch.py`, `cmatch_meta.json`) | Holds v1; v2 only if it passes the gate |
| `t3jas06/casmi26-v5-lb` | Baseline + CMatch v1 re-rank | LB 0.372 |
| `t3jas06/casmi26-v5-lb-nofusion` | Baseline without final engine-2 fusion | LB 0.377 |
| casmi26-instrument-check (new, browser import) | CPU check of the test set's instrument | Built, not run |

## 4. History: what worked and what didn't

| Date | Step | Result | Verdict and why |
|---|---|---|---|
| 09-16→09-30 | Fork of a public notebook (prvsiyan + ahmedberatozer datasets) → v4n | LB 0.401 | Worked. Rests on about 8 constants tuned on the public LB. |
| 10-02 | Rules guardrails (CLAUDE.md, .gitignore, EXTERNAL_RESOURCES.md) | — | Worked. Licence/version TODOs remain. |
| 10-02 | sim1x validation gate | sim1x lost | Didn't work; replaced by sim2 built inside Kaggle. |
| 10-02→10-08 | sim2 on held-out *training* compounds | MRR 0.93–0.97 | Didn't work: the pipeline's models were trained on those compounds. |
| 10-08→10-10 | sim2 on unseen MassSpecGym compounds | Pilot (11 q) 0.322; baseline (103 q) final 0.319, pre-fusion 0.362 | Worked as a realistic test, but only about 125 usable questions (noise ±0.035), and MassSpecGym is Orbitrap/QTOF, not timsTOF. |
| 10-10 | Failure breakdown (103 q) | Truth not in pool 29%; in pool but not retrieved 19% (pool 71% vs engine recall 52%); retrieved but not first 23%; first 29% | Points to retrieval and pool coverage, not re-ranking. |
| 10-09→10-10 | CMatch v1 training (own contrastive matcher; replaces DreaMS/ChemBERTa, which can't run offline beside RDKit 2026) | Val isomer MRR 0.642 on a random-compound split; 1.50M spectra, 270,810 compounds, 46 min T4 | Passed the 0.30 gate, but the split leaked scaffolds. |
| 10-11 | CMatch v1 within-formula re-rank on sim2 | Pre-fusion 0.347 (−0.015), final 0.302 (−0.017) | Didn't work: fails the +0.01 gate. |
| 10-11 | CMatch v1 on the LB (`casmi26-v5-lb`) | **0.372** (−0.029) | Didn't work; dropped. sim2 and LB agree. |
| 10-11 | No final engine-2 fusion (v5i) on the LB | **0.377** (−0.024) | Didn't work, although sim2 said +0.043. The fusion stays; sim2 can't be trusted alone on engine-2 changes. |
| 10-08→10-11 | MARASON, MARLIN | No public weights / code | Dropped. |

**Possible confound:** recent public notebooks found that on the newer Python 3.13 Kaggle image, ICEBERG's bundled
RDKit 2025.3.6 wheel (cp312) won't install, so ICEBERG and GLACIER fail silently and the ranker order is kept. If that
happened in our 0.377 / 0.372 runs, those results are confounded. **Not checked yet** (next steps, step 2). Our push
folders copy the baseline's pinned docker image, which may avoid it.

## 5. Work built in this session (all on the branch, untested on Kaggle)

### 5a. CMatch attempt 2 (`tools/cmatch/train_cmatch.py`, `tools/cmatch/cmatch.py`, commit 4ab9d53)
- Validation = whole held-out Murcko scaffolds (2%); decoys never come from held-out scaffolds.
- Fixed validation candidate lists (drawn once), so epochs and variants are scored on identical questions.
- Two variants, each keeping its best scaffold-val epoch (patience 6): `v1s` (v1 recipe) and `reg` (peak dropout 0.2,
  intensity jitter 0.15, dropout 0.2). The better one is saved as `cmatch.pt`; both as `cmatch_<name>.pt`.
- Logs seen-train MRR too: the gap to scaffold-val shows memorisation.
- Final log line: `CMATCH SUMMARY {...}` with `passes_gate` (scaffold-val isomer MRR ≥ 0.30).
- Notebook: `python tools/make_cmatch_notebook.py <any kernel-metadata.json> <out_dir>` (only the `.ipynb` is used).
- `CMatch(drop=...)` is new; old checkpoints still load (default 0.1).
- **Plan if it passes:** use it for dense retrieval (add candidates over the whole mass window), not re-ranking.
  **If it fails:** the contrastive line ends (two-attempts rule).

### 5b. Instrument check (`tools/diag/instrument_check.py`, commit 99d6218)
- CPU notebook: prints `test.parquet` columns and any instrument/ion-mobility fields; trains a spectrum→instrument
  LightGBM on a 200k train sample (GroupKFold by compound, reports held-out accuracy) and predicts the test mix.
- Final log line: `INSTRUMENT CHECK {...}`.
- Why: Tejas says the hidden test is **timsTOF**. If `test.parquet` has no `instrument_type` column, the CMatch
  re-rank (`tools/lb/cmatch_rerank.py`) set the timsTOF flag to 0 for every test spectrum, which would be a bug.
- Build: wrap the script in one code cell (see how it was done in this session) or import it in the browser.

### 5c. v6 overlay (`tools/lb/v6_final.py`, `--v6` in both builders, commit 75ce79a)
Ports the direction of public notebooks onto our own v4n (no fork; our code, ideas credited in EXPERIMENTS.md):
1. Base lists stay in ranker order (snapshot `V6_BASE_PRE` before the ICEBERG cell, restore after it).
2. Final fusion is rebuilt from `sub_v4_prefusion.csv` with one vote per metric identity per engine (window 40).
3. The fused list is re-scored by GLACIER where GLACIER has scores, and by ICEBERG only where GLACIER is blind
   (GLACIER only runs on `[M+H]+`).
4. Metric-safe tail: dedupe by the official metric identity (stereo removed, then canonical tautomer, InChIKey-14,
   plus heavy-atom composition), then fill empty slots from fused overflow, pre-fusion list, ranker top-60,
   engine 2 and the PubChem list. Any per-molecule error keeps the baseline row.
- Log line: `V6 FINAL {...}` (branch counts, filled slots, molecules with GLACIER/ICEBERG scores; zero GLACIER
  coverage means the simulators failed).
- Build: `python tools/make_lb_notebook.py <baseline.ipynb> <baseline kernel-metadata.json> <out_dir> --v6 --slug casmi26-v6-lb`
  and `python tools/make_sim2_notebook.py <baseline.ipynb> <out.ipynb> --source external --v6`, then
  `python tools/make_kaggle_push.py ...` for the sim2 folder (see NOTEBOOKS.md).
- Tested only on a stand-in notebook with stubbed RDKit (wiring and logic OK).
- Left out: their final fill from an extra DreaMS natural-product dataset (`dreams_np_mass.npy`, hengck23). Adding
  it needs an EXTERNAL_RESOURCES entry, a licence check and a forum/Data-tab ban check first.
- Their reported public-LB path (not verified by us): ICE+GL 0.402 → GLACIER-only 0.417; metric dedupe + tail
  0.425 → 0.433; identity fusion about 0.436; ICEBERG-where-GLACIER-is-blind 0.438 → about 0.44.

## 6. Next steps (in order)

1. **Get the 0.401 baseline notebook into the local repo folder** (never commit it):
   `python -m kaggle kernels pull t3jas06/<0.401-slug> -p base -m`.
2. **Check the simulator confound:** `python -m kaggle kernels output t3jas06/casmi26-v5-lb -p v5lb_out`, then
   `findstr /C:"ICE FAILED" /C:"GL FAILED" /C:"ICE meta" /C:"GL meta" v5lb_out\*.log`.
3. **Build v6** with `--v6` for the LB track (`casmi26-v6-lb`) and sim2 (`--source external --v6`).
4. **Push `casmi26-v6-lb`**, confirm `V6 FINAL` with non-zero GLACIER coverage in the log, then submit. Expect
   about 0.43–0.44 if our pipeline behaves like theirs.
5. **Push sim2 `--v6`** (about 1.5 GPU h); compare `summary.json` with the 103-q baseline (final 0.319, pre-fusion 0.362).
6. **Run CMatch attempt 2:** `kernels pull t3jas06/casmi26-cmatch-train -p cmtrain2 -m`, swap in the new `.ipynb`,
   push, check status; paste `CMATCH SUMMARY`. Save as a new `casmi26-cmatch` version only if `passes_gate`.
7. **Run the instrument check** (CPU, browser import, add the competition data); paste `INSTRUMENT CHECK` and
   `test columns:`.
8. **If the test is timsTOF without an `instrument_type` column:** fix the timsTOF flag in `cmatch_rerank.py`
   before any further CMatch use.
9. **Log every result** in `EXPERIMENTS.md` (date, change, commit, sim2 Δ, LB, decision).
10. **If v6 wins, make it the new baseline.** Then layer CMatch dense retrieval (if it passed) on top of v6, not v4n.
11. **Grow the practice set with timsTOF spectra** of compounds not in train: find public timsTOF MS/MS libraries,
    check licences and host bans, dedupe by InChIKey-14 + tautomer; aim for ≥ 500 questions.
12. **Later:** pool expansion for the 29% not-in-pool (check COCONUT/PubChem coverage first), learned stacker once
    ≥ 500 unseen questions exist, fill EXTERNAL_RESOURCES TODOs and MANIFEST hashes.

## 7. Known issues and open questions

- sim2's scorer (`metric_key` in `tools/sim2/sim_setup.py`) doesn't remove stereo before tautomer canonicalization,
  unlike the official metric and v6's identity. Small; fixing it changes logged baseline numbers, so do it together
  with a baseline rerun.
- Public LB covers about 130 of about 400 hidden molecules: noise ±0.035; use it only for yes/no decisions.
- `EXTERNAL_RESOURCES.md` still has TODOs (versions, licences of the baseline's input datasets, notebook URL).
- MANIFEST-style hashes for every file the scoring notebook loads are not kept yet.

## 8. Links

- Project-steps doc (history in about 150 words per step, notebooks table):
  https://claude.ai/code/artifact/9263a6f1-2769-4837-9805-dd9472d01ba2
- Earlier plain-language plan (from the "Kaggle review" session):
  https://claude.ai/code/artifact/1d63a626-bd61-4523-a26c-603b0b6d1f48
- Repo branch: https://github.com/Tejas-006/ENVEDA-CASMI-26/tree/claude/gracious-maxwell-04dmc2
- Public notebook analysed for v6 (uploaded, not in repo): canqiang/casmi26-e006-window-40-ice-fallback-true
  (lineage: obstacledeveloper V1.1, flexonafft v80, samanyu1808 "ICEBERG where GLACIER is blind").
