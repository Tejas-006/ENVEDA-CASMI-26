# Handoff: ENVEDA CASMI 2026, state as of 2026-10-14

Everything a new session needs to continue. Read `CLAUDE.md` first: its competition rules and validation gate
apply to every change. Results live in `EXPERIMENTS.md`, the Kaggle workflow in `NOTEBOOKS.md`, and external
resources in `EXTERNAL_RESOURCES.md`.

## 1. Where things stand

- **Best submission: 0.425 public LB** (v6 = v4n + GLACIER-first re-scoring, identity fusion, metric-safe tail;
  `t3jas06/casmi26-v6-lb` version 1). sim2 0.379 (+0.060 vs v4n). **v6 is the new baseline**: build on it, not v4n.
- **CMatch attempt 2 passed** (scaffold-val isomer MRR 0.665). Next use: dense retrieval on top of v6, not re-ranking.
- Previous best: 0.401 (v4n, notebook `t3jas06/baseline`, a fork of a public notebook).
- **Branch with all work:** `claude/gracious-maxwell-04dmc2` (it contains `claude/vibrant-edison-b7wd63` plus later
  commits). `main` only has the first 5 commits; this branch is under review as https://github.com/Tejas-006/ENVEDA-CASMI-26/pull/2 (new commits to the branch update it).
- **Repo is public.** Never commit notebooks (`*.ipynb` is gitignored), baseline code, Competition Data or
  anything derived from it (sim2 rows, spectrum caches).

## 2. How we work

- Claude writes code and notebook generators in this repo.
- **A local Claude Code session on Tejas's PC can drive Kaggle directly** (since 2026-10-13): the Kaggle CLI in
  `%USERPROFILE%\kaggle-ml\.venv` (2.2.4, OAuth as `t3jas06`) pushes notebooks, polls status, downloads logs and
  outputs, creates private datasets and submits. Cloud/web Claude sessions still can't reach Kaggle.
  Free T4, about 30 GPU h/week. Competition: `enveda-CASMI26-molecule-id-mass-spectra`.
- Typical loop: build a push folder (`kernel-metadata.json` + `.ipynb`) in `kaggle-ml\<name>`, then
  `kaggle kernels push -p <folder>`, poll `kaggle kernels status t3jas06/<slug>`, read the log from
  `kaggle kernels output t3jas06/<slug> -p <out> --file-pattern ".*\.log$"`. **Submissions need Tejas's OK each time**:
  `kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k t3jas06/<slug> -v <N> -f submission.csv -m "<desc>"`.
- A push run uses the visible stand-in test (`RERUN False`): only ~12 molecules are processed and the rest are
  one-SMILES placeholders, so `v6 kept baseline row ... no valid candidates` for ~388 rows there is expected.
- Datasets created from notebook output in the browser (e.g. `casmi26-cmatch`) can't take CLI-uploaded versions
  ("Incompatible Dataset Type"); create a new private dataset instead.
- Code-competition limits: internet off, ≤ 9 h, 5 submissions/day, 2 final picks. Pipeline is about 6.5 h on
  about 400 hidden molecules; keep any addition under about 8.5 h total.
- Keep rule (EXPERIMENTS.md): keep a change if sim2 gains ≥ 0.01 AND the public LB drops ≤ 0.02 AND runtime fits.
  Two attempts per direction, then move on. Never tune constants on the public LB. Final picks: one LB-best,
  one sim2-best.

## 3. Kaggle assets

| Asset | What | Status |
|---|---|---|
| `t3jas06/baseline` | 0.401 v4n pipeline: v1 engine + LightGBM ranker, engine 2 (analog propagation), PubChem channel, ICEBERG + GLACIER, final RRF fusion. Input to every builder | Reference |
| `t3jas06/casmi26-v6-lb` | v6 = baseline + `--v6` | **LB 0.425 (best)** |
| `t3jas06/casmi26-sim2-build` | Practice test (sim2); `--source external` uses unseen MassSpecGym compounds | v8: v4n baseline 0.319 (103 q); v9: v6 0.379 |
| `t3jas06/casmi26-massspecgym` (private dataset) | MassSpecGym HF revision d2e86d0 | Done |
| `t3jas06/casmi26-cmatch-train` | CMatch training notebook | Attempt 2 done: passes (0.665) |
| `t3jas06/casmi26-cmatch` (private dataset) | CMatch v1 weights (`cmatch.pt`, `cmatch.py`, `cmatch_meta.json`) | v1 only; used by `casmi26-v5-lb` |
| `t3jas06/casmi26-cmatch-v2` (private dataset) | CMatch attempt 2: `cmatch_out/cmatch.pt` (= v1s winner), `cmatch_v1s.pt`, `cmatch_reg.pt`, meta; `cmatch_src/cmatch.py` | Created 2026-10-14 |
| `t3jas06/casmi26-v5-lb` | Baseline + CMatch v1 re-rank | LB 0.372 |
| `t3jas06/casmi26-v5-lb-nofusion` | Baseline without final engine-2 fusion | LB 0.377 |
| `t3jas06/casmi26-instrument-check` | CPU check of the test set's instrument | Done: 100% timsTOF, column present |

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
| 10-11 | CMatch attempt 2 (scaffold hold-out val, variants v1s and reg) | v1s 0.665, reg 0.656 scaffold-val isomer MRR; seen-train 0.823 | Passes the 0.30 gate on an honest split. Weights in `casmi26-cmatch-v2`. |
| 10-13 | Simulator-confound check | ICE and GL `status: ok` in the 0.372 and 0.377 logs | Ruled out: those losses were real. |
| 10-13 | v6 on sim2 (103 q) | final 0.379 (+0.060), top1 0.291, recall@25 0.524 (−0.03) | Passes sim2 gate. |
| 10-13 | v6 on the LB (`casmi26-v6-lb` v1) | **0.425** (+0.024) | **Worked: new best and new baseline.** sim2 and LB agree. |

**Possible confound:** recent public notebooks found that on the newer Python 3.13 Kaggle image, ICEBERG's bundled
RDKit 2025.3.6 wheel (cp312) won't install, so ICEBERG and GLACIER fail silently and the ranker order is kept. If that
happened in our 0.377 / 0.372 runs, those results are confounded. **Checked 2026-10-13: not the case.** Both logs
show ICE and GL `status: ok` with RDKit 2025.03.6; our push folders copy the baseline's pinned docker image. Keep
copying that image.

## 5. Work built in the 2026-10-12 session (now run on Kaggle: v6 kept, CMatch attempt 2 passed)

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

Done 2026-10-13/14: baseline pulled (`t3jas06/baseline`), confound ruled out, v6 built, run and submitted (LB 0.425,
sim2 0.379, kept), CMatch attempt 2 run (passes) and saved as `casmi26-cmatch-v2`, instrument check run (test is
timsTOF; no CMatch bug).

1. ~~Instrument check~~ done 2026-10-14: `test.parquet` has `instrument_type`, 100% `timsTOF` (classifier agrees:
   95.4%). `cmatch_rerank.py` reads it correctly (lowercased -> `timstof`), so no fix needed. Any new CMatch code must
   read the column the same way. sim2 has no timsTOF questions, which makes step 5 more important.
2. **Investigate v6's recall@25 drop** (0.55 -> 0.524 on sim2): find the questions where the truth was in v4n's top 25
   but not v6's (compare `sub_final.csv` of sim2-build v8 vs v9) and check whether dedupe or the tail fill pushed it out.
3. **CMatch dense retrieval on top of v6** (EXPERIMENTS step 4): CMatch v2 top-K over the whole mass window, union
   into the candidate lists before the ranker. Gate: sim2 recall@25 +3 points, runtime <= +20 min; then LB.
4. **Log every result** in `EXPERIMENTS.md` (date, change, commit, sim2 Δ, LB, decision).
5. **Grow the practice set with timsTOF spectra** of compounds not in train: find public timsTOF MS/MS libraries,
   check licences and host bans, dedupe by InChIKey-14 + tautomer; aim for >= 500 questions.
6. **Later:** pool expansion for the 29% not-in-pool (check COCONUT/PubChem coverage first), the DreaMS NP final fill
   the v6 source notebooks use (needs an EXTERNAL_RESOURCES entry, licence and ban check), learned stacker once
   >= 500 unseen questions exist, fill EXTERNAL_RESOURCES TODOs and MANIFEST hashes.
7. **Final picks:** currently LB-best = v6 (0.425); sim2-best = v6 (0.379).

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
- Pull request: https://github.com/Tejas-006/ENVEDA-CASMI-26/pull/2
- Public notebook analysed for v6 (uploaded, not in repo): canqiang/casmi26-e006-window-40-ice-fallback-true
  (lineage: obstacledeveloper V1.1, flexonafft v80, samanyu1808 "ICEBERG where GLACIER is blind").
