# External resources (Rules section 6 / winner obligations, section 2.8)

Each must be public and free to all participants. "TODO" items are unverified: fill them in before relying on the resource.

| Resource | Source | Version / ID | Licence | Used for |
|---|---|---|---|---|
| RDKit | https://www.rdkit.org | 2026.03.3 (main notebook, `metric/rdkit-2026-3-3-wheel`, asserted by E006); 2025.03.6 inside the ICEBERG / GLACIER subprocesses (their bundled wheel) | BSD-3-Clause | Chemistry toolkit |
| LightGBM | https://github.com/microsoft/LightGBM | the pinned Kaggle image's build (image `gcr.io/kaggle-private-byod/python@sha256:37c64f7d...`; exact version not printed yet) | MIT | Candidate ranker |
| PubChem | https://pubchem.ncbi.nlm.nih.gov | via `ahmedberatozer/casmi26-pubchem-tier` (dataset of 2026-09-23: CID-SMILES + CID-Mass, ~106M structures) and `dmitriigluzdov/casmi26-pubchem-popularity-prior` | Public domain (NCBI) | PubChem candidate channel, popularity prior |
| DreaMS | https://github.com/pluskal-lab/DreaMS, https://zenodo.org/records/10997887 | Zenodo 10997887 | MIT | DreamsFP feature views inside the baseline's fe_v4 (not used by our own code) |
| ICEBERG (ms-pred) | https://github.com/coleygroup/ms-pred (MassSpecGym-trained weights linked in its README) | ICEBERG 2.1, MassSpecGym `msg_all` checkpoint (per `ahmedberatozer/casmi26-iceberg`, 2026-09-26) | MIT | Post-ranker isomer re-scoring |
| GLACIER (ms-pred) | https://github.com/coleygroup/ms-pred (MassSpecGym-trained weights linked in its README) | MassSpecGym checkpoint, stripped (per `ahmedberatozer/casmi26-glacier`, 2026-09-28) | MIT | Post-ranker re-scoring ([M+H]+ only) |
| prvsiyan public Kaggle notebook (2026-09-16) | https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline | 2026-09-16 | Apache-2.0 (Kaggle default for public notebooks) | Analog-propagation engine components (engine 2) |
| CMatch (ours) | tools/cmatch/ in this repo | trained by casmi26-cmatch-train | n/a (own code; trained only on Competition Data) | Contrastive spectrum↔molecule score |
| MassSpecGym | https://huggingface.co/datasets/roman-bushuiev/MassSpecGym (downloaded by casmi26-massspecgym-download into private dataset casmi26-massspecgym) | HF revision d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a, data/MassSpecGym.tsv (231,104 spectra, 28,929 compounds) | MIT | sim2 practice questions: only compounds NOT in competition train (inchikey14 + tautomer-key dedupe) |
| Enveda-180 public release | https://zenodo.org/records/21346580 | Zenodo 21346580 (enveda-180.parquet.gz) | CC BY 4.0 | sim2 practice questions: timsTOF spectra of compounds NOT in competition train (host: allowed, discussion 745148 / 743569) |
| CASMI 2016 / 2022 contest data (via MassFormer data release) | https://zenodo.org/records/8399738 | casmi_2016.tgz, casmi_2022.tgz | CC BY 4.0 | planned: extra sim2 practice questions (not yet downloaded) |
| MS2DeepScore 2.11 + dual-mode model | https://github.com/matchms/ms2deepscore, https://zenodo.org/records/17826815 | ms2deepscore 2.11.0; model v1 (2025-12-05, for MS2DeepScore >= 2.6) | Apache-2.0 (code), CC BY 4.0 (model); trained on GNPS / MoNA / MassBank / MSnLib, no NIST | evaluated offline only (`casmi26-ms2ds-package`, `casmi26-ms2ds-eval`); **not in any submission** |
| GNPS / MassBank / MoNA newer releases | GNPS ALL_GNPS_NO_PROPOGATED (export 2026-03-29), MassBank-data release 2026.03, MoNA predefined export | see `casmi26-newer-libraries` report | GNPS community CC0; MoNA CC BY; MassBank per record (host: CC BY-SA and NC fine) | counted only (2,248 not-in-train compounds with usable MS2, 0 timsTOF); **not in any submission** |
| DiffMS (desk check) | https://github.com/coleygroup/DiffMS, https://zenodo.org/records/15122968 | — | MIT (code + weights) | not downloaded, not used (EXPERIMENTS.md 2026-10-19) |

## Kaggle inputs of the E006 fork (public datasets by other participants; from kernel-metadata.json)

Licences and last-update dates read with `kaggle datasets metadata` / `kaggle datasets list` on 2026-10-20. All are
public and free. CC BY-NC-SA and "other" are acceptable per the host (discussion/748363) and are exempt from the MIT
winner grant (Rules section 6). Submitted notebook versions keep the dataset versions attached when they were pushed;
**a new push attaches the latest version** (see the popularity-prior note).

| Kaggle dataset | Licence | Last updated | Used for |
|---|---|---|---|
| ahmedberatozer/casmi26-v4b-models | other (code MIT; see LICENSE-NOTICE.txt) | 2026-09-26 | v1 engine code (`code/casmi`), fe_v4 models, LightGBM ranker, MANIFEST.json |
| ahmedberatozer/casmi26-v3-models | other (code MIT) | 2026-09-23 | PubChem channel code + FPNet A+B |
| ahmedberatozer/casmi26-v2-pool | other (data only; reading code MIT) | 2026-09-21 | Candidate pool (train ∪ COCONUT) and train-structure tables |
| ahmedberatozer/casmi26-pubchem-tier | other (NCBI PubChem public data) | 2026-09-23 | PubChem SMILES/masses for the PubChem channel |
| ahmedberatozer/casmi26-iceberg | other (ms-pred MIT) | 2026-09-26 | ICEBERG re-scoring package |
| ahmedberatozer/casmi26-glacier | other (ms-pred MIT) | 2026-09-28 | GLACIER re-scoring package |
| ahmedberatozer/casmi26-fpnet-full1 | other (weights; code MIT) | 2026-09-29 | Engine FPNet bank (fpnet_full1.pt) |
| prvsiyan/casmi26-fp-models-v2 | CC BY-NC-SA 4.0 | 2026-09-16 | Engine-2 fingerprint models |
| prvsiyan/casmi26-ranker-features | CC BY-NC-SA 4.0 | 2026-09-17 | Engine-2 ranker training rows (rank_train.npz) |
| prvsiyan/coconut-casmi26-candidates | CC BY-NC-SA 4.0 | 2026-09-15 | COCONUT 2.0 candidates + fingerprints |
| prvsiyan/chebi-lipidmaps-casmi26 | CC BY-NC-SA 4.0 | 2026-09-16 | ChEBI + LIPID MAPS candidates |
| megayak/casmi26-simulated-ranker-rows | CC0-1.0 | 2026-09-20 | Engine-2 simulation rows (sim_rank_rows_nofp.npz) |
| dmitriigluzdov/casmi26-pubchem-popularity-prior | other | **2026-10-19 18:47 UTC** (all files re-uploaded) | PubChem popularity prior (popularity re-rank of base and PubChem lists) |
| metric/rdkit-2026-3-3-wheel | unknown (RDKit itself BSD-3) | 2026-09-08 | RDKit 2026.3.3 wheel |
| eveliaveldrine/rdkit-wheels | Apache-2.0 | 2026-10-04 | RDKit 2025.3.6 / numpy / Pillow cp313 wheels (fallback image) |

**Popularity-prior update:** every submission up to 57404158 (pushed before 2026-10-19 18:47 UTC) used the previous
version. The next push will attach the new one: before comparing a new submission with the earlier ones, check the
log line `popularity re-ranked lists N` and diff its submission against a re-push of the unchanged E006 fork.

## Host clarifications on external data (competition forum, read 2026-10-17)

Rules section 6: external data and models are allowed if publicly available and free to all participants, unless the
host prohibits them. Winner licence: open source (MIT); input data or pretrained models with incompatible licences are
exempt from that grant. Foundational rules: no submission may infringe third-party IP.

| Topic | Host answer | Thread |
|---|---|---|
| Newer GNPS / MassBank / MoNA releases (records beyond the train snapshot), CC0 / CC BY / CC BY-SA records | "Newer releases of public datasets are fine to use and CC BY-SA records are acceptable." (Marie Killian) | discussion/745228 |
| Enveda-180 Zenodo release, incl. 816 structures absent from train | "They likely were filtered out in the train.parquet preparation. Data in the Zenodo release of Enveda-180 are fine to use." | discussion/745148 |
| CC BY-NC-SA, "other" or unspecified licences on freely downloadable data | "Those should be fine ... we're mostly worried about datasets with very restrictive redistribution licenses, like NIST" (David Healey) | discussion/748363 |
| Open-weight models | fine "as long as the open weight model is distributed in compliance with the license for the data it was trained on"; NIST-trained weights are not | discussion/742275 |
| GNPS-NIST14-MATCHES | allowed for library matching and training (long-standing GNPS library); newly created NIST-derived data are not | discussion/745832 |
| Candidate-derived formulas as model inputs | permitted | discussion/745832 |

Consequences for this repo: MIST's released weights (trained partly on NIST) are not usable; MVP (no licence file,
trained on MassSpecGym) is probably acceptable under 748363 but should be asked about specifically before use.

## E006 fork (from 2026-10-18)

Base notebook: antriksharya/casmi26-e006-e2rank2 (public Kaggle notebook, Apache 2.0 by default), lineage canqiang
(E006), flexonafft (v80 metric-identity fusion), samanyu1808 (ICEBERG where GLACIER is blind), obstacledeveloper (metric
audit, safe tail), skomuro (PubChem promotion rule), dmitriigluzdov (popularity prior), prvsiyan / megayak (engine 2),
ahmedberatozer (v4n engine, pool, models), ms-pred (ICEBERG / GLACIER).

| Extra input vs our 0.401 baseline | Licence | Decision |
|---|---|---|
| eveliaveldrine/rdkit-wheels | Apache-2.0 (contents: RDKit 2025.3.6 BSD-3, numpy BSD, Pillow HPND, cp313) | kept (fallback for Python 3.13 images) |
| hengck23/hengck23-dreams-enveda-casmi26 | MIT | dropped: unused by E006's code (optional dreams_np_*.npy files absent); DreaMS checkpoint training data unverified |

| tools/lb/frag_gap.py (vendored) | public notebook antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert, cell 32 | 2026-10-18 copy | Apache-2.0 (frag_rescore.py credited to Huseyin Emre Aksoy) | A/B arm `frag_gap`: MetFrag-lite re-rank for molecules without simulator coverage |
