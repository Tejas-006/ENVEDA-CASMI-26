# External resources (Rules section 6 / winner obligations, section 2.8)

Each must be public and free to all participants. "TODO" items are unverified: fill them in before relying on the resource.

| Resource | Source | Version / ID | Licence | Used for |
|---|---|---|---|---|
| RDKit | https://www.rdkit.org | wheel in Kaggle input (TODO version) | BSD-3-Clause | Chemistry toolkit |
| LightGBM | https://github.com/microsoft/LightGBM | TODO | MIT | Candidate ranker |
| PubChem | https://pubchem.ncbi.nlm.nih.gov | TODO snapshot date | Public domain (NCBI) | PubChem candidate channel, popularity prior |
| DreaMS | https://github.com/pluskal-lab/DreaMS, https://zenodo.org/records/10997887 | Zenodo 10997887 | MIT | DreamsFP feature views inside the baseline's fe_v4 (not used by our own code) |
| ICEBERG (ms-pred) | https://github.com/coleygroup/ms-pred (MassSpecGym-trained weights linked in its README) | TODO version used by casmi26-iceberg | MIT | Post-ranker isomer re-scoring |
| GLACIER (ms-pred) | https://github.com/coleygroup/ms-pred (MassSpecGym-trained weights linked in its README) | TODO version used by casmi26-glacier | MIT | Post-ranker re-scoring |
| prvsiyan public Kaggle notebook (2026-09-16) | TODO notebook URL | 2026-09-16 | TODO (Kaggle notebook licence) | Analog-propagation engine components |
| CMatch (ours) | tools/cmatch/ in this repo | trained by casmi26-cmatch-train | n/a (own code; trained only on Competition Data) | Contrastive spectrum↔molecule score |
| MassSpecGym | https://huggingface.co/datasets/roman-bushuiev/MassSpecGym (downloaded by casmi26-massspecgym-download into private dataset casmi26-massspecgym) | HF revision d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a, data/MassSpecGym.tsv (231,104 spectra, 28,929 compounds) | MIT | sim2 practice questions: only compounds NOT in competition train (inchikey14 + tautomer-key dedupe) |
| Enveda-180 public release | https://zenodo.org/records/21346580 | Zenodo 21346580 (enveda-180.parquet.gz) | CC BY 4.0 | sim2 practice questions: timsTOF spectra of compounds NOT in competition train (host: allowed, discussion 745148 / 743569) |
| CASMI 2016 / 2022 contest data (via MassFormer data release) | https://zenodo.org/records/8399738 | casmi_2016.tgz, casmi_2022.tgz | CC BY 4.0 | planned: extra sim2 practice questions (not yet downloaded) |

## Kaggle inputs of the 0.401 baseline (public datasets by other participants; from kernel-metadata.json)

The baseline is a fork of a public notebook; its inputs are public Kaggle datasets. TODO: record each dataset's licence (dataset page → Licence).

| Kaggle dataset | Used for |
|---|---|
| ahmedberatozer/casmi26-v4b-models | v1 engine code, fe_v4 models, LightGBM ranker, MANIFEST.json |
| ahmedberatozer/casmi26-v3-models | PubChem channel code + FPNet A+B |
| ahmedberatozer/casmi26-v2-pool | Candidate pool (train ∪ COCONUT) and train-structure tables |
| ahmedberatozer/casmi26-pubchem-tier | PubChem SMILES/fingerprints for the PubChem channel |
| ahmedberatozer/casmi26-iceberg | ICEBERG re-scoring package |
| ahmedberatozer/casmi26-glacier | GLACIER re-scoring package |
| ahmedberatozer/casmi26-fpnet-full1 | Engine FPNet bank (fpnet_full1.pt) |
| prvsiyan/casmi26-fp-models-v2 | Engine-2 fingerprint models |
| prvsiyan/casmi26-ranker-features | Engine-2 ranker training rows (rank_train.npz) |
| prvsiyan/coconut-casmi26-candidates | COCONUT 2.0 candidates + fingerprints |
| prvsiyan/chebi-lipidmaps-casmi26 | ChEBI + LIPID MAPS candidates |
| megayak/casmi26-simulated-ranker-rows | Engine-2 simulation rows (sim_rank_rows_nofp.npz) |
| dmitriigluzdov/casmi26-pubchem-popularity-prior | PubChem popularity prior |
| metric/rdkit-2026-3-3-wheel | RDKit 2026.3.3 wheel |

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
