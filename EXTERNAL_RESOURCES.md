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
