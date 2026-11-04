# ACES: matched semi-supervised echocardiographic segmentation

[中文说明](README_CN.md)

Code and experiment records for **Effect of anatomical pseudo-label filtering on semi-supervised echocardiographic segmentation: A matched evaluation**.

ACES combines a compact U-Net-style backbone, a known-view-conditioned graph, an EMA teacher, and a training-only anatomical pseudo-label filter. The recorded evaluation uses CAMUS A2C/A4C images, four output classes, 25%/100% patient-label budgets and seeds 17, 29 and 43.

The package contains **33 completed run records across 11 configurations**, their patient-level and frame/class-level metrics, training histories, calibration summaries, and fixed patient/annotation lists. Original ultrasound images, reference masks, prepared arrays and trained checkpoints are obtained or generated separately.

## Recompute the reported tables without a GPU

Use Python 3.10–3.12 from the repository root:

    python -m pip install -r requirements-analysis.txt
    python scripts/verify_release.py
    python scripts/reproduce_results.py

On Windows, double-click **REPRODUCE_RESULTS.bat**. Its first execution creates a small analysis environment and installs NumPy; subsequent runs use that environment.

The script checks the supplied metric records and writes Tables 1–4 as CSV, paired patient-bootstrap intervals and figure-source CSVs to **outputs/reproduced/**. Archived records are not overwritten.

This is record-level reproduction: it recomputes aggregation and bootstrap intervals from recorded metrics. Image-level inference and pixel-level metric recomputation require CAMUS and model checkpoints. NLL/Brier values come from saved summaries; ECE is additionally checked against saved calibration-bin masses.

## Recorded main results

| 25% labels | Patient macro Dice, mean ± sample SD |
| --- | --- |
| Supervised U-Net-style | 0.8961 ± 0.0005 |
| Confidence-only Mean Teacher | 0.8979 ± 0.0021 |
| SegFormer-B0 | 0.8928 ± 0.0046 |
| ACES | 0.8978 ± 0.0003 |

Exact values and the other configurations are in **results/published/**. Three training seeds are averaged within each patient before paired patient bootstrapping. The bootstrap uses 2,000 replicates and seed 20261029.

## Repository contents

| Location | Contents |
| --- | --- |
| aces/ | Original model, data, prior, loss, training, calibration and evaluation code |
| launcher.py, bootstrap.ps1, *.bat | Original Windows training and resume entry points |
| settings.json | Fresh-run configuration using the archived explicit patient split |
| configs/recorded_settings.json | Settings saved with the completed experiment records |
| data/splits/ | 400/50/50 patient lists and fixed 25%/100% annotation subsets |
| data/metadata/ | Portable metadata snapshots, source-relative image names and cache hashes |
| data/DATA_DICTIONARY.md | Labels, record fields, metrics and units |
| results/archived_runs/ | 33 run directories with recorded CSV/JSON outputs |
| results/published/ | Original cross-run summaries and paired comparisons |
| results/tables/ | Tables and analysis CSVs recomputed during repository preparation |
| scripts/reproduce_results.py | NumPy-only aggregation and paired-bootstrap reproduction |
| scripts/verify_release.py | File integrity and patient-split checks |
| tests/ | Original test coverage, one portable disk-space fixture adjustment, and repository record checks |
| docs/REPRODUCIBILITY.md | Code provenance, numerical precision and validation scope |

Fresh training writes to **runs/** and **results/**; the historical run set stays under **results/archived_runs/**. Generated files, raw images, checkpoints and virtual environments are excluded by .gitignore.

## Data

Obtain CAMUS through the [official project](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) and follow the provider's applicable access and use terms. Place the downloaded image files under **data/camus/**, retaining their patient folders. The loader supports NIfTI and MHD/MHA images.

The supplied split is the **actual 400/50/50 split recorded in this experiment**. The CAMUS project describes its original release as 450 development patients and 50 testing patients; this repository publishes its exact experiment lists rather than treating every CAMUS release as having the same split. Original split-file hashes are retained in data/metadata/original_partition_provenance.json.

Output labels are 0 = background, 1 = LV cavity, 2 = myocardium, 3 = left atrium. After checking the label meanings against your downloaded release, set **data.label_map_confirmed** to **true** in settings.json. Keep the explicit split_file setting unchanged when reproducing this experiment.

Prepare local data without invoking the downloader:

    python -m aces.cli prepare --root data/camus --out prepared/camus --settings settings.json

All 500 listed patients must be present. Metadata snapshots in data/metadata/ are historical records, not ready-to-load image manifests; preparation rebuilds a manifest with your own local paths.

## Train on an NVIDIA GPU

The recorded environment used Windows, Python 3.12.10, PyTorch 2.7.1+cu128 and an RTX 5060 Laptop GPU. The version is pinned for this implementation, not presented as the latest PyTorch release.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Linux: source .venv/bin/activate
    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements.txt
    python -m aces.cli doctor
    python -u launcher.py --mode main

Windows one-click entries:

- START.bat: ACES, 25% labels, seed 17.
- RUN_ALL.bat: all 33 configurations/seeds, with checkpoint resume.
- EVALUATE_ONLY.bat: evaluate the primary run using an existing local best.pt.
- RUN_TESTS.bat: software tests.
- SMOKE_TEST.bat: small synthetic-data software test, kept separate from experiment results.

For all jobs after setup:

    python -u launcher.py --mode suite

For one explicit run:

    python -m aces.cli train --manifest prepared/camus/manifest.json --out runs/aces_labels025_seed17 --method aces --seed 17 --labels 0.25
    python -m aces.cli calibrate --checkpoint runs/aces_labels025_seed17/best.pt --manifest prepared/camus/manifest.json --out runs/aces_labels025_seed17/temperature.json
    python -m aces.cli evaluate --checkpoint runs/aces_labels025_seed17/best.pt --manifest prepared/camus/manifest.json --out runs/aces_labels025_seed17/test --temperature-json runs/aces_labels025_seed17/temperature.json

Supported method names are unet, mean_teacher, segformer_b0, graph_supervised, graph_confidence, graph_gaussian, aces_view_agnostic and aces. SegFormer-B0 is randomly initialized; it does not download ImageNet weights. Current code uses FP32 for SegFormer; one historical seed switched from AMP at epoch 88, recorded in its numerical_precision.json.

AP-Net is not included as a validated reproduction. HMC-QU external evaluation was not run; the mapping template is provided for future verified external use.

## Tests

After installing training dependencies:

    python -m unittest discover -s tests -v

GitHub Actions runs the record checks and CPU software tests. It does not run full GPU training. See docs/VALIDATION.md for the checks actually executed while preparing this package.

## References and rights

Please cite the CAMUS dataset paper: [Leclerc et al., IEEE TMI 2019](https://doi.org/10.1109/TMI.2019.2900516), and the ACES manuscript title above when referring to this implementation. CITATION.cff contains the software citation metadata.

No additional reuse license was selected on behalf of the copyright holders. See NOTICE.md. CAMUS and third-party dependencies retain their own terms.
