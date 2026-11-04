# Data and experiment-record dictionary

## Raw and derived data

CAMUS images and reference masks are acquired from the provider separately. The repository contains public CAMUS identifiers, portable metadata and derived experiment metrics, not original image pixels.

The release contains 500 distinct patient IDs: 400 training, 50 validation and 50 testing. The split lists are mutually disjoint. The 25% budget reveals ED/ES masks from a fixed 100 training patients (400 labeled frames); 100% reveals all 400 training patients (1,600 labeled frames).

Training uses 8,000 frames. The 25% unlabeled pool has 7,600 frames and the 100% pool has 6,400 intermediate frames. Validation and testing each have 200 labeled frames. Intermediate frames stay in the training partition.

## Data metadata

| Field | Meaning |
| --- | --- |
| patient_id | Anonymous CAMUS ID; not a hospital identifier |
| case_id | Patient, acquisition and phase/frame identifier |
| view | A2C or A4C |
| phase | ED, ES or an intermediate-frame designation |
| split | train, val or test |
| source_image, source_mask | Source-relative paths in the portable historical snapshot |
| path | Relative prepared-array path; arrays must be rebuilt locally |
| has_mask | Whether a reference mask is available for this record |
| spacing_xy | Original physical pixel spacing in millimeters |
| quality | Good, Medium or Poor, from acquisition metadata |
| cache_sha256 | Hash of the historical prepared array; the array itself is not distributed |
| source_signature | Historical source-data fingerprint |

Label IDs are 0=background, 1=left ventricular cavity, 2=myocardium and 3=left atrium.

## Run directories

Directory names follow method_labels025_seed17 or method_labels100_seed17.

| File | Contents |
| --- | --- |
| contract.json | Method, seed, fraction, frozen training configuration, labeled IDs and historical manifest hash |
| annotation_budget.json | Visible patient IDs and labeled/unlabeled frame counts |
| anatomical_prior.json | Fitted per-view anatomical score statistics, when used |
| history.csv | Epoch, mean loss, accepted fraction, validation macro Dice, checkpoint eligibility and learning rate |
| hardware.json | Recorded software/GPU environment |
| training_complete.json | Completion and validation-selection metadata; resources are historical measurements |
| numerical_precision.json | SegFormer precision policy and any historical precision transition |
| latency.json | Forward-only batch-one timing protocol and values |
| temperature.json | Validation-only temperature fitting metadata |
| test/per_frame_class.csv | 200 test frames × 3 foreground classes = 600 metric rows |
| test/per_patient_class.csv | 50 patients × 3 classes = 150 rows |
| test/per_patient.csv | 50 patient-level macro metric rows |
| test/stratified.csv | Per-view, phase and quality summaries |
| test/summary.json | Test summary and raw/scaled probability calibration |
| test/evaluation_identity.json | Checkpoint and historical manifest hash linkage |

## Metrics

Dice and IoU lie in [0,1]. Reported macro values average the three foreground classes with equal patient weighting. HD95 and ASSD use native spacing and are in millimeters. The per-frame recording_id field is preserved exactly; the analysis script follows the recorded aggregation.

accepted_pixel_fraction is the share of unlabeled pixels with positive pseudo-label weight, not the mean weight. Table 2 averages epochs 21–200 and then summarizes the three seeds. Supervised rows have no applicable acceptance rate.

Calibration includes all four classes. NLL is multiclass negative log-likelihood; Brier is the recorded multiclass score; ECE uses 15 equal-width confidence bins. Calibration-bin masses permit ECE verification; per-pixel probabilities needed to refit temperature or recalculate NLL/Brier are not included.

Quality is attached to view sequences. A patient with different grades in its two views belongs to multiple groups. Good/Medium/Poor groups contain 30/26/10 patients and 47/38/15 sequences. The group-specific metrics aggregate the frames having that grade.

Saved training completion timing and peak-memory fields can reflect only the last resumed invocation. Peak memory is not included in the public main-result table or treated as a comparable resource endpoint.
