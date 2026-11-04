# Reproducibility and packaging

## Source identity

The available v1.4 source archive matches every file hash listed in the experiment collection's SOURCE_MANIFEST.json before packaging. Original model/training code under aces/, launcher.py, downloader and Windows entry points are preserved byte-for-byte. docs/recorded_code_manifest.json retains the recorded source-file hashes.

The original 64 test cases are retained. One evaluation-entry test now mocks free disk capacity, just as it already mocks GPU/data access; otherwise a small CPU fixture fails on a host with less than the full-training 15 GiB space requirement. Production disk checks are unchanged. Three checks on archived source/cohort records were added.

The actual deployed source bytes were not part of the result-collection ZIP; the source/hash agreement identifies the corresponding available source archive. Archive SHA-256 values and packaging transformations are in release_provenance.json.

Added files provide documentation, explicit split lists, record-level analysis, integrity checks and CI. settings.json now points to the explicit recorded patient split; numerical training settings are unchanged. configs/recorded_settings.json preserves the historical settings.

## Fresh training and historical results

Historical records are under results/archived_runs/. Original cross-run outputs are under results/published/. New runs write to runs/ and can be summarized with:

    python -m aces.cli report --runs runs --out outputs/new_run_report

Do not mix newly trained checkpoints with historical metric directories. Historical model checkpoints and optimizer states were not supplied. A historical completion marker alone cannot resume a model without its checkpoint.

Explicit split lists preserve cohort membership across machines. Local source paths and explicit split provenance change a newly prepared manifest's digest; historical digest strings are retained as historical identifiers. This expected path/config identity change is not resolved by rewriting old checkpoint/metric identities.

Portable metadata snapshots replace Windows user-directory paths with source-relative paths. All patient IDs, metric values, class mappings, configuration numbers, historical digests and precision records are retained. Terminal logs were excluded from the public package.

## Numerical implementation

The graph projects the concatenation of the 512-channel encoder feature, 2D position and 16D view embedding into 128 internal channels; four attention heads operate there. The output projects back to encoder-channel size for the residual connection. The auxiliary head reads the encoder bottleneck before the graph.

The unlabeled loss is normalized over positive weights across a microbatch, with max(1,sum(weights)) in the denominator. Gradient accumulation averages microbatch losses; it does not retrospectively normalize over all pixels of the accumulated effective batch.

The teacher generates detached pseudo labels, and its parameters update by EMA. Teacher weights are evaluated for all teacher-student methods. Anatomical score statistics use only labeled training patients and remain frozen. Score gating/weighting is a training-only operation.

SegFormer-B0 is constructed from its configuration and random weights. Current code enforces FP32 for it. Historical seed17 first used AMP and switched to FP32 at epoch88/update51; this history is retained rather than described as an entirely FP32 historical run.

## Scope of validation

Record-level checks can verify cross-file consistency, arithmetic aggregation, saved-bin ECE and patient-bootstrap intervals. They do not establish that recorded metrics match an absent prediction array. Full training and native-spacing pixel metrics require raw images, reference masks and checkpoints.

The original 64 software tests use small generated fixtures for model/gradient/data/restore/numerical behavior. They are software tests, not new CAMUS results. CI uses CPU PyTorch and does not validate CUDA kernels or rerun the 33 full GPU jobs.

AP-Net reproduction and HMC-QU external evaluation are not reported as completed.
