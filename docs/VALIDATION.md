# Validation during repository preparation

Validation date: 2026-11-04.

| Check | Observed result |
| --- | --- |
| Source archive against experiment's recorded v1.4 source hashes, before packaging | All 43 listed files match |
| Training/model code under aces/ after packaging | All 12 Python modules remain byte-identical |
| Original experiment-collection audit | 181,492 record consistency checks, zero issues |
| Archived run files | 445 CSV/JSON files preserved; only path strings normalized where necessary |
| Patient partitions | 400/50/50, mutually disjoint |
| Fixed annotation subset | Same 100 visible training patients for every 25% run |
| Run/configuration coverage | 33 runs, 11 configurations, seeds 17/29/43 |
| Per-frame/class metric records reaggregated | 19,800 rows |
| Published paired intervals | All 9 comparisons reproduced within absolute tolerance 1e-10 |
| Figures' source data | Recomputed acceptance/validation curves and 50-patient scatter CSVs |
| Recorded scatter check | Correlation 0.9664424218542691; ACES higher in 24 patients |
| CPU software tests | 67 passed: 64 retained original cases and 3 archive checks |
| Python syntax | All source files parse successfully |
| Release-file integrity | Passed for all files listed in FILE_MANIFEST.json |
| Local user paths and common credential patterns | None found in release text files |
| Git ignore rules | Raw images, checkpoints, environments, runtime outputs and logs excluded; archived metrics and fixed splits remain included |

CPU checks used Python 3.12.14, PyTorch 2.7.1+cpu, NumPy 1.26.4, SciPy 1.13.1, Pillow 11.2.1, SimpleITK 2.4.1 and Transformers 4.51.3. The original experiment environment is separately recorded in each run's hardware.json.

The first CPU-test invocation exposed a host-dependent fixture: an evaluation-orchestration test invoked the production 15 GiB disk-capacity check. That test now mocks free capacity, while production capacity checks are unchanged. The complete 67-test suite then passed.

This session did not repeat the full 33 CAMUS GPU training jobs, retrieve original ultrasound images or verify absent checkpoint/prediction arrays. The Windows batch/PowerShell launchers were inspected and preserved; Windows/CUDA execution was not repeated in the Linux CPU validation environment.
