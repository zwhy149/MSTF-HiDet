# MSTF-HiDet: AE9 Review-Ready Code and Results Package

This repository provides the public code release of the MSTF-HiDet framework for hierarchical diagnosis of external short-circuit conditions in LFP batteries. The package is organized as a review-ready minimum reproducible repository: it preserves the core training and inference code, the repeated-average manuscript-facing evaluation, the main figures and tables used in revision, and the auxiliary files needed to track the train/validation/test split logic.

## What This Repository Covers

The public code directly supports the three-class diagnostic setting used in the AE9 package:

- `Normal`
- `Charging short-circuit`
- `Full-SOC Resting Short-circuit`

The manuscript discusses a broader experimental context, but this public release should be cited according to the scope actually exposed by the code and interfaces in this repository. A concise scope note is provided in [docs/SCENARIO_SCOPE.md](docs/SCENARIO_SCOPE.md).

## Main Result Reported in This Package

The manuscript-facing primary result in this repository is the repeated Route B evaluation over five random seeds:

- MSTF-HiDet L2 accuracy: `0.981 ± 0.014`
- MSTF-HiDet L2 macro-F1: `0.983 ± 0.013`

Baseline averages retained in the public package are:

- Random Forest: `0.835 ± 0.027`
- 1D-CNN: `0.853 ± 0.016`
- XGBoost: `0.872 ± 0.010`
- KNN: `0.937 ± 0.006`

## Repository Contents

This repository includes:

- the preserved core implementation;
- the repeated-evaluation wrapper used for manuscript-facing reporting;
- compact checkpoints for inference and figure regeneration;
- exported battery-ID-based split files;
- final averaged figures and tables;
- supplementary outputs for detection delay and ECM confidence analysis;
- documentation for reproducibility, data layout, and manuscript file mapping.


## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

or create the conda environment:

```bash
conda env create -f environment.yml
conda activate mstf-hidet-ae9
```

Set dataset paths before running the pipeline on a new machine:

```powershell
$env:MSTF_VIRTUAL_DATA = "D:\path\to\VRDATA"
$env:MSTF_REAL_DATA = "D:\path\to\dataset_holographic"
```

Then run either the stepwise workflow:

```bash
python "feature and train.py"
python feature_and_train_repeated_avg_eval.py --seeds 42 52 62 72 82
python make_avg_publication_figs.py
python scripts/export_split_files.py
```

or the wrapper:

```bash
python run_all.py
```

Detailed instructions are provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Suggested Entry Points

- Training and original pipeline: [feature and train.py](feature%20and%20train.py)
- Repeated-average evaluation: [feature_and_train_repeated_avg_eval.py](feature_and_train_repeated_avg_eval.py)
- Inference and fault localization: [detection.py](detection.py)
- Averaged manuscript figures: [make_avg_publication_figs.py](make_avg_publication_figs.py)
- Split export: [scripts/export_split_files.py](scripts/export_split_files.py)

## Data and Code Availability

- Data availability note: [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md)
- Code availability note: [docs/CODE_AVAILABILITY.md](docs/CODE_AVAILABILITY.md)
- Dataset layout: [data/DATASET.md](data/DATASET.md)
- Expected folder structure: [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- Checkpoint description: [checkpoints/README.md](checkpoints/README.md)

## Manuscript File Mapping

The correspondence between repository outputs and manuscript-facing figures/tables is summarized in:

- [docs/MANUSCRIPT_FILE_MAP.md](docs/MANUSCRIPT_FILE_MAP.md)
- [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md)

## Notes for Reuse

- The repeated-average outputs are the primary results intended for manuscript reporting.
- The preserved single-run package is kept for transparency and traceability only.
- The averaged t-SNE figure should be interpreted as qualitative support rather than a primary statistical result.
- Internal folder tokens such as `GZ` are retained where they are needed to remain consistent with the original dataset layout and split files.

## Citation and License

Please cite the associated manuscript when using this repository. Citation metadata are provided in [CITATION.cff](CITATION.cff). This repository is released under the MIT license; see [LICENSE](LICENSE).
