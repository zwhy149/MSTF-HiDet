# MSTF-HiDet AE9 Minimum Reproducible Repository

This repository is the AE9-based minimum reproducible package for the MSTF-HiDet manuscript.

It is organized to support:

- the original MSTF-HiDet training and inference code;
- the repeated-average manuscript-facing evaluation over 5 seeds;
- the final averaged tables and figures used for revision;
- the battery-ID-based split export utility;
- reviewer-oriented reproducibility documentation.

## Scope

This public package is centered on the finalized `AE9` code and results.

The public code directly exposes the hierarchical diagnosis of:

- `Normal`
- `Charging Short`
- `Full-SOC Resting Short-circuit`

See [docs/SCENARIO_SCOPE.md](docs/SCENARIO_SCOPE.md) for the boundary between manuscript scope and public-code scope.

## Main Manuscript-Facing Results

The primary result is the 5-seed repeated Route B average:

- MSTF-HiDet L2 Accuracy: `0.981 ± 0.014`
- MSTF-HiDet L2 Macro-F1: `0.983 ± 0.013`

Fixed baseline averages:

- Random Forest: `0.835 ± 0.027`
- 1D-CNN: `0.853 ± 0.016`
- XGBoost: `0.872 ± 0.010`
- KNN: `0.937 ± 0.006`

Main result files:

- [results/repeated_average/final_report_avg.json](results/repeated_average/final_report_avg.json)
- [results/repeated_average/repeated_eval_results.csv](results/repeated_average/repeated_eval_results.csv)
- [results/repeated_average/table2_avg_results.md](results/repeated_average/table2_avg_results.md)
- [results/repeated_average/table_safety_metrics_avg.md](results/repeated_average/table_safety_metrics_avg.md)

## What Is Included

- Original AE9 core code
- Repeated-evaluation wrapper
- Main manuscript-facing averaged figures
- Compact single-run reference reports
- One model bundle and one checkpoint bundle
- Config templates, split export utility, and reproducibility docs

## What Is Not Included

- Raw virtual dataset files
- Raw real battery dataset files
- Python virtual environment
- Cache files such as `*_feature_cache.pkl`
- Every per-seed generated image
- Historical duplicate figure directories

This is intentional: the goal is a lean, review-friendly, minimum reproducible repository.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── feature and train.py
├── detection.py
├── feature_and_train_repeated_avg_eval.py
├── run_single_seed_avg_eval.py
├── make_avg_publication_figs.py
├── run_all.py
├── configs/
├── data/
├── docs/
├── scripts/
├── splits/
├── checkpoints/
└── results/
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

This repository does not bundle raw datasets.

Recommended environment variables:

```bash
set MSTF_VIRTUAL_DATA=D:\path\to\VRDATA
set MSTF_REAL_DATA=D:\path\to\dataset_holographic
```

See:

- [data/DATASET.md](data/DATASET.md)
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- [configs/data_paths.example.json](configs/data_paths.example.json)

## Reproducing the Results

```bash
python "feature and train.py"
python feature_and_train_repeated_avg_eval.py --seeds 42 52 62 72 82
python make_avg_publication_figs.py
python scripts/export_split_files.py
```

Or run the wrapper:

```bash
python run_all.py
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Figures in This Public Package

For GitHub, it is not necessary to upload every generated image.

This package keeps only the main manuscript-facing figures for convenience:

- averaged training curve
- averaged confusion matrix
- averaged t-SNE
- averaged SOTA comparison
- averaged ablation
- averaged FPR
- one combined 2x2 figure

All other repetitive or historical figures can be regenerated from code when data are available.

## Important Notes

- Use the repeated-average result for manuscript-facing reporting.
- The single-run reference result is kept only for transparency.
- The averaged t-SNE is a qualitative support figure.
- The current public code should be described consistently with its actual scope and split logic.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
