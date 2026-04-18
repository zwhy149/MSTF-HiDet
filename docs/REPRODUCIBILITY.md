# Reproducibility

This package contains both the preserved single-run reference and the AE9 manuscript-facing repeated-average outputs.

## 1. Train the Preserved Original Pipeline

```bash
python "feature and train.py"
```

This writes a fresh `detection_results/` directory in the repository root.

## 2. Run the Repeated-Average Evaluation

```bash
python feature_and_train_repeated_avg_eval.py --seeds 42 52 62 72 82
```

This writes `avg_evaluation_results/`.

## 3. Regenerate the Main Averaged Figures

```bash
python make_avg_publication_figs.py
```

## 4. Export Split Files

```bash
python scripts/export_split_files.py
```

## 5. Optional Supplementary Figure Scripts

```bash
python scripts/gen_detection_delay_figs.py
python scripts/gen_combined_fig.py
python scripts/gen_combined_2x2.py
python scripts/run_ecm_analysis.py
```

## 6. One-Command Wrapper

```bash
python run_all.py
```

## Notes

- Raw datasets are not included in this repository.
- Set `MSTF_VIRTUAL_DATA` and `MSTF_REAL_DATA` before running the pipeline on a new machine.
- The repeated-average result is the manuscript-facing primary result in this package.
