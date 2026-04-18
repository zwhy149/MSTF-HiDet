# Reviewer Summary

This repository contains the finalized `AE9` code and the manuscript-facing repeated-average results.

## Primary Result

The primary result is the 5-seed repeated Route B average:

- L2 Accuracy: `0.981 ± 0.014`
- L2 Macro-F1: `0.983 ± 0.013`

## Where to Inspect the Main Outputs

- [results/repeated_average/final_report_avg.json](../results/repeated_average/final_report_avg.json)
- [results/repeated_average/repeated_eval_results.csv](../results/repeated_average/repeated_eval_results.csv)
- [results/repeated_average/table2_avg_results.md](../results/repeated_average/table2_avg_results.md)
- [results/repeated_average/table_safety_metrics_avg.md](../results/repeated_average/table_safety_metrics_avg.md)

## What Is Preserved for Transparency

- compact single-run reference summary files under `results/reference_single_run/`
- model and checkpoint bundles under `checkpoints/`

## Important Note

The averaged t-SNE is a qualitative support figure. The primary evidence remains the repeated mean ± standard deviation metrics, the averaged confusion matrix, and the class-wise safety-oriented metrics.
