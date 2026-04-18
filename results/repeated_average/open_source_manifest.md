# AE9 Open-Source Manifest

## New scripts

- [run_single_seed_avg_eval.py](/D:/AE9/run_single_seed_avg_eval.py)
- [feature_and_train_repeated_avg_eval.py](/D:/AE9/feature_and_train_repeated_avg_eval.py)

## Main averaged outputs

- [repeated_eval_results.csv](/D:/AE9/avg_evaluation_results/repeated_eval_results.csv)
- [repeated_eval_summary.csv](/D:/AE9/avg_evaluation_results/repeated_eval_summary.csv)
- [final_report_avg.json](/D:/AE9/avg_evaluation_results/final_report_avg.json)
- [table2_avg_results.csv](/D:/AE9/avg_evaluation_results/table2_avg_results.csv)
- [table2_avg_results.md](/D:/AE9/avg_evaluation_results/table2_avg_results.md)
- [section4_4_avg_result_text.md](/D:/AE9/avg_evaluation_results/section4_4_avg_result_text.md)
- [absolute_claim_revisions.md](/D:/AE9/avg_evaluation_results/absolute_claim_revisions.md)

## Main averaged figures

- [Fig2_confusion_matrices_avg.png](/D:/AE9/avg_evaluation_results/Fig2_confusion_matrices_avg.png)
- [Fig3_tsne_repeated_pooled.png](/D:/AE9/avg_evaluation_results/Fig3_tsne_repeated_pooled.png)
- [Fig4_sota_fixed_baselines_avg.png](/D:/AE9/avg_evaluation_results/Fig4_sota_fixed_baselines_avg.png)

## Publication-style averaged figures

- [Fig2_confusion_matrices_avg_blue.png](/D:/AE9/avg_evaluation_results/Fig2_confusion_matrices_avg_blue.png)
- [Fig3_tsne_avg_style.png](/D:/AE9/avg_evaluation_results/Fig3_tsne_avg_style.png)
- [Fig_FPR_avg_curves.png](/D:/AE9/avg_evaluation_results/Fig_FPR_avg_curves.png)
- [Fig7_ablation_avg.png](/D:/AE9/avg_evaluation_results/Fig7_ablation_avg.png)
- [Fig_combined_2x2_avg_publication.png](/D:/AE9/avg_evaluation_results/Fig_combined_2x2_avg_publication.png)
- [ablation_avg_summary.csv](/D:/AE9/avg_evaluation_results/ablation_avg_summary.csv)
- [fpr_avg_summary.csv](/D:/AE9/avg_evaluation_results/fpr_avg_summary.csv)
- [tsne_q1_reviewer_note.md](/D:/AE9/avg_evaluation_results/tsne_q1_reviewer_note.md)

## Per-run outputs

- [runs](/D:/AE9/avg_evaluation_results/runs)

Each seed directory under `runs/` contains the original single-run outputs produced by the copied AE7 pipeline, plus `fixed_baseline_results.json` for the fixed-baseline averaging layer.

## Reporting guidance

- Keep `D:\AE9\detection_results\*` as the preserved copied single-run reference assets.
- Use `D:\AE9\avg_evaluation_results\*` for manuscript-facing repeated-average reporting.
