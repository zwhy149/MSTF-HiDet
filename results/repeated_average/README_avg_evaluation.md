# AE9 Repeated-Average Evaluation

- Source logic: copied from `D:\AE7 - Copy` without editing the original folder.
- Repeated seeds: 42, 52, 62, 72, 82
- Fixed baseline set for the new SOTA figure: Random Forest, 1D-CNN, XGBoost, KNN.
- The averaged SOTA figure is produced by the wrapper script and does not overwrite the original single-run figure.
- The new pooled t-SNE figure aggregates embeddings across repeated test runs because a literal arithmetic average of t-SNE coordinates is not statistically meaningful.
