The manuscript-facing averaged SOTA figure in `AE9` uses a fixed baseline set:

- Random Forest
- 1D-CNN
- XGBoost
- KNN

This fixed selection is implemented only in the new wrapper script
[`feature_and_train_repeated_avg_eval.py`](</D:/AE9/feature_and_train_repeated_avg_eval.py>).
The copied original script [`feature and train.py`](</D:/AE9/feature and train.py>) is left unchanged.

Accordingly:

- the original single-run figure in `detection_results/Fig4_sota.*` remains reproducible from the copied source;
- the new averaged SOTA figure in `avg_evaluation_results/Fig4_sota_fixed_baselines_avg.*` does not depend on the original threshold branch;
- no hidden branch was added for manuscript reporting.
