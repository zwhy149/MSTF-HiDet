# Dataset Note

This public repository does not bundle the raw datasets.

## Expected Data Roots

- Virtual data root: `MSTF_VIRTUAL_DATA`
- Real data root: `MSTF_REAL_DATA`

By default, the AE9 source code falls back to the original local path conventions:

- virtual data: `D:\AE2\VRDATA`
- real data: `D:\AE\dataset_holographic`

## Expected Scenario Folders

The public code currently expects the following folders:

- `Normal`
- `充电短路`
- `GZ`

## Expected Per-File Fields

Each Excel file is expected to contain at least:

- a time column
- a voltage column

Optional columns used by parts of the code:

- label column
- temperature column

## Important Scope Note

The current public code directly exposes the three-class diagnosis setting:

- `Normal`
- `Charging Short`
- `Rest-Stage Short`

If the manuscript contains broader experimental scenarios, those should be documented separately as manuscript scope versus public-code scope.
