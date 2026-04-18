# Data Layout

This repository expects external datasets and does not bundle them.

## Recommended Layout

```text
<virtual_data_root>/
├── Normal/
├── <charging_short_folder>/
└── <full_soc_resting_short_circuit_folder>

<real_data_root>/
├── Normal/
├── <charging_short_folder>/
└── <full_soc_resting_short_circuit_folder>
```

## Path Control

Recommended environment variables:

- `MSTF_VIRTUAL_DATA`
- `MSTF_REAL_DATA`
- `MSTF_OUTPUT_DIR`
- `MSTF_PYTHON_EXE`

## Notes

- Public-facing documentation uses the English scenario name `Charging Short`.
- The source code still preserves the original external folder tokens required by the working dataset layout.
- The current source code still preserves the original local fallback paths from the AE9 working environment.
- For public use, set environment variables before running the scripts.
