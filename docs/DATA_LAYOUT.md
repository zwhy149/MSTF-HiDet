# Data Layout

This repository expects external datasets and does not bundle them.

## Recommended Layout

```text
<virtual_data_root>/
├── Normal/
├── 充电短路/
└── GZ/

<real_data_root>/
├── Normal/
├── 充电短路/
└── GZ/
```

## Path Control

Recommended environment variables:

- `MSTF_VIRTUAL_DATA`
- `MSTF_REAL_DATA`
- `MSTF_OUTPUT_DIR`
- `MSTF_PYTHON_EXE`

## Notes

- The current source code still preserves the original local fallback paths from the AE9 working environment.
- For public use, set environment variables before running the scripts.
