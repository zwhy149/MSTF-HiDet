# Upload Manifest

## Must Include

- core training and inference scripts
- repeated-average evaluation wrapper
- main averaged summary CSV/JSON/MD files
- key manuscript-facing averaged figures
- checkpoint bundle
- docs and config templates
- split export utility and split CSV files

## Recommended Include

- compact single-run reference summary files
- supplementary ECM and detection-delay outputs

## Exclude

- `.venv/`
- `__pycache__/`
- logs
- `*_feature_cache.pkl`
- raw data folders
- every per-seed generated image
- historical duplicate figure directories

## Rationale

This package is intended to be a minimum reproducible repository rather than a full archival dump of all intermediate outputs.
