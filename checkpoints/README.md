# Checkpoints

This directory contains compact checkpoint assets needed for inference and figure regeneration.

## Files

- `mstf_hidet_bundle.pkl`
  - lightweight inference bundle
  - used by `detection.py` and detection-delay scripts

- `checkpoint_results.pkl`
  - compact result checkpoint used by some visualization scripts
  - contains embeddings, predictions, probabilities, and training history

## Notes

- These files come from the finalized AE9 result package.
- Large cache files are intentionally excluded from the public minimum package.
