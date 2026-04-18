# Scenario Scope

This repository is built from the finalized `AE9` code and result directories.

## Directly Exposed in the Public Code

The diagnostic label space directly implemented in the public code is:

- `Normal`
- `Charging Short`
- `Full-SOC Resting Short-circuit`

These are mapped from the scenario folders used by the code:

- `Normal`
- charging-short folder in the original dataset layout
- full-SOC resting short-circuit folder in the original dataset layout

## Important Manuscript Alignment Note

If the manuscript discusses broader ESC scenarios, including:

- continuous-cycling ESC
- 2P parallel-module ESC

then those scenarios should not be described as fully reproduced by this public minimum package unless corresponding public scripts, labels, and data interfaces are also included.

## Recommended Reporting Language

Use the current public repository as the minimum reproducible implementation of the core MSTF-HiDet pipeline and the manuscript-facing repeated-average reporting workflow.
