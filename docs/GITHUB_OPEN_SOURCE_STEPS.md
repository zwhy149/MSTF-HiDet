# GitHub Open-Source Steps

This file describes the simplest path to publish this repository to GitHub.

## 1. Create an Empty GitHub Repository

Create a new repository on GitHub, for example:

- repository name: `MSTF-HiDet-review-ready`
- visibility: `Public` for open review, or `Private` first and switch to `Public` later

Do not add a new README, `.gitignore`, or license from the GitHub web page, because these files already exist locally.

## 2. Connect the Local Repository to GitHub

Run the following commands in this repository root after replacing `<YOUR_GITHUB_URL>` with the remote URL shown by GitHub:

```bash
git remote add origin <YOUR_GITHUB_URL>
git branch -M main
git push -u origin main
```

## 3. Recommended Repository Settings

- enable Issues only if you want public feedback
- keep Discussions optional
- add a short repository description on GitHub
- pin the repository if it will be cited in the manuscript revision

## 4. Suggested Release Actions

- create a release tag such as `v1.0-review-ready`
- upload the manuscript PDF as a release asset only if journal policy allows it
- keep the repository README focused on code, results, and reproducibility

## 5. Suggested Public Repository Description

```
Review-ready open repository for MSTF-HiDet, including preserved single-run outputs, repeated-average held-out evaluation results, manuscript-aligned figures/tables, and reproducibility documentation.
```

## 6. Suggested Topics

Suggested GitHub topics:

- battery-diagnostics
- fault-detection
- time-series
- domain-adaptation
- machine-learning
- reproducibility

## 7. Before Making the Repository Public

Confirm that:

- raw datasets that cannot be shared are not committed;
- machine-specific absolute paths are documented;
- large cache or checkpoint files are excluded unless intentionally provided;
- the README and documentation point reviewers to the manuscript-facing outputs.
