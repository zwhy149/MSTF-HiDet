from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(args: list[str]) -> None:
    print(f"[RUN] {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main MSTF-HiDet AE9 reproduction steps.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-repeated", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        run_step(["feature and train.py"])
    if not args.skip_repeated:
        run_step(["feature_and_train_repeated_avg_eval.py", "--seeds", "42", "52", "62", "72", "82"])
    if not args.skip_figures:
        run_step(["make_avg_publication_figs.py"])
    if not args.skip_splits:
        run_step(["scripts/export_split_files.py"])


if __name__ == "__main__":
    main()
