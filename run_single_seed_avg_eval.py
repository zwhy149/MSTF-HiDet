import argparse
import importlib.util
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path


def load_original_module(module_path: Path):
    loader = SourceFileLoader("mstf_hidet_original_main", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Run the original AE7-copy pipeline once with a specific seed and output directory."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--module-path",
        default=str(Path(__file__).resolve().parent / "feature and train.py"),
    )
    args = parser.parse_args()

    module_path = Path(args.module_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mod = load_original_module(module_path)
    mod.CFG["SEED"] = int(args.seed)
    mod.CFG["OUTPUT_DIR"] = str(output_dir)
    mod.set_seed(int(args.seed))
    os.makedirs(mod.CFG["OUTPUT_DIR"], exist_ok=True)

    print("=" * 70)
    print(f"Single-seed AE7-copy run")
    print(f"Seed: {args.seed}")
    print(f"Module: {module_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    mod.main()


if __name__ == "__main__":
    main()
