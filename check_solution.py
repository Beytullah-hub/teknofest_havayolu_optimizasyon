"""Standalone feasibility checker for a completed run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.checker import check_run_output
from src.config import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a TEKNOFEST optimization output run")
    parser.add_argument("--data_dir", default="./data", help="Kept for CLI compatibility; not required by current checks")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--run_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    _, text = check_run_output(run_dir, ModelConfig())
    print(text)


if __name__ == "__main__":
    main()

