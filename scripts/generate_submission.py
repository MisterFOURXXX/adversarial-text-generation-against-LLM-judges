"""CLI: generate a submission CSV.

Adds ``src/`` to ``sys.path`` so the script runs from a clean clone without
``pip install -e .``.
"""

import argparse
import sys
from pathlib import Path

# Resolve project root relative to this file, then expose src/ for imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import load_config          # noqa: E402
from pipeline import build_submission   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = ap.parse_args()
    build_submission(load_config(args.config))


if __name__ == "__main__":
    main()