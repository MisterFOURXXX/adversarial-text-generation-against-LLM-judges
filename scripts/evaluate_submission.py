"""CLI: evaluate a submission CSV against the local committee."""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import load_config            # noqa: E402
from pipeline import evaluate_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = ap.parse_args()
    evaluate_submission(load_config(args.config))


if __name__ == "__main__":
    main()