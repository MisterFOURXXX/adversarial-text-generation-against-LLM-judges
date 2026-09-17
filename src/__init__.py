"""Flat-layout source root for the LLM Adversarial Judge Exploitation project.

This file exists for tooling clarity (mypy, IDEs, `python -c "import src"`)
rather than runtime requirements. Importable units are the sibling modules and
packages on this path, e.g.:

    from config import load_config
    from pipeline import build_submission, evaluate_submission
    from utils.seeds import set_seeds
    from models.loader import load_model_and_tokenizer
    from generation.vocabulary import generate_vocabulary
    from evaluation.metrics import compute_metrics

No `advjudge` namespace is used anywhere in this repository.
"""

__all__ = [
    "config",
    "logging_utils",
    "pipeline",
    "utils",
    "models",
    "generation",
    "evaluation",
]