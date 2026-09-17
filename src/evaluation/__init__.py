"""Committee judging and competition-metric computation.

Submodules:
    templates -- Per-model chat templates for the judge prompts.
    parsing   -- Regex-based numeric score extraction.
    english   -- Lingua-based English confidence + heuristic fallback.
    judge     -- Committee evaluation loop.
    metrics   -- Competition metrics (avg_q, avg_h, min_v, avg_e, avg_s).
"""

from evaluation.english import detect_english
from evaluation.judge import JUDGE_PROMPT, run_committee
from evaluation.metrics import CompetitionMetrics, compute_metrics
from evaluation.parsing import parse_score
from evaluation.templates import DEFAULT_TEMPLATE, MODEL_CONFIGS, get_model_template

__all__ = [
    # templates
    "MODEL_CONFIGS",
    "DEFAULT_TEMPLATE",
    "get_model_template",
    # parsing
    "parse_score",
    # english
    "detect_english",
    # judge
    "JUDGE_PROMPT",
    "run_committee",
    # metrics
    "CompetitionMetrics",
    "compute_metrics",
]