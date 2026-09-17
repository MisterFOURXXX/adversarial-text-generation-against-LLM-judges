"""End-to-end orchestration for adversarial essay generation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import Config
from logging_utils import get_logger
from utils.seeds import set_seeds
from utils.memory import free_gpu
from models.quantization import build_nf4_config
from models.loader import load_model_and_tokenizer
from generation.vocabulary import generate_vocabulary, save_vocabulary
from generation.nonsense import NonsenseGenerator
from generation.exploits import EXPLOITS, ALL_STABLE_MARKERS
from generation.attack_assigner import build_balanced_type_list, TYPE_TO_EXPLOIT_KEY
from generation.baseline import (
    BASELINE_INSTRUCTIONS,
    generate_baseline,
    apply_baseline_suffixes,
)
from evaluation.judge import run_committee
from evaluation.metrics import compute_metrics

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _assemble_attack_pool(
    gen: NonsenseGenerator,
    tmpl: str,
    examples: int,
    max_chars: int,
) -> list[str]:
    """Build ``examples`` attack strings from one exploit template.

    Matches the reference notebook logic:

        outer = nonsense()
        body  = tmpl.format(nonsense())  if '{}' in tmpl
                tmpl                      otherwise
        essay = trim(outer + body)
    """
    pool: list[str] = []
    needs_inner = "{}" in tmpl
    for _ in range(examples):
        outer = gen.generate()
        body = tmpl.format(gen.generate()) if needs_inner else tmpl
        pool.append(gen.trim(outer + body, max_chars))
    return pool


# --------------------------------------------------------------------------- #
# Submission assembly                                                         #
# --------------------------------------------------------------------------- #

def build_submission(cfg: Config) -> pd.DataFrame:
    set_seeds(cfg.seed)
    test_df = pd.read_csv(cfg.paths.test_csv)
    bnb = build_nf4_config()

    # ---- 1. Vocabulary (cached) ---------------------------------------------
    if cfg.paths.words_json.exists():
        log.info("Vocabulary cache hit: %s", cfg.paths.words_json)
    else:
        words = generate_vocabulary(cfg.generation.vocab, cfg.hf_token)
        save_vocabulary(words, cfg.paths.words_json)

    # ---- 2–3. Nonsense pools + exploit assembly (cached) --------------------
    if cfg.paths.attacks_json.exists():
        log.info("Attack cache hit: %s", cfg.paths.attacks_json)
        attacks = json.loads(Path(cfg.paths.attacks_json).read_text())
    else:
        gen = NonsenseGenerator.build(cfg.generation.nonsense, bnb, cfg.hf_token)
        examples = cfg.generation.exploits["examples_per_type"]
        attacks = {
            key: _assemble_attack_pool(
                gen, tmpl, examples, cfg.generation.essay_max_chars
            )
            for key, tmpl in EXPLOITS.items()
        }
        cfg.paths.attacks_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.paths.attacks_json.write_text(json.dumps(attacks))

        # Drop references so VRAM is reclaimed before next model load
        gen.release()
        del gen
        free_gpu()

    # ---- 4. Balanced attack assignment -------------------------------------
    type_list = build_balanced_type_list(len(test_df))
    counters = {k: 0 for k in attacks}
    essays: list[str | None] = [None] * len(test_df)
    for i, t in enumerate(type_list):
        exploit_key = TYPE_TO_EXPLOIT_KEY[t]
        pool = attacks[exploit_key]
        essays[i] = pool[counters[exploit_key] % len(pool)]
        counters[exploit_key] += 1

    # Snapshot which rows are attacks *before* baseline fills the rest
    is_attack: list[bool] = [e is not None for e in essays]

    # ---- 5. Baseline reviews for non-attacked rows --------------------------
    review_model, review_tok = load_model_and_tokenizer(
        cfg.generation.nonsense.model, bnb, cfg.hf_token
    )
    for idx, (instr, params) in enumerate(BASELINE_INSTRUCTIONS):
        for i, topic in enumerate(test_df["topic"]):
            if essays[i] is not None:
                continue
            if i % len(BASELINE_INSTRUCTIONS) != idx:
                continue
            essays[i] = generate_baseline(
                topic, instr, params, review_model, review_tok
            )

    review_model = None
    review_tok = None
    free_gpu()

    # Coerce None to "" then apply suffixes only to non-attack rows
    essays_str: list[str] = [e or "" for e in essays]
    essays_str = apply_baseline_suffixes(essays_str, is_attack=is_attack)

    submission = pd.DataFrame({"id": test_df["id"], "essay": essays_str})
    cfg.paths.submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(cfg.paths.submission_csv, index=False)
    log.info("Wrote submission to %s", cfg.paths.submission_csv)
    return submission


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def evaluate_submission(cfg: Config) -> dict:
    df = pd.read_csv(cfg.paths.submission_csv)
    essays = df["essay"].astype(str).tolist()

    scores, english = run_committee(essays, cfg.evaluation, cfg.hf_token)
    metrics = compute_metrics(
        scores, english, essays, cfg.evaluation.similarity_floor
    )

    cfg.paths.scores_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.scores_json.write_text(json.dumps({
        "scores": scores,
        "english": english,
        "metrics": metrics.__dict__,
        "final_score": metrics.final_score,
    }, indent=2))

    log.info(
        "avg_q=%.4f avg_h=%.4f min_v=%.4f avg_e=%.4f avg_s=%.4f final=%.4f",
        metrics.avg_q, metrics.avg_h, metrics.min_v,
        metrics.avg_e, metrics.avg_s, metrics.final_score,
    )
    return metrics.__dict__ | {"final_score": metrics.final_score}