"""End-to-end pipeline: build a submission, evaluate it, compute metrics."""

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
from generation.exploits import EXPLOITS
from generation.attack_assigner import TYPE_TO_EXPLOIT_KEY, build_balanced_type_list
from generation.baseline import (
    BASELINE_INSTRUCTIONS,
    generate_baseline,
    apply_baseline_suffixes,
)
from evaluation.judge import run_committee
from evaluation.metrics import compute_metrics

log = get_logger(__name__)


def _load_or_generate_vocabulary(cfg: Config) -> list[str]:
    if cfg.paths.words_json.exists():
        log.info("Reusing vocabulary cache at %s", cfg.paths.words_json)
        return json.loads(Path(cfg.paths.words_json).read_text())
    words = generate_vocabulary(cfg.generation.vocab, cfg.hf_token)
    save_vocabulary(words, cfg.paths.words_json)
    return words


def _build_attack_pools(cfg: Config, bnb) -> dict[str, list[str]]:
    """Generate 5 adversarial essays per exploit type."""
    gen = NonsenseGenerator.build(cfg.generation.nonsense, bnb, cfg.hf_token)
    attacks: dict[str, list[str]] = {}

    try:
        n_per_type = cfg.generation.exploits["examples_per_type"]
        for key, tmpl in EXPLOITS.items():
            pool: list[str] = []
            for _ in range(n_per_type):
                body = gen.generate()
                if "{}" in tmpl:
                    exploit_text = tmpl.format(gen.generate())
                else:
                    exploit_text = tmpl
                pool.append(
                    gen.trim(body + exploit_text, cfg.generation.essay_max_chars)
                )
            attacks[key] = pool
    finally:
        gen.close()

    cfg.paths.attacks_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.attacks_json.write_text(json.dumps(attacks))
    return attacks


def _assign_attacks(
    n_rows: int,
    attacks: dict[str, list[str]],
    seed: int,
) -> list[str | None]:
    """Return a list of length ``n_rows`` with attacks or ``None`` placeholders."""
    import random

    rng = random.Random(seed)
    type_list = build_balanced_type_list(n_rows, rng=rng)

    counters: dict[str, int] = {k: 0 for k in attacks}
    essays: list[str | None] = [None] * n_rows
    for i, t in enumerate(type_list):
        key = TYPE_TO_EXPLOIT_KEY[t]
        pool = attacks[key]
        essays[i] = pool[counters[key] % len(pool)]
        counters[key] += 1
    return essays


def _fill_baselines(
    essays: list[str | None],
    topics: list[str],
    cfg: Config,
    bnb,
) -> None:
    """In-place fill of non-attacked rows using Llama-3.1-8B-Instruct."""
    model, tokenizer = load_model_and_tokenizer(
        cfg.generation.nonsense.model, bnb, cfg.hf_token
    )
    try:
        n_templates = len(BASELINE_INSTRUCTIONS)
        for i, topic in enumerate(topics):
            if essays[i] is not None:
                continue
            instr, params = BASELINE_INSTRUCTIONS[i % n_templates]
            essays[i] = generate_baseline(topic, instr, params, model, tokenizer)
    finally:
        del model, tokenizer
        free_gpu()


def build_submission(cfg: Config) -> pd.DataFrame:
    """Run stages 1–5 and write ``submission.csv``."""
    set_seeds(cfg.seed)
    test_df = pd.read_csv(cfg.paths.test_csv)
    bnb = build_nf4_config()

    # ---- Stage 1: vocabulary ----
    _load_or_generate_vocabulary(cfg)

    # ---- Stages 2–3: nonsense + exploit assembly ----
    if cfg.paths.attacks_json.exists():
        log.info("Reusing attacks cache at %s", cfg.paths.attacks_json)
        attacks = json.loads(cfg.paths.attacks_json.read_text())
    else:
        attacks = _build_attack_pools(cfg, bnb)

    # ---- Stage 4: balanced assignment ----
    essays = _assign_attacks(len(test_df), attacks, cfg.seed)

    # ---- Stage 5: baseline fill + suffix pass ----
    _fill_baselines(essays, test_df["topic"].tolist(), cfg, bnb)
    essays_str = [e or "" for e in essays]
    essays_str = apply_baseline_suffixes(essays_str, list(EXPLOITS.values()))

    submission = pd.DataFrame({"id": test_df["id"], "essay": essays_str})
    cfg.paths.submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(cfg.paths.submission_csv, index=False)
    log.info("Wrote submission to %s", cfg.paths.submission_csv)
    return submission


def evaluate_submission(cfg: Config) -> dict:
    """Run stages 6–7 and write ``scores.json``."""
    df = pd.read_csv(cfg.paths.submission_csv)
    essays = df["essay"].astype(str).tolist()

    scores, english = run_committee(essays, cfg.evaluation, cfg.hf_token)
    metrics = compute_metrics(
        scores, english, essays, cfg.evaluation.similarity_floor
    )

    cfg.paths.scores_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.scores_json.write_text(
        json.dumps(
            {
                "scores": scores,
                "english": english,
                "metrics": metrics.__dict__,
                "final_score": metrics.final_score,
            },
            indent=2,
        )
    )

    log.info(
        "avg_q=%.4f avg_h=%.4f min_v=%.4f avg_e=%.4f avg_s=%.4f final=%.4f",
        metrics.avg_q,
        metrics.avg_h,
        metrics.min_v,
        metrics.avg_e,
        metrics.avg_s,
        metrics.final_score,
    )
    return {**metrics.__dict__, "final_score": metrics.final_score}