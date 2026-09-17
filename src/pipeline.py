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
from generation.attack_assigner import build_balanced_type_list
from generation.baseline import (
    BASELINE_INSTRUCTIONS, generate_baseline, apply_baseline_suffixes,
)
from evaluation.judge import run_committee
from evaluation.metrics import compute_metrics

log = get_logger(__name__)


def build_submission(cfg: Config) -> pd.DataFrame:
    set_seeds(cfg.seed)
    test_df = pd.read_csv(cfg.paths.test_csv)
    bnb = build_nf4_config()

    # ---- 1. Vocabulary ----
    if cfg.paths.words_json.exists():
        words = json.loads(Path(cfg.paths.words_json).read_text())
    else:
        words = generate_vocabulary(cfg.generation.vocab, cfg.hf_token)
        save_vocabulary(words, cfg.paths.words_json)

    # ---- 2. Nonsense generator + attack pools ----
    gen = NonsenseGenerator.build(cfg.generation.nonsense, bnb, cfg.hf_token)
    attacks: dict[str, list[str]] = {}
    for key, tmpl in EXPLOITS.items():
        pool = []
        for _ in range(cfg.generation.exploits["examples_per_type"]):
            core = gen.generate()
            if "{}" in tmpl:
                core = tmpl.format(gen.generate())
            else:
                core = tmpl
            pool.append(gen.trim(gen.generate() + core, cfg.generation.essay_max_chars))
        attacks[key] = pool
    cfg.paths.attacks_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.attacks_json.write_text(json.dumps(attacks))

    free_gpu(gen.model, gen.tokenizer)

    # ---- 3. Assign attacks ----
    type_list = build_balanced_type_list(len(test_df))
    key_order = ["909", "099", "990"]  # indices 0,1,2
    counters = {k: 0 for k in key_order}
    essays: list[str | None] = [None] * len(test_df)
    for i, t in enumerate(type_list):
        k = key_order[t]
        pool = attacks[k]
        essays[i] = pool[counters[k] % len(pool)]
        counters[k] += 1

    # ---- 4. Baseline for non-attacked rows ----
    review_model, review_tok = load_model_and_tokenizer(
        cfg.generation.nonsense.model, bnb, cfg.hf_token
    )
    for idx, (instr, params) in enumerate(BASELINE_INSTRUCTIONS):
        for i, topic in enumerate(test_df["topic"]):
            if essays[i] is not None:
                continue
            if i % len(BASELINE_INSTRUCTIONS) != idx:
                continue
            essays[i] = generate_baseline(topic, instr, params, review_model, review_tok)
    free_gpu(review_model, review_tok)

    essays = [e or "" for e in essays]
    essays = apply_baseline_suffixes(essays, list(EXPLOITS.values()))

    submission = pd.DataFrame({"id": test_df["id"], "essay": essays})
    cfg.paths.submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(cfg.paths.submission_csv, index=False)
    log.info("Wrote submission to %s", cfg.paths.submission_csv)
    return submission


def evaluate_submission(cfg: Config) -> dict:
    df = pd.read_csv(cfg.paths.submission_csv)
    essays = df["essay"].astype(str).tolist()

    scores, english = run_committee(essays, cfg.evaluation, cfg.hf_token)
    metrics = compute_metrics(scores, english, essays, cfg.evaluation.similarity_floor)

    cfg.paths.scores_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.scores_json.write_text(json.dumps({
        "scores": scores,
        "english": english,
        "metrics": metrics.__dict__,
        "final_score": metrics.final_score,
    }, indent=2))

    log.info("avg_q=%.4f avg_h=%.4f min_v=%.4f avg_e=%.4f avg_s=%.4f final=%.4f",
             metrics.avg_q, metrics.avg_h, metrics.min_v,
             metrics.avg_e, metrics.avg_s, metrics.final_score)
    return metrics.__dict__ | {"final_score": metrics.final_score}