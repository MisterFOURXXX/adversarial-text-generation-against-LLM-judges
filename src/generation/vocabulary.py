"""Stage 1 — vocabulary generation using Qwen2.5-3B-Instruct."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config import VocabConfig
from models.loader import load_model_and_tokenizer
from models.quantization import build_nf4_config
from logging_utils import get_logger
from utils.memory import free_gpu

log = get_logger(__name__)

PROMPT = (
    "Generate a list of at least 1000 unique, complex English words suitable for "
    "adversarial attacks. Focus on long (6+ characters), uncommon, or technical terms. "
    "Include hyphenated technical terms but avoid repetition. Format the output as a "
    "space-separated list of words, with no additional text or formatting."
)


def _is_valid(word: str, min_len: int) -> bool:
    if len(word) <= min_len:
        return False
    if word.isalpha():
        return True
    return "-" in word and all(c.isalpha() or c == "-" for c in word)


def generate_vocabulary(cfg: VocabConfig, hf_token: str | None) -> list[str]:
    """Sample a diverse, long-word lexicon from a small instruction model.

    Caches are *not* handled here; `pipeline.build_submission` decides whether
    to call this or to read from disk.
    """
    bnb = build_nf4_config()
    model, tokenizer = load_model_and_tokenizer(cfg.model, bnb, hf_token)

    words: set[str] = set()
    try:
        for _ in range(cfg.n_iterations):
            inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            for w in text.split():
                w = w.strip().lower()
                if _is_valid(w, cfg.min_word_len):
                    words.add(w)
            if len(words) >= cfg.target_size:
                break
    finally:
        # Release model references *before* clearing CUDA cache.
        del model, tokenizer
        free_gpu()

    result = list(words)[: cfg.target_size]
    log.info("Generated %d unique vocabulary words", len(result))
    return result


def save_vocabulary(words: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(words))