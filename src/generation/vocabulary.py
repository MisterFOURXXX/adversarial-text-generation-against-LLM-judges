"""Vocabulary generation with Qwen2.5-3B-Instruct.

Fixes applied:

* The prompt is wrapped with ``apply_chat_template`` so the model's
  instruction-tuning kicks in. The raw prompt used previously produced
  degenerate output (only 10 unique words in the user's run).
* Only the assistant turn is parsed, not the full prompt+response string.
* Per-iteration progress is logged and the loop stops early once the pool
  stops growing (``patience``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config import VocabConfig
from logging_utils import get_logger
from models.loader import load_model_and_tokenizer
from models.quantization import build_nf4_config
from utils.memory import free_gpu

log = get_logger(__name__)


PROMPT = (
    "Generate a list of at least 1000 unique, complex English words suitable "
    "for adversarial attacks. Focus on long (6+ characters), uncommon, or "
    "technical terms (e.g., medical, scientific, literary, or obscure "
    "vocabulary). Include hyphenated technical terms (e.g., "
    "'machine-learning') but avoid repetition, narrative text, or non-word "
    "content. Prioritize diversity and ensure all words are valid English. "
    "Format the output as a space-separated list of words, with no additional "
    "text or formatting."
)


def _is_valid(word: str, min_len: int) -> bool:
    if len(word) <= min_len:
        return False
    if word.isalpha():
        return True
    return "-" in word and all(c.isalpha() or c == "-" for c in word)


def _extract_assistant_turn(decoded: str, tokenizer, prompt_text: str) -> str:
    """Best-effort extraction of the assistant's generated text."""
    # Preferred: split on the assistant marker produced by apply_chat_template.
    try:
        assistant_marker = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if assistant_marker and assistant_marker in decoded:
            return decoded.split(assistant_marker)[-1].strip()
    except Exception:
        pass

    # Fallback: drop the prompt prefix if it appears verbatim.
    if prompt_text and prompt_text in decoded:
        return decoded.split(prompt_text, 1)[-1].strip()

    return decoded.strip()


def generate_vocabulary(cfg: VocabConfig, hf_token: str | None) -> list[str]:
    bnb = build_nf4_config()
    model, tokenizer = load_model_and_tokenizer(
        cfg.model, bnb, hf_token, attn_implementation="eager"
    )

    # Chat-template the prompt so the model behaves as an instruction follower.
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    words: set[str] = set()
    stagnant = 0
    last_size = 0

    for it in range(cfg.n_iterations):
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        body = _extract_assistant_turn(decoded, tokenizer, prompt_text)

        added = 0
        for w in body.split():
            w = w.strip().lower().strip(".,;:()[]{}")
            if _is_valid(w, cfg.min_word_len) and w not in words:
                words.add(w)
                added += 1

        if it % 5 == 0 or it == cfg.n_iterations - 1:
            log.info(
                "Vocabulary iter %d/%d | +%d words | pool=%d",
                it + 1, cfg.n_iterations, added, len(words),
            )

        if len(words) >= cfg.target_size:
            break

        if len(words) == last_size:
            stagnant += 1
            if stagnant >= cfg.patience:
                log.warning(
                    "Vocabulary pool stagnant for %d iterations at %d words; "
                    "stopping early.",
                    stagnant, len(words),
                )
                break
        else:
            stagnant = 0
            last_size = len(words)

    free_gpu(model, tokenizer)

    result = list(words)[: cfg.target_size]
    log.info("Generated %d unique vocabulary words", len(result))
    if len(result) < cfg.target_size:
        log.warning(
            "Vocabulary smaller than target (%d < %d); downstream stages "
            "will tolerate this.",
            len(result), cfg.target_size,
        )
    return result


def save_vocabulary(words: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(words))


__all__ = ["PROMPT", "generate_vocabulary", "save_vocabulary"]