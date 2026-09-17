"""Nonsense-text generation with Llama-3.1-8B-Instruct."""

from __future__ import annotations

import random

import torch

from config import NonsenseConfig
from models.loader import load_model_and_tokenizer
from logging_utils import get_logger

log = get_logger(__name__)

NONSENSE_PROMPT = (
    "Generate a sequence of 10000 complex, uncommon, or technical English words. "
    "Ensure each word is at least 6 characters long, and format the output as a "
    "space-separated list with no additional text."
)


class NonsenseGenerator:
    """Wraps a Llama model for repeated nonsense-text generation."""

    def __init__(self, model, tokenizer, cfg: NonsenseConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    @classmethod
    def build(
        cls,
        cfg: NonsenseConfig,
        bnb_config,
        hf_token: str | None,
    ) -> "NonsenseGenerator":
        model, tokenizer = load_model_and_tokenizer(cfg.model, bnb_config, hf_token)
        return cls(model, tokenizer, cfg)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(self) -> str:
        """Return exactly ``cfg.n_words`` alphabetic tokens (len >= 6)."""
        messages = [{"role": "user", "content": NONSENSE_PROMPT}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        assistant_marker = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
        )
        body = decoded.split(assistant_marker)[-1].strip()

        words = [w for w in body.split() if len(w) > 5 and w.isalpha()]
        words = words[: self.cfg.n_words]

        # Guard: empty pool would crash random.choices
        if not words:
            log.warning("Nonsense generator returned no valid words; returning empty payload")
            return ""

        # Backfill shortfalls so every payload has the same token count
        if len(words) < self.cfg.n_words:
            shortfall = self.cfg.n_words - len(words)
            words += random.choices(words, k=shortfall)

        return " ".join(words)

    def trim(self, text: str, length: int = 900) -> str:
        """Return the *last* ``length`` characters, dropping a partial leading word.

        If ``text`` is already shorter than ``length``, it is returned
        unchanged — previously the leading word was dropped even when no
        truncation was needed.
        """
        if len(text) <= length:
            return text
        text = text[-length:]
        parts = text.split(" ", 1)
        return parts[1] if len(parts) > 1 else text

    # ------------------------------------------------------------------ #
    # Reference release                                                   #
    # ------------------------------------------------------------------ #

    def release(self) -> None:
        """Drop internal references so ``free_gpu()`` can reclaim VRAM."""
        self.model = None
        self.tokenizer = None