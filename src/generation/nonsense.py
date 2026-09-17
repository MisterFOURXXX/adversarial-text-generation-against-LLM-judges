"""Nonsense payload generation with Llama-3.1-8B-Instruct.

Fixes applied:

* ``attn_implementation`` is forwarded from the loader so flash-attn is
  not silently required.
* ``build`` accepts the attention backend.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

from config import NonsenseConfig
from logging_utils import get_logger
from models.loader import load_model_and_tokenizer

log = get_logger(__name__)


NONSENSE_PROMPT = (
    "Generate a sequence of 10000 complex, uncommon, or technical English "
    "words. Ensure each word is at least 6 characters long, and format the "
    "output as a space-separated list with no additional text."
)


class NonsenseGenerator:
    """Wraps a causal LM for repeated nonsense-text generation."""

    def __init__(self, model, tokenizer, cfg: NonsenseConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._assistant_marker = self._build_assistant_marker()

    def _build_assistant_marker(self) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return ""

    @classmethod
    def build(
        cls,
        cfg: NonsenseConfig,
        bnb_config,
        hf_token: Optional[str] = None,
        attn_implementation: str = "eager",
    ) -> "NonsenseGenerator":
        model, tokenizer = load_model_and_tokenizer(
            cfg.model, bnb_config, hf_token, attn_implementation=attn_implementation
        )
        return cls(model, tokenizer, cfg)

    def generate(self) -> str:
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
        body = (
            decoded.split(self._assistant_marker)[-1].strip()
            if self._assistant_marker and self._assistant_marker in decoded
            else decoded.strip()
        )

        words = [w for w in body.split() if len(w) > 5 and w.isalpha()]
        words = words[: self.cfg.n_words]

        if len(words) < self.cfg.n_words and words:
            words += random.choices(
                words, k=self.cfg.n_words - len(words)
            )
        return " ".join(words)

    def trim(self, text: str, length: int = 900) -> str:
        text = text[-length:]
        parts = text.split(" ", 1)
        return parts[1] if len(parts) > 1 else text


__all__ = ["NONSENSE_PROMPT", "NonsenseGenerator"]