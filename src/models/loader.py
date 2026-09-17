"""Hugging Face model + tokenizer loader with 4-bit NF4 quantization.

Fixes applied:

* ``attn_implementation`` is now configurable (default ``"eager"``) so
  flash-attn version mismatches do not abort the run.
* Qwen3 ships a ``generation_config.json`` whose sampling defaults
  (``temperature``, ``top_p``, ``top_k``) trigger noisy warnings when
  overridden by ``do_sample=False``. We clear those fields after load.
* A clear log line reports the quantization and attention backend in use.
"""

from __future__ import annotations

from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from logging_utils import get_logger

log = get_logger(__name__)


def _clear_sampling_defaults(model) -> None:
    """Remove sampling-related defaults inherited from generation_config.

    This silences the ``The following generation flags are not valid``
    warning that HF emits when greedy decoding is requested but the model
    config still carries temperature/top_p/top_k values.
    """
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, None)
            except Exception:
                # Some configs are frozen; ignore.
                pass


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = "eager",
):
    """Load a causal LM + tokenizer with the given quantization config."""
    log.info(
        "Loading model=%s (attn=%s)", model_name, attn_implementation or "default"
    )

    tok_kwargs: dict = {"trust_remote_code": trust_remote_code}
    mdl_kwargs: dict = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        mdl_kwargs["attn_implementation"] = attn_implementation
    if hf_token:
        tok_kwargs["token"] = hf_token
        mdl_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)
    model.eval()

    _clear_sampling_defaults(model)
    return model, tokenizer


__all__ = ["load_model_and_tokenizer"]