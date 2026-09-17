"""Hugging Face model + tokenizer loading with 4-bit NF4 quantization."""

from __future__ import annotations

import os
import warnings
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from logging_utils import get_logger

log = get_logger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",
):
    """Load a quantized causal LM and its tokenizer.

    Parameters
    ----------
    model_name
        Full Hugging Face repository id, optionally ``owner/name@revision``.
    bnb_config
        A pre-built ``BitsAndBytesConfig`` (typically NF4 4-bit).
    hf_token
        Optional Hugging Face token for gated repositories.
    trust_remote_code
        Allow custom modelling code from the repository.
    attn_implementation
        Attention backend. ``"eager"`` is the safest cross-model choice and
        silences Phi's "flash-attention not found" warnings. Set to
        ``"flash_attention_2"`` only if the package is installed and the
        hardware supports it (Ampere+).

    Returns
    -------
    (model, tokenizer)
    """
    log.info("Loading model=%s (attn=%s)", model_name, attn_implementation)

    tok_kwargs = {"trust_remote_code": trust_remote_code}
    mdl_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    if hf_token:
        tok_kwargs["token"] = hf_token
        mdl_kwargs["token"] = hf_token

    # Silence the "temperature/top_p/top_k not valid" warning that fires when
    # a repo's generation_config.json ships stochastic defaults but the caller
    # uses greedy decoding. Judges pass explicit None values, but the warning
    # is emitted at load time by some tokenizers, so we suppress it here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*generation flags are not valid.*",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)

    model.eval()
    return model, tokenizer


# Ensure HF doesn't print progress bars twice when nested loggers fire.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "FALSE")