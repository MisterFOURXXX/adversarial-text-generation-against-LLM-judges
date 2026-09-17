"""Hugging Face model / tokenizer loading with quantization and warning suppression."""

from __future__ import annotations

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from logging_utils import get_logger

log = get_logger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """Load a quantized causal LM and its tokenizer.

    Sampling defaults (temperature, top_p, top_k) are cleared on the model's
    ``generation_config`` so that subsequent greedy calls (``do_sample=False``)
    do not trigger HuggingFace's "generation flags are not valid" warning.
    """
    log.info("Loading model=%s", model_name)

    tok_kwargs = {"trust_remote_code": trust_remote_code}
    mdl_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if hf_token:
        tok_kwargs["token"] = hf_token
        mdl_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)
    model.eval()

    # --- Warning suppression -------------------------------------------------
    # Many instruct models ship a generation_config.json that sets
    # temperature/top_p/top_k defaults. Those flags are ignored when
    # do_sample=False, but HuggingFace still emits a warning each call.
    # Nulling them out on the config is the documented remedy.
    gc_cfg = getattr(model, "generation_config", None)
    if gc_cfg is not None:
        for flag in ("temperature", "top_p", "top_k"):
            if getattr(gc_cfg, flag, None) is not None:
                setattr(gc_cfg, flag, None)
    # ------------------------------------------------------------------------

    return model, tokenizer