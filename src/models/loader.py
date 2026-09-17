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
    return model, tokenizer