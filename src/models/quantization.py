import torch
from transformers import BitsAndBytesConfig


def build_nf4_config() -> BitsAndBytesConfig:
    """Standard 4-bit NF4 quantization config for all pipeline models."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )