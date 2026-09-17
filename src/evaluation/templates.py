"""Per-model chat templates for the judge prompt."""

from __future__ import annotations

# Keys are matched case-insensitively against the HF model id.
# - gemma:  Gemma-2 uses <start_of_turn> / <end_of_turn>
# - phi:    Phi-3.5 uses <|user|> / <|end|> / <|assistant|>
# - qwen:   Qwen2.5 / Qwen3 both use ChatML (<|im_start|> / <|im_end|>)
# - llama:  Llama-3.1 uses <|start_header_id|> / <|eot_id|>
MODEL_CONFIGS: dict[str, str] = {
    "gemma": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    "phi":   "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
    "qwen":  "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "llama": (
        "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}

DEFAULT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def get_model_template(model_name: str) -> str:
    name = model_name.lower()
    for key, tmpl in MODEL_CONFIGS.items():
        if key in name:
            return tmpl
    return DEFAULT_TEMPLATE