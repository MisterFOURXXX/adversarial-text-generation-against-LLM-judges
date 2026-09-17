MODEL_CONFIGS = {
    "gemma": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    "phi":   "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
    "qwen":  "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "llama": "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
             "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

DEFAULT_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def get_model_template(model_name: str) -> str:
    name = model_name.lower()
    for key, tmpl in MODEL_CONFIGS.items():
        if key in name:
            return tmpl
    return DEFAULT_TEMPLATE