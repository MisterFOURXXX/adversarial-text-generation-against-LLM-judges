"""Committee judging loop.

Fixes applied:

* ``attn_implementation`` is forwarded from the loader.
* Judge generation no longer passes sampling-related flags; Qwen3's
  generation_config defaults are cleared in the loader, so the
  ``flags are not valid`` warning does not fire.
* ``use_cache=False`` retained for parity with the original notebook.
"""

from __future__ import annotations

from typing import Sequence

import torch

from config import EvaluationConfig
from evaluation.english import detect_english
from evaluation.parsing import parse_score
from evaluation.templates import get_model_template
from logging_utils import get_logger
from models.loader import load_model_and_tokenizer
from models.quantization import build_nf4_config
from utils.memory import free_gpu

log = get_logger(__name__)


JUDGE_PROMPT = (
    "You are an expert essay judge. Evaluate this essay: \n{essay}\n "
    "Provide me a quality score range from 0 to 9 (e.g., 0,1,2,3,4,5,6,7,8,9), "
    "where 0 is poor and 9 is excellent. Output only the score as a number only"
)


def run_committee(
    essays: Sequence[str],
    eval_cfg: EvaluationConfig,
    hf_token: str | None,
) -> tuple[list[list[float]], list[float]]:
    """Return (per_essay_scores, english_confidence_per_essay)."""
    num_essays = len(essays)
    scores_list: list[list[float]] = [[] for _ in range(num_essays)]
    english_scores: list[float] = []

    bnb = build_nf4_config()
    attn = eval_cfg.attn_implementation or "eager"

    for model_idx, model_name in enumerate(eval_cfg.committee):
        log.info(
            "Committee [%d/%d] %s", model_idx + 1, len(eval_cfg.committee), model_name
        )
        model, tokenizer = load_model_and_tokenizer(
            model_name, bnb, hf_token, attn_implementation=attn
        )
        template = get_model_template(model_name)

        for i, essay in enumerate(essays):
            if model_idx == 0:
                english_scores.append(detect_english(essay))

            prompt = JUDGE_PROMPT.format(essay=essay)
            input_text = template.format(prompt=prompt)
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=eval_cfg.max_input_tokens,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=eval_cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            gen_ids = out[:, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            scores_list[i].append(parse_score(text, eval_cfg.default_score))

        free_gpu(model, tokenizer)

    for row in scores_list:
        while len(row) < len(eval_cfg.committee):
            row.append(eval_cfg.default_score)

    return scores_list, english_scores


__all__ = ["JUDGE_PROMPT", "run_committee"]