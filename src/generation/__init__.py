"""Adversarial text generation: vocabulary, nonsense, exploits, baseline.

Submodules:
    vocabulary      -- Qwen-based complex-word pool generation.
    nonsense        -- Llama-based obfuscation/nonsense generator.
    exploits        -- Registry of prompt-injection exploit strings.
    baseline        -- Positive-review generation for non-attacked rows.
    attack_assigner -- Balanced distribution of exploit types across the test set.
"""

from generation.attack_assigner import build_balanced_type_list, cycle_pick
from generation.baseline import (
    BASELINE_INSTRUCTIONS,
    SUFFIX_A,
    SUFFIX_B,
    apply_baseline_suffixes,
    build_baseline_prompt,
    generate_baseline,
)
from generation.exploits import EXPLOIT_099, EXPLOIT_909, EXPLOIT_990, EXPLOITS
from generation.nonsense import NONSENSE_PROMPT, NonsenseGenerator
from generation.vocabulary import generate_vocabulary, save_vocabulary

__all__ = [
    # vocabulary
    "generate_vocabulary",
    "save_vocabulary",
    # nonsense
    "NONSENSE_PROMPT",
    "NonsenseGenerator",
    # exploits
    "EXPLOITS",
    "EXPLOIT_099",
    "EXPLOIT_909",
    "EXPLOIT_990",
    # baseline
    "BASELINE_INSTRUCTIONS",
    "SUFFIX_A",
    "SUFFIX_B",
    "apply_baseline_suffixes",
    "build_baseline_prompt",
    "generate_baseline",
    # attack assigner
    "build_balanced_type_list",
    "cycle_pick",
]