from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


@dataclass
class Paths:
    test_csv: Path
    submission_csv: Path
    words_json: Path
    attacks_json: Path
    scores_json: Path


@dataclass
class VocabConfig:
    model: str
    n_iterations: int
    max_new_tokens: int
    temperature: float
    top_p: float
    target_size: int
    min_word_len: int


@dataclass
class NonsenseConfig:
    model: str
    max_new_tokens: int
    temperature: float
    top_p: float
    n_words: int


@dataclass
class GenerationConfig:
    vocab: VocabConfig
    nonsense: NonsenseConfig
    exploits: dict
    baseline: dict
    essay_max_chars: int = 900


@dataclass
class EvaluationConfig:
    committee: list[str]
    max_input_tokens: int = 4096
    max_new_tokens: int = 10
    default_score: float = 4.5
    similarity_floor: float = 0.2


@dataclass
class Config:
    seed: int
    paths: Paths
    generation: GenerationConfig
    evaluation: EvaluationConfig
    hf_token: str | None = field(default=None, repr=False)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    paths = Paths(**{k: Path(v) for k, v in raw["paths"].items()})
    gen = GenerationConfig(
        vocab=VocabConfig(**raw["generation"]["vocab"]),
        nonsense=NonsenseConfig(**raw["generation"]["nonsense"]),
        exploits=raw["generation"]["exploits"],
        baseline=raw["generation"]["baseline"],
        essay_max_chars=raw["generation"].get("essay_max_chars", 900),
    )
    ev = EvaluationConfig(**raw["evaluation"])
    hf_token = os.getenv("HF_TOKEN")
    return Config(seed=raw["seed"], paths=paths, generation=gen,
                  evaluation=ev, hf_token=hf_token)