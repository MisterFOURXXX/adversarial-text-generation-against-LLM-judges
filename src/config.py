"""Typed configuration objects and YAML loader.

All paths are validated at load time. Output paths that are unwritable
(e.g. they live under the read-only /kaggle/input tree) are transparently
redirected to a writable fallback directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Writability helpers
# ---------------------------------------------------------------------------

def _default_output_dir() -> Path:
    """Return a writable base directory for pipeline artefacts.

    Preference order:
        1. $ADVJUDGE_OUTPUT_DIR if set.
        2. /kaggle/working/outputs if running on Kaggle.
        3. ./outputs relative to CWD.
    """
    env = os.getenv("ADVJUDGE_OUTPUT_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/outputs")
    return Path.cwd() / "outputs"


def _is_writable_dir(path: Path) -> bool:
    """Return True if ``path`` can be created and written to."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _resolve_output(raw: str, base: Path) -> Path:
    """Resolve an output path, redirecting to ``base`` if unwritable.

    * Relative paths are joined onto ``base``.
    * Absolute paths are returned as-is if writable, else redirected.
    """
    p = Path(raw)
    if not p.is_absolute():
        target = base / p.name
    else:
        target = p

    if _is_writable_dir(target.parent):
        return target

    fallback = base / p.name
    log.warning(
        "Output path %s is not writable; redirecting to %s", target, fallback
    )
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    patience: int = 15


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
    attn_implementation: str | None = "eager"


@dataclass
class Config:
    seed: int
    paths: Paths
    generation: GenerationConfig
    evaluation: EvaluationConfig
    hf_token: str | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = raw["paths"]
    paths = Paths(
        test_csv=Path(raw_paths["test_csv"]),
        submission_csv=_resolve_output(raw_paths["submission_csv"], output_dir),
        words_json=_resolve_output(raw_paths["words_json"], output_dir),
        attacks_json=_resolve_output(raw_paths["attacks_json"], output_dir),
        scores_json=_resolve_output(raw_paths["scores_json"], output_dir),
    )

    gen_raw = raw["generation"]
    gen = GenerationConfig(
        vocab=VocabConfig(**gen_raw["vocab"]),
        nonsense=NonsenseConfig(**gen_raw["nonsense"]),
        exploits=gen_raw["exploits"],
        baseline=gen_raw.get("baseline", {}),
        essay_max_chars=gen_raw.get("essay_max_chars", 900),
    )

    ev_raw = raw["evaluation"]
    # Forward-compatibility: allow older YAMLs without the new field.
    ev_raw.setdefault("attn_implementation", "eager")
    ev = EvaluationConfig(**ev_raw)

    hf_token = os.getenv("HF_TOKEN")
    return Config(
        seed=raw["seed"],
        paths=paths,
        generation=gen,
        evaluation=ev,
        hf_token=hf_token,
    )