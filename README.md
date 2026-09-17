# LLM Adversarial Judge Exploitation

A modular, reproducible research repository for studying adversarial
text-generation attacks against **LLM-as-a-judge** committees. The pipeline
generates short adversarial essays that maximise scoring disagreement among a
panel of language models tasked with grading subjective text.

This repository is a refactor of the original competition notebook into a
research-grade Python package with typed configuration, structured logging,
CLI entry points, and unit tests.

> **⚠️ Research use only.** The techniques and exploit strings in this
> repository are documented for AI-safety research and defensive ablation.
> See [§8 Ethics & Intended Use](#8-ethics--intended-use) before proceeding.

---

## Table of contents

1. [Overview](#1-overview)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Usage](#4-usage)
5. [Configuration](#5-configuration)
6. [Method summary](#6-method-summary)
7. [Reproducibility](#7-reproducibility)
8. [Ethics & intended use](#8-ethics--intended-use)
9. [References](#9-references)
10. [License](#10-license)

---

## 1. Overview

Modern LLM-as-a-judge pipelines are widely deployed for subjective evaluation
tasks (essay grading, response ranking, code review), yet they remain
vulnerable to a family of well-documented failure modes:

- **Prompt injection / jailbreaks** — contradictory or conditional
  instructions embedded in the input text.
- **Style bias** — inflated scores when adversarial inputs look technical,
  verbose, or code-like.
- **Length bias** — rewarding longer, more "complex" text regardless of
  substance.
- **Self- and position-bias** — inconsistent scoring across committee members
  depending on their identity, ordering, or training lineage.

This repository implements a two-stage attack pipeline:

1. **Adversarial generation** — `Qwen2.5-3B-Instruct` produces a diverse,
   technical vocabulary; `Llama-3.1-8B-Instruct` assembles that vocabulary
   into "nonsense" payloads that are combined with hand-crafted exploit
   strings (`099`, `990`, `909`).
2. **Committee evaluation** — four quantized judges
   (`Gemma-2-2B-IT`, `Qwen3-4B-Instruct-2507`, `Llama-3.1-8B-Instruct`,
   `Phi-3.5-mini-instruct`) score each essay under **greedy decoding**, so
   scores are deterministic for a given input and model revision.

The competition objective is a scalar:

```
score = (avg_h * min_v * avg_e) / (avg_s * (9 - avg_q))
```

| Symbol    | Meaning                                                       | Desired direction |
|-----------|---------------------------------------------------------------|-------------------|
| `avg_q`   | Mean quality score across all essays and judges               | — (penalty term)  |
| `avg_h`   | Mean *horizontal* std-dev across judges per essay             | ↑ maximise        |
| `min_v`   | Minimum *vertical* std-dev across essays for any single judge | ↑ maximise        |
| `avg_e`   | Mean English-language confidence (Lingua)                     | ↑ maximise        |
| `avg_s`   | Mean pairwise TF-IDF cosine similarity between essays         | ↓ (floored at 0.2) |

The numerator rewards disagreement (a well-designed committee should not be
easily fooled) and coherence. The denominator penalises low-quality consensus
and inter-essay repetition.

---

## 2. Repository layout

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml             # Pipeline behaviour
│   └── models.yaml              # Model registry + quantization presets
├── data/
│   ├── .gitkeep
│   └── test.csv                 # Not tracked; supply your own
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py                # Typed config dataclasses + YAML loader
│   ├── logging_utils.py         # Structured logging
│   ├── pipeline.py              # End-to-end orchestration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seeds.py             # Reproducibility
│   │   └── memory.py            # GPU cleanup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── quantization.py      # BitsAndBytesConfig factory
│   │   └── loader.py            # HF model/tokenizer loading
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── vocabulary.py        # Qwen-based word pool
│   │   ├── nonsense.py          # Llama-based nonsense generator
│   │   ├── exploits.py          # Exploit-string registry
│   │   ├── baseline.py          # Positive-review generation
│   │   └── attack_assigner.py   # Balanced attack distribution
│   └── evaluation/
│       ├── __init__.py
│       ├── templates.py         # Per-model chat templates
│       ├── parsing.py           # Score regex extraction
│       ├── english.py           # Lingua + fallback detector
│       ├── judge.py             # Committee judge runner
│       └── metrics.py           # Competition metrics
├── scripts/
│   ├── generate_submission.py
│   └── evaluate_submission.py
└── tests/
    ├── __init__.py
    ├── test_metrics.py
    ├── test_parsing.py
    └── test_attack_assigner.py
```

---

## 3. Installation

**Requirements**

- Python ≥ 3.10
- A CUDA-capable GPU (4-bit NF4 quantization is used throughout so the full
  ~17 B-parameter pipeline fits on a single 24 GB device)
- `bitsandbytes` ≥ 0.48 (some earlier versions fail on Qwen3/Qwen2.5)

**Steps**

```bash
git clone <repo-url> llm-adv-judge
cd llm-adv-judge

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

**Hugging Face token**

`meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen*` are gated. Export a token
with access to those repositories:

```bash
export HF_TOKEN="hf_..."
```

The loader passes the token only to models whose name matches `llama` or
`qwen`, so ungated committee members (`gemma`, `phi`) work without one.

**Smoke test**

```bash
pytest -q
```

All tests are CPU-only and do not require model downloads.

---

## 4. Usage

### 4.1 Prepare input

Place your test set under `data/`:

```
data/test.csv
```

Required columns:

| column | type   | description                                              |
|--------|--------|----------------------------------------------------------|
| `id`   | any    | Unique identifier; echoed verbatim into the submission.  |
| `topic`| string | Essay prompt; used to generate baseline reviews.         |

The public `test.csv` for the original competition contains 3 rows; the hidden
set has ~1000.

### 4.2 Generate a submission

```bash
python scripts/generate_submission.py --config configs/default.yaml
```

Artefacts written to `outputs/`:

| File                  | Contents                                                     |
|-----------------------|--------------------------------------------------------------|
| `generated_words.json`| Cached vocabulary pool (re-used on subsequent runs)          |
| `attacks.json`        | Exploit pools — 5 examples per exploit type                  |
| `submission.csv`      | Final essays, ready for the committee                        |

The generator caches `generated_words.json` and `attacks.json`; delete either
to force regeneration. This makes iterating on the baseline or the assignment
strategy cheap while keeping the expensive vocabulary step fixed.

### 4.3 Evaluate a submission

```bash
python scripts/evaluate_submission.py --config configs/default.yaml
```

Artefacts:

| File          | Contents                                                          |
|---------------|-------------------------------------------------------------------|
| `scores.json` | Per-essay judge scores, English confidences, metrics, final score |

Console output includes a structured log line:

```
avg_q=6.3063 avg_h=3.1499 min_v=3.5409 avg_e=0.2559 avg_s=0.2000 final=5.2978
```

### 4.4 Run tests

```bash
pytest -q
```

### 4.5 End-to-end

```bash
python scripts/generate_submission.py && \
python scripts/evaluate_submission.py
```

---

## 5. Configuration

All tunables live under `configs/`. Two files drive the pipeline:

| File           | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `default.yaml` | Runtime behaviour: seed, I/O paths, generation hyperparameters, committee |
| `models.yaml`  | Model registry: HF ids, parameter counts, quantization presets, judge prompt |

Every field in `default.yaml` maps to a dataclass in `src/config.py`.
**Unknown keys raise `TypeError` at load time** — this is intentional so
configuration drift is caught immediately rather than silently ignored.

### Key generation knobs

| Setting                              | Meaning                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
| `generation.vocab.temperature`       | Higher → more diverse / adversarial word pool                           |
| `generation.vocab.top_p`             | Nucleus sampling threshold for vocabulary                               |
| `generation.nonsense.temperature`    | Higher → more chaotic nonsense payloads                                 |
| `generation.nonsense.top_p`          | Nucleus sampling threshold for nonsense                                 |
| `generation.essay_max_chars`         | Hard cap on final essay length (competition targets ~100 words)         |
| `generation.exploits.examples_per_type` | Pool size per exploit; used cyclically across the test set           |
| `evaluation.committee`               | Ordered list of judge model ids (keep consistent with `models.yaml`)    |
| `evaluation.max_input_tokens`        | Truncation limit for judge prompts (default 4096)                       |
| `evaluation.similarity_floor`        | Lower bound on `avg_s` per competition rules (default 0.2)              |

### Swapping a judge

1. Edit `configs/models.yaml` to add the new model under
   `evaluation.committee` with the correct `template_key`.
2. Update `evaluation.committee` in `configs/default.yaml` to match.
3. If the model uses a non-standard chat format, add a template entry in
   `src/evaluation/templates.py::MODEL_CONFIGS`.

### Swapping quantization

The pipeline currently hard-wires `nf4_fp16_double` via
`src/models/quantization.py::build_nf4_config`. To ablate another preset,
either edit that function or extend `Config` to accept a
`quantization_preset` field and look it up in `models.yaml`.

---

## 6. Method summary

```
              ┌────────────────────┐
              │   test.csv topics  │
              └─────────┬──────────┘
                        │
        ┌───────────────┴─────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────┐               ┌───────────────────┐
│ Qwen vocab    │               │ Llama nonsense    │
│ 1000 words    │──────────────▶│ + exploit strings │
└───────────────┘               └─────────┬─────────┘
                                          │
                              ┌───────────┴───────────┐
                              │   Balanced attack     │
                              │   assignment (≈ 1/3)  │
                              └───────────┬───────────┘
                                          │
                              ┌───────────┴───────────┐
                              │  Baseline reviews for │
                              │  non-attacked rows    │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  LLM committee judge  │
                              │  (4 × quantized LLMs) │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  Competition metrics  │
                              └───────────────────────┘
```

### Stage-by-stage

**Stage 1 — Vocabulary (`src/generation/vocabulary.py`).**
`Qwen2.5-3B-Instruct` is sampled 100 times at `temperature=1.2`,
`top_p=0.95`, `max_new_tokens=2000`. Words shorter than 6 characters or
containing digits/punctuation (other than internal hyphens) are discarded.
The first 1000 unique survivors are cached to `generated_words.json`.

**Stage 2 — Nonsense payloads (`src/generation/nonsense.py`).**
`Llama-3.1-8B-Instruct` is prompted via its chat template to emit long
sequences of complex technical words at `temperature=0.9`, `top_p=0.95`.
The decoded response is split on the assistant marker, filtered to
alphabetic tokens ≥ 6 characters, and truncated to `n_words` (default 100).
Shortfalls are backfilled by sampling with replacement from the survivors.

**Stage 3 — Exploit assembly (`src/generation/exploits.py`).**
Each nonsense payload is concatenated with one of three hand-crafted exploit
strings:

| Key   | Pattern                                              | Target bias            |
|-------|------------------------------------------------------|------------------------|
| `099` | Contradictory numeric instructions ("9" / "must be 0") | Instruction-following |
| `990` | Conditional branching on judge identity ("from Google?") | Self-bias / parsing  |
| `909` | Fenced code-block containing a literal score          | Style bias / formatting |

Placeholders (`{}`) in `099` and `909` are filled with a second nonsense
payload at assembly time. The final string is right-trimmed to
`essay_max_chars` (default 900).

**Stage 4 — Balanced assignment (`src/generation/attack_assigner.py`).**
A list of type indices with counts differing by ≤ 1 is generated, shuffled
with the run's seed, then mapped to exploit keys in a fixed order. Five
pre-generated examples per exploit type are cycled across the submission.

**Stage 5 — Baseline reviews (`src/generation/baseline.py`).**
Non-attacked rows receive a short, highly positive review generated by
`Llama-3.1-8B-Instruct` at `temperature=0.2` under one of three
instruction templates. Two suffix heuristics (`SUFFIX_A`, `SUFFIX_B`) are
appended to alternating rows to reinforce the "9/9" signal and, in the case
of `SUFFIX_B`, to inject an additional conditional prompt-injection hint.

**Stage 6 — Committee judging (`src/evaluation/judge.py`).**
Each of the four judges receives the full essay inside a model-specific
chat template (`<start_of_turn>` for Gemma, `<|user|>` for Phi,
`<|im_start|>` for Qwen, `<|start_header_id|>` for Llama). Decoding is
**greedy** (`do_sample=False`, `max_new_tokens=10`) so scores are
deterministic. Output is parsed by the first numeric regex match; missing
numbers fall back to `evaluation.default_score` (4.5).

**Stage 7 — Metrics (`src/evaluation/metrics.py`).**
`avg_q`, `avg_h`, `min_v`, and `avg_e` are computed from the score matrix
and English-confidence vector. `avg_s` is computed from TF-IDF vectors
(log-IDF, TF normalised by essay length) via pairwise cosine distance, then
floored at `evaluation.similarity_floor` per competition rules.

---

## 7. Reproducibility

- **Seeds.** Global seeds are set via `utils.seeds.set_seeds` using
  `config.seed`. `random`, `numpy`, and `torch` (CPU + all CUDA devices) are
  all seeded.
- **Quantization.** Every model is loaded with
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. Compute
  dtype is FP16 by default; use the `nf4_bf16` preset on Ampere+ GPUs if
  numeric drift is a concern.
- **Decoding.** Judge decoding is greedy (`do_sample=False`) so scores are
  deterministic for a given input and model revision. Generation stages
  sample stochastically by design, but their outputs are cached in
  `generated_words.json` and `attacks.json`.
- **Caching.** Delete `outputs/generated_words.json` to regenerate the
  vocabulary; delete `outputs/attacks.json` to regenerate the exploit pools.
- **Revision pinning.** For archival runs, pin each model to a specific
  commit, e.g.
  `meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659`.
  `configs/models.yaml` documents the format; `default.yaml` uses the bare
  ids for developer convenience.

---

## 8. Ethics & intended use

This repository is intended **strictly for AI-safety research**: to
characterise how easily LLM-based evaluators can be manipulated, so that
defensive mechanisms (ensembles, consensus scoring, adversarial training,
logit-level anomaly detection) can be developed and validated.

**Do not** deploy these techniques against:

- production grading or ranking systems,
- human reviewers,
- any context where the goal is to deceive or extract unearned scores.

The exploit strings in `src/generation/exploits.py` are included for
reproducibility and ablation. They are not novel jailbreaks; they are
documented patterns that already appear in the public literature
(see [§9](#9-references)).

If you use this code, please cite the underlying research and note that the
metric-maximisation objective is a proxy for adversarial robustness, **not**
a goal in itself.

---

## 9. References

1. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and
   Chatbot Arena.* NeurIPS.
2. Wang, P., et al. (2023). *Large Language Models are not Fair Evaluators.*
   arXiv:2305.17926.
3. Panickssery, A., et al. (2024). *LLM Evaluators Recognize and Favor Their
   Own Generations.* arXiv:2404.13076.
4. Wallace, E., et al. (2021). *Universal Adversarial Triggers for Attacking
   and Analyzing NLP.* EMNLP.
5. Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on
   Aligned Language Models.* arXiv:2307.15043.
6. Li, H., et al. (2024). *Lockpicking LLMs: A Logit-Based Jailbreak Using
   Token-level Manipulation.* arXiv:2405.13068.
7. Rando, J., et al. (2024). *Finding Universal Jailbreak Backdoors in
   Aligned LLMs.* arXiv:2404.14461.

Additional resources referenced by the original notebook:

- `lingua-language-detector` for English confidence estimation.
- `scipy.spatial.distance.pdist` for pairwise cosine similarity.
- Hugging Face `transformers` and `bitsandbytes` for quantized inference.

---

## 10. License

Provided for academic and defensive research use. See `LICENSE` for the full
text. Redistribution of the exploit strings outside a research context is
discouraged; if you fork this repository, preserve the
[§8 Ethics & Intended Use](#8-ethics--intended-use) section verbatim.