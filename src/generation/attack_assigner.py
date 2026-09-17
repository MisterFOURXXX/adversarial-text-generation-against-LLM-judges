"""Balanced assignment of exploit types across the test set.

The competition's similarity metric (``avg_s``) is sensitive to how many
essays share each exploit type. If one exploit dominates the submission, its
stylistic fingerprints inflate inter-essay similarity and depress the final
score. This module distributes the three exploit types as evenly as possible
and provides a cyclic selector so a small pool of pre-generated attacks can
cover a much larger submission.
"""

from __future__ import annotations

from collections import Counter
import random


# Canonical mapping from type index -> exploit key in ``EXPLOITS``.
# Kept here (rather than imported from ``exploits.py``) so the assigner stays
# import-light and testable without touching model code.
TYPE_TO_EXPLOIT_KEY: dict[int, str] = {
    0: "909",
    1: "099",
    2: "990",
}

N_TYPES = len(TYPE_TO_EXPLOIT_KEY)


def build_balanced_type_list(
    n_total: int,
    rng: random.Random | None = None,
) -> list[int]:
    """Return a shuffled list of length ``n_total`` with near-equal counts.

    Counts of each type differ by at most 1 (``n_total % 3`` types get one
    extra entry). The result is shuffled with the supplied RNG so that
    assignments are reproducible given a seeded ``random.Random``.

    Parameters
    ----------
    n_total:
        Number of entries to produce (typically ``len(test_df)``).
    rng:
        Optional ``random.Random`` instance for deterministic shuffling.
        Defaults to a fresh, unseeded instance.

    Raises
    ------
    ValueError
        If ``n_total`` is negative.
    """
    if n_total < 0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")

    rng = rng or random.Random()
    base, rem = divmod(n_total, N_TYPES)

    types: list[int] = []
    for t in range(N_TYPES):
        count = base + (1 if t < rem else 0)
        types.extend([t] * count)

    rng.shuffle(types)

    # Sanity check: the whole point of this function is balance.
    counts = Counter(types)
    if counts and (max(counts.values()) - min(counts.values())) > 1:
        raise AssertionError(f"Type distribution is imbalanced: {counts}")

    return types


def cycle_pick(pool: list[str], idx: int) -> str:
    """Select ``pool[idx % len(pool)]`` so a small pool covers many rows.

    Raises
    ------
    ValueError
        If ``pool`` is empty.
    """
    if not pool:
        raise ValueError("cycle_pick received an empty pool")
    return pool[idx % len(pool)]


def counts_per_type(type_list: list[int]) -> dict[int, int]:
    """Return a ``{type_index: count}`` summary for logging/assertions."""
    return dict(Counter(type_list))


__all__ = [
    "TYPE_TO_EXPLOIT_KEY",
    "N_TYPES",
    "build_balanced_type_list",
    "cycle_pick",
    "counts_per_type",
]