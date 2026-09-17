import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation.attack_assigner import build_balanced_type_list


def test_balanced_counts():
    types = build_balanced_type_list(1000)
    counts = Counter(types).values()
    assert max(counts) - min(counts) <= 1
    assert len(types) == 1000