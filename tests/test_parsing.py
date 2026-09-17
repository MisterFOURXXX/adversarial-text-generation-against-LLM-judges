import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.parsing import parse_score


def test_parse_numeric():
    assert parse_score("Score: 7") == 7.0
    assert parse_score("9") == 9.0
    assert parse_score("no number") == 4.5
    assert parse_score("10") == 9.0  # clamped