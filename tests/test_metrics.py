import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import compute_metrics


def test_final_score_positive():
    scores = [[9, 0, 9, 0], [0, 9, 0, 9], [5, 5, 5, 5]]
    english = [0.9, 0.8, 1.0]
    essays = ["alpha beta", "gamma delta", "epsilon zeta"]
    m = compute_metrics(scores, english, essays, similarity_floor=0.2)
    assert m.final_score > 0
    assert m.avg_s >= 0.2