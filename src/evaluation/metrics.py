from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import re
import numpy as np
from scipy.spatial.distance import pdist


@dataclass
class CompetitionMetrics:
    avg_q: float
    avg_h: float
    min_v: float
    avg_e: float
    avg_s: float

    @property
    def final_score(self) -> float:
        denom = self.avg_s * (9.0 - self.avg_q)
        if denom <= 0:
            return float("inf")
        return (self.avg_h * self.min_v * self.avg_e) / denom


def _tfidf_cosine(essays: list[str]) -> float:
    tokenized = [re.findall(r"\w+", e.lower()) for e in essays]
    n = len(tokenized)
    if n < 2:
        return 0.0

    vocab = set().union(*(set(t) for t in tokenized))
    doc_freq = Counter()
    for toks in tokenized:
        doc_freq.update(set(toks))
    idf = {w: np.log(n / (doc_freq[w] + 1)) for w in vocab}

    def vec(toks):
        tf = Counter(toks)
        length = max(len(toks), 1)
        return np.array([tf.get(w, 0) / length * idf[w] for w in vocab])

    vectors = np.array([vec(t) for t in tokenized])
    sims = 1 - pdist(vectors, metric="cosine")
    return float(np.mean(sims)) if sims.size else 0.0


def compute_metrics(
    scores: list[list[float]],
    english_scores: list[float],
    essays: list[str],
    similarity_floor: float = 0.2,
) -> CompetitionMetrics:
    arr = np.array(scores)
    avg_quality = np.mean(arr, axis=1)
    horizontal = np.std(arr, axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0])
    vertical = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])

    avg_s = max(_tfidf_cosine(essays), similarity_floor)
    return CompetitionMetrics(
        avg_q=float(np.mean(avg_quality)),
        avg_h=float(np.mean(horizontal)),
        min_v=float(np.min(vertical)),
        avg_e=float(np.mean(english_scores)) if english_scores else 1.0,
        avg_s=avg_s,
    )