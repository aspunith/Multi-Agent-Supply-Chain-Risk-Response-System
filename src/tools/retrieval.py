"""Retrieval + anomaly tools for the Signal agent.

lead_time_anomaly  numeric z-score check on recent lead times vs a baseline.
NewsRetriever      TF-IDF search over the supplier-news corpus.

TF-IDF is enough for a small corpus (deterministic, no external calls); move to dense
embeddings + hybrid retrieval once the corpus grows.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def lead_time_anomaly(recent: list[float], baseline_mean: float, baseline_std: float) -> dict:
    """Flag a disruption if recent mean lead time is a statistical outlier vs baseline."""
    recent_mean = float(np.mean(recent)) if recent else baseline_mean
    std = max(baseline_std, 1e-6)
    z = (recent_mean - baseline_mean) / std
    is_anomaly = z >= 2.0
    return {
        "recent_mean_lead_time": round(recent_mean, 2),
        "baseline_mean": round(baseline_mean, 2),
        "z_score": round(float(z), 2),
        "is_anomaly": bool(is_anomaly),
    }


class NewsRetriever:
    """Tiny TF-IDF retriever over supplier news snippets."""

    _DISRUPTION_TERMS = ("delay", "congestion", "shortage", "strike", "disruption", "outage")

    def __init__(self, docs: list[dict]):
        """docs: [{'supplier_id': ..., 'text': ...}, ...]"""
        self.docs = docs
        self._texts = [d["text"] for d in docs]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        q = self._vectorizer.transform([text])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.docs[i], "score": round(float(sims[i]), 3)}
            for i in order
            if sims[i] > 0
        ]

    def disruption_flag(self, supplier_id: str) -> dict:
        """Keyword screen of a supplier's own news doc for disruption language."""
        text = next((d["text"] for d in self.docs if d["supplier_id"] == supplier_id), "")
        hits = [t for t in self._DISRUPTION_TERMS if t in text.lower()]
        return {"supplier_id": supplier_id, "keyword_hits": hits, "flagged": bool(hits), "text": text}
