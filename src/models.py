"""Load and apply the models produced by the training scripts.

The app only ever loads. Training happens in `train/*.py` and writes joblib
artifacts to `models/`, so nothing slow ever runs inside a Streamlit rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

SENTIMENT_PATH = MODEL_DIR / "sentiment.joblib"
AUTHOR_PATH = MODEL_DIR / "author_id.joblib"
TOPICS_PATH = MODEL_DIR / "topics.joblib"


def _load(path: Path):
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:  # noqa: BLE001
        return None


@dataclass
class SentimentModel:
    pipeline: object
    metrics: dict

    def predict(self, texts: list[str]) -> np.ndarray:
        """1 for positive, 0 for negative."""
        return self.pipeline.predict(texts)

    def score(self, texts: list[str]) -> np.ndarray:
        """Signed decision value; larger means more positive."""
        if hasattr(self.pipeline, "decision_function"):
            return np.asarray(self.pipeline.decision_function(texts)).ravel()
        proba = self.pipeline.predict_proba(texts)[:, 1]
        return proba - 0.5


@dataclass
class AuthorModel:
    pipeline: object
    labels: list[str]
    metrics: dict

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)


@dataclass
class TopicModel:
    vectorizer: object
    nmf: object
    top_words: list[list[str]]
    top_weights: list[list[float]]
    metrics: dict

    @property
    def n_topics(self) -> int:
        return len(self.top_words)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.nmf.transform(self.vectorizer.transform(texts))

    def assign(self, texts: list[str]) -> np.ndarray:
        """Dominant topic index per document."""
        w = self.transform(texts)
        return w.argmax(axis=1)


def load_sentiment() -> SentimentModel | None:
    d = _load(SENTIMENT_PATH)
    return SentimentModel(d["pipeline"], d.get("metrics", {})) if d else None


def load_author() -> AuthorModel | None:
    d = _load(AUTHOR_PATH)
    return AuthorModel(d["pipeline"], d.get("labels", []), d.get("metrics", {})) if d else None


def load_topics() -> TopicModel | None:
    d = _load(TOPICS_PATH)
    if not d:
        return None
    return TopicModel(d["vectorizer"], d["nmf"], d["top_words"],
                      d.get("top_weights", []), d.get("metrics", {}))


def available() -> dict[str, bool]:
    return {
        "sentiment": SENTIMENT_PATH.exists(),
        "author_id": AUTHOR_PATH.exists(),
        "topics": TOPICS_PATH.exists(),
    }
