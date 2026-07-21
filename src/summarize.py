"""Summarising a chat, in two tiers.

Extractive comes first and never depends on the model: it ranks real messages
by how central they are to the conversation and returns them verbatim, so it
cannot hallucinate. Abstractive is the nicer read, but it is optional and runs
map-reduce because a small model's context window is far shorter than a real
group chat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_WORDS = 4          # one-word replies carry no information to extract
CHUNK_MESSAGES = 60    # per map step in the abstractive path


@dataclass
class ExtractiveSummary:
    messages: pd.DataFrame     # the selected messages, in time order
    method: str = "centroid TF-IDF"


def extractive(df: pd.DataFrame, n: int = 8) -> ExtractiveSummary:
    """Pick the n most representative messages.

    Scores each message by cosine similarity to the centroid of the whole
    conversation, which favours messages that sit near the middle of what the
    group was actually talking about.
    """
    u = df[df["is_user"] & (df["kind"] == "text")].copy()
    u = u[u["n_words"] >= MIN_WORDS]
    if len(u) < 2:
        return ExtractiveSummary(u.sort_values("timestamp"))

    vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, stop_words="english",
                          min_df=1, max_df=0.9, strip_accents="unicode")
    try:
        X = vec.fit_transform(u["text"].tolist())
    except ValueError:
        return ExtractiveSummary(u.head(n).sort_values("timestamp"))

    centroid = np.asarray(X.mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)
    if norm < 1e-9:
        return ExtractiveSummary(u.head(n).sort_values("timestamp"))

    scores = (X @ centroid) / norm
    scores = np.asarray(scores).ravel()
    u = u.assign(_score=scores)

    # Spread picks across the timeline rather than clustering them in one busy
    # hour: take the best message from each of n equal time slices.
    u = u.sort_values("timestamp")
    slices = np.array_split(np.arange(len(u)), min(n, len(u)))
    picks = [u.iloc[s]["_score"].idxmax() for s in slices if len(s)]
    out = u.loc[picks].sort_values("timestamp").drop(columns=["_score"])
    return ExtractiveSummary(out)


def format_extractive(summary: ExtractiveSummary) -> str:
    lines = [
        f"**{r['sender']}** ({r['timestamp']:%d %b %H:%M}) — {r['text']}"
        for _, r in summary.messages.iterrows()
    ]
    return "\n\n".join(lines) if lines else "_Not enough text to summarise._"


def _chunk_texts(df: pd.DataFrame, size: int = CHUNK_MESSAGES) -> list[str]:
    u = df[df["is_user"] & (df["kind"] == "text")].sort_values("timestamp")
    rows = [f"{r['sender']}: {r['text']}" for _, r in u.iterrows()]
    return ["\n".join(rows[i:i + size]) for i in range(0, len(rows), size)]


def abstractive(df: pd.DataFrame, llm, max_chunks: int = 6) -> tuple[str, str]:
    """Map-reduce summary. Returns (summary, note).

    A small model cannot see a whole chat at once, so each chunk is summarised
    separately and those summaries are then summarised together.
    """
    chunks = _chunk_texts(df)
    if not chunks:
        return "", "No text messages to summarise."

    note = ""
    if len(chunks) > max_chunks:
        # Sample evenly across the whole span rather than truncating to the start.
        idx = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in sorted(set(idx))]
        note = (f"Sampled {len(chunks)} sections evenly across the chat — "
                "summarising every message would take far too long on CPU.")

    partials = []
    for c in chunks:
        try:
            partials.append(llm.generate(
                "Summarise what this part of a group chat is about, in one or "
                f"two sentences.\n\n{c[:3000]}\n\nSUMMARY:",
                max_new_tokens=110,
            ))
        except Exception as e:  # noqa: BLE001
            return "", f"Generation failed: {e}"

    if len(partials) == 1:
        return partials[0], note

    joined = "\n".join(f"- {p}" for p in partials)
    try:
        final = llm.generate(
            "These are summaries of consecutive parts of one group chat. Combine "
            "them into a single short paragraph describing what the group talks "
            f"about.\n\n{joined[:3500]}\n\nOVERALL SUMMARY:",
            max_new_tokens=200,
        )
    except Exception as e:  # noqa: BLE001
        return "\n".join(partials), f"Could not combine sections: {e}"
    return final, note


def label_topic(top_words: list[str], llm) -> str:
    """Turn a topic's top terms into a readable name."""
    if llm is None:
        return ", ".join(top_words[:5])
    try:
        out = llm.generate(
            "These words describe one topic from a group chat: "
            f"{', '.join(top_words[:10])}.\n"
            "Give a short label for this topic, at most four words. "
            "Reply with the label only.",
            max_new_tokens=20, temperature=0.2,
        )
        label = out.strip().strip('"').split("\n")[0]
        return label[:50] if label else ", ".join(top_words[:5])
    except Exception:  # noqa: BLE001
        return ", ".join(top_words[:5])
