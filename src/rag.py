"""Chat with your chats — retrieval-augmented generation over a group chat.

Retrieval does the heavy lifting and generation is a thin layer on top. That is
a deliberate split: TF-IDF over conversation windows is genuinely strong for the
name-and-keyword questions people ask a chat log ("what did Alice say about the
trip?"), costs nothing to run, and always returns real messages with real
timestamps. The small model only rephrases what retrieval already found, and
every answer ships with its sources so you can check it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

WINDOW_SIZE = 20        # messages per retrievable chunk
WINDOW_STRIDE = 10      # overlap, so a topic split across a boundary is still found


@dataclass
class Window:
    """A retrievable block of consecutive messages."""

    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    senders: list[str]
    lines: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        return "\n".join(self.lines)

    def header(self) -> str:
        return f"{self.start:%d %b %Y %H:%M} – {self.end:%H:%M}"


def build_windows(df: pd.DataFrame, size: int = WINDOW_SIZE,
                  stride: int = WINDOW_STRIDE) -> list[Window]:
    """Slice the chat into overlapping conversation windows."""
    u = df[df["is_user"] & (df["kind"] == "text")].sort_values("timestamp")
    if u.empty:
        return []

    rows = u[["timestamp", "sender", "text"]].to_dict("records")
    windows: list[Window] = []
    for i in range(0, len(rows), stride):
        chunk = rows[i:i + size]
        if not chunk:
            break
        windows.append(Window(
            index=len(windows),
            start=chunk[0]["timestamp"],
            end=chunk[-1]["timestamp"],
            senders=sorted({c["sender"] for c in chunk}),
            lines=[f"[{c['timestamp']:%d %b %H:%M}] {c['sender']}: {c['text']}"
                   for c in chunk],
        ))
        if i + size >= len(rows):
            break
    return windows


@dataclass
class Hit:
    window: Window
    score: float


class Retriever:
    """TF-IDF retrieval over conversation windows."""

    def __init__(self, windows: list[Window]):
        self.windows = windows
        self._vec = None
        self._matrix = None
        if windows:
            # Sub-linear tf damps a window that merely repeats one word a lot.
            self._vec = TfidfVectorizer(
                lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
                min_df=1, max_df=0.9, strip_accents="unicode",
            )
            self._matrix = self._vec.fit_transform([w.as_text() for w in windows])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self.windows or self._vec is None or not query.strip():
            return []
        q = self._vec.transform([query])
        sims = (self._matrix @ q.T).toarray().ravel()
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Hit(self.windows[i], float(sims[i])) for i in top if sims[i] > 0]


def build_prompt(question: str, hits: list[Hit], max_chars: int = 4000) -> str:
    """Assemble a grounded prompt, trimmed to fit a small context window."""
    parts, used = [], 0
    for h in hits:
        block = f"--- {h.window.header()} ---\n{h.window.as_text()}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)

    context = "\n".join(parts) if parts else "(no relevant messages found)"
    return (
        "You are answering questions about a WhatsApp group chat.\n"
        "Use ONLY the excerpts below. If they do not contain the answer, say so "
        "plainly — do not guess.\n"
        "Keep the answer to two or three sentences and name who said what.\n\n"
        f"CHAT EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    generated: bool          # False when returning retrieval only
    note: str = ""


def answer(question: str, retriever: Retriever, llm=None, k: int = 5) -> Answer:
    """Retrieve, then optionally generate. Sources always come back."""
    hits = retriever.search(question, k=k)
    if not hits:
        return Answer(
            "Nothing in this chat matches that question. Try different wording, "
            "or a name that appears in the conversation.",
            [], generated=False,
        )

    if llm is None:
        return Answer(
            "", hits, generated=False,
            note="Showing the most relevant messages. Enable the language model "
                 "in the sidebar for a written answer.",
        )

    try:
        text = llm.generate(build_prompt(question, hits))
        return Answer(text, hits, generated=True)
    except Exception as e:  # noqa: BLE001 - fall back to retrieval, never crash
        return Answer(
            "", hits, generated=False,
            note=f"Generation failed, showing retrieved messages instead. ({e})",
        )
