"""Fit the models the app displays, from a chat that is already in memory.

Both the `train/` scripts and the Train buttons in the app call these
functions, so the two paths cannot drift. Nothing here touches the filesystem
or Streamlit: text goes in, a fitted model and the metrics needed to judge it
come out.

Failures are returned, not raised. Training legitimately fails on real chats —
a group that never uses emoji cannot be labelled by emoji, and three people
with forty messages between them cannot support an authorship model. Those are
results to report, not exceptions.

Every fit is deterministic given a seed, including the subsampling used to keep
very large chats inside a modest memory budget.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import models as M
from . import parser as P
from . import text as T

POSITIVE_EMOJI = set("😀😃😄😁😆😊🙂😍🥰😘🤗🤩😎👍👌🙏❤️💕💖💯🎉🥳😂🤣✨🔥💪👏🙌😇🥹")
NEGATIVE_EMOJI = set("😞😔😟😕🙁☹️😣😖😫😩😢😭😤😠😡🤬😰😨😱😥💔👎🤢🤮😒🙄😬😐")

BLOCK = 8            # messages per pseudo-document for topic modelling


@dataclass
class TrainResult:
    """The outcome of one training run — success or otherwise."""

    model: Any = None            # a models.SentimentModel / AuthorModel / TopicModel
    payload: dict | None = None  # exactly what the CLI scripts write with joblib
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.model is not None

    def log(self) -> str:
        return "\n".join(self.notes)


# --- pulling training input out of whatever the caller has ------------------

def texts_from_messages(messages: list[P.Message]) -> list[str]:
    """Every real text message, in order."""
    return [m.text for m in messages if m.is_user and m.kind == P.KIND_TEXT]


def pairs_from_messages(messages: list[P.Message]) -> list[tuple[str, str]]:
    """(text, sender) for every real text message."""
    return [(m.text, m.sender) for m in messages
            if m.is_user and m.kind == P.KIND_TEXT]


def texts_from_dataframe(df) -> list[str]:
    sub = df[df["is_user"] & (df["kind"] == P.KIND_TEXT)]
    return sub["text"].tolist()


def pairs_from_dataframe(df) -> list[tuple[str, str]]:
    sub = df[df["is_user"] & (df["kind"] == P.KIND_TEXT)]
    return list(zip(sub["text"].tolist(), sub["sender"].tolist()))


def _subsample(items: list, cap: int | None, seed: int) -> tuple[list, bool]:
    """Keep at most `cap` items, chosen at random but in original order."""
    if cap is None or len(items) <= cap:
        return items, False
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=cap, replace=False))
    return [items[i] for i in idx], True


# --- sentiment --------------------------------------------------------------

def label_by_emoji(text: str) -> int | None:
    """1 positive, 0 negative, None when the message gives no clear signal."""
    found = set(T.extract_emoji(text))
    pos = len(found & POSITIVE_EMOJI)
    neg = len(found & NEGATIVE_EMOJI)
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    return None


def sentiment_examples(texts: list[str]) -> tuple[list[str], list[int]]:
    """Distant supervision: label by emoji, then remove the emoji.

    Stripping the emoji is the whole point — it forces the classifier to learn
    sentiment from words, so it can go on to label the ~90% of messages that
    carry no emoji at all.
    """
    xs, ys = [], []
    for t in texts:
        y = label_by_emoji(t)
        if y is None:
            continue
        stripped = T.strip_emoji(t).strip()
        if len(stripped.split()) < 2:      # nothing left to learn from
            continue
        xs.append(stripped)
        ys.append(y)
    return xs, ys


def train_sentiment(texts: list[str], *, min_examples: int = 60,
                    test_size: float = 0.25, seed: int = 0,
                    max_examples: int | None = None) -> TrainResult:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    notes = [f"{len(texts):,} text messages to work from"]

    xs, ys = sentiment_examples(texts)
    n_pos, n_neg = sum(ys), len(ys) - sum(ys)
    notes.append(f"Distant supervision found {len(xs):,} emoji-labelled messages "
                 f"({n_pos:,} positive, {n_neg:,} negative)")

    if len(xs) < min_examples or n_pos < 10 or n_neg < 10:
        return TrainResult(notes=notes, error=(
            f"Not enough emoji-labelled messages: found {len(xs):,} "
            f"({n_pos:,} positive, {n_neg:,} negative), and training needs at "
            f"least {min_examples} with 10 of each. This chat does not use "
            "enough emoji for distant supervision — everything else in the app "
            "works without a sentiment model."
        ))

    pairs, cut = _subsample(list(zip(xs, ys)), max_examples, seed)
    if cut:
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        n_pos, n_neg = sum(ys), len(ys) - sum(ys)
        notes.append(f"Sampled {len(xs):,} of them to keep training quick")

    y = np.array(ys)
    X_tr, X_te, y_tr, y_te = train_test_split(
        xs, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2,
            max_df=0.9, strip_accents="unicode", lowercase=True,
        )),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, class_weight="balanced"), cv=3,
        )),
    ])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    baseline = max(n_pos, n_neg) / len(y)

    notes += [
        f"Trained on {len(X_tr):,} messages, tested on {len(X_te):,}",
        f"Held-out accuracy : {acc:.3f}",
        f"Majority baseline : {baseline:.3f}",
        f"{'BEATS' if acc > baseline else 'DOES NOT BEAT'} the baseline "
        f"by {acc - baseline:+.3f}",
        "",
        classification_report(y_te, pred, target_names=["negative", "positive"],
                              zero_division=0),
    ]

    metrics = {
        "accuracy": acc, "baseline": float(baseline),
        "n_train": len(X_tr), "n_test": len(X_te),
        "n_positive": int(n_pos), "n_negative": int(n_neg),
    }
    return TrainResult(
        model=M.SentimentModel(pipe, metrics),
        payload={"pipeline": pipe, "metrics": metrics},
        notes=notes,
    )


# --- authorship -------------------------------------------------------------

def train_author(pairs: list[tuple[str, str]], *, min_messages: int = 40,
                 min_words: int = 3, test_size: float = 0.25, seed: int = 0,
                 max_per_person: int | None = None) -> TrainResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    notes = []
    pairs = [(t, s) for t, s in pairs if len(t.split()) >= min_words]
    counts = Counter(s for _, s in pairs)
    keep = {s for s, c in counts.items() if c >= min_messages}

    dropped = sorted(set(counts) - keep)
    if dropped:
        notes.append(f"Skipping {len(dropped)} people with fewer than "
                     f"{min_messages} messages of {min_words}+ words: "
                     f"{', '.join(dropped[:6])}{' ...' if len(dropped) > 6 else ''}")

    if len(keep) < 2:
        return TrainResult(notes=notes, error=(
            f"Only {len(keep)} person in this chat has {min_messages}+ messages "
            f"of at least {min_words} words, and telling people apart needs at "
            "least two. Try a longer export."
        ))

    # Cap per person rather than overall: it keeps the classes balanced and
    # stops one prolific member from dominating the character n-grams.
    kept: list[tuple[str, str]] = []
    capped = False
    for person in sorted(keep):
        theirs = [(t, s) for t, s in pairs if s == person]
        theirs, cut = _subsample(theirs, max_per_person, seed)
        capped |= cut
        kept += theirs
    if capped:
        notes.append(f"Sampled at most {max_per_person:,} messages per person "
                     "to keep training quick")

    X = [t for t, _ in kept]
    y = np.array([s for _, s in kept])
    labels = sorted(keep)
    notes.append(f"Training on {len(X):,} messages from {len(labels)} people")
    notes += [f"  {s:<20} {sum(1 for v in y if v == s):>6,}"
              for s in sorted(labels, key=lambda k: -counts[k])]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=2, max_features=60000, lowercase=True,
        )),
        ("clf", LinearSVC(C=1.0, class_weight="balanced")),
    ])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    baseline = max(Counter(y_te).values()) / len(y_te)
    chance = 1 / len(labels)
    cm = confusion_matrix(y_te, pred, labels=labels)

    notes += [
        f"Held-out accuracy  : {acc:.3f}",
        f"Majority baseline  : {baseline:.3f}",
        f"Random chance      : {chance:.3f}",
        f"{'BEATS' if acc > baseline else 'DOES NOT BEAT'} the majority "
        f"baseline by {acc - baseline:+.3f}",
        "",
        classification_report(y_te, pred, zero_division=0),
    ]

    # Who gets mistaken for whom — the genuinely interesting result.
    conf = [(labels[i], labels[j], int(cm[i, j]))
            for i in range(len(labels)) for j in range(len(labels))
            if i != j and cm[i, j] > 0]
    notes.append("Most common confusions:")
    notes += [f"  {a} mistaken for {b}: {n}"
              for a, b, n in sorted(conf, key=lambda x: -x[2])[:5]] or ["  (none)"]

    metrics = {
        "accuracy": acc, "baseline": float(baseline), "chance": float(chance),
        "confusion_matrix": cm.tolist(),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    return TrainResult(
        model=M.AuthorModel(pipe, labels, metrics),
        payload={"pipeline": pipe, "labels": labels, "metrics": metrics},
        notes=notes,
    )


# --- topics -----------------------------------------------------------------

def topic_documents(texts: list[str], stops: set[str], block: int = BLOCK) -> list[str]:
    """Group consecutive messages into blocks.

    Chat messages are far too short to model individually — a topic is a
    stretch of conversation, not a single "ok".
    """
    docs = [
        " ".join(" ".join(T.tokenize(t, stops)) for t in texts[i:i + block])
        for i in range(0, len(texts), block)
    ]
    return [d for d in docs if len(d.split()) >= 5]


def train_topics(texts: list[str], *, k: int = 8, block: int = BLOCK,
                 n_top_words: int = 12, stops: set[str] | None = None,
                 seed: int = 0, max_docs: int | None = None) -> TrainResult:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    notes = []
    if len(texts) < block * 4:
        return TrainResult(notes=notes, error=(
            f"Only {len(texts):,} text messages — too few to find topics. "
            f"Topic modelling needs at least {block * 4}."
        ))

    docs = topic_documents(texts, stops or T.stopwords(), block)
    notes.append(f"Built {len(docs):,} conversation blocks of {block} messages "
                 f"from {len(texts):,} messages")

    if max_docs is not None and len(docs) > max_docs:
        # Even stride rather than a random sample, so the blocks kept still
        # span the whole history rather than clumping.
        step = len(docs) / max_docs
        docs = [docs[int(i * step)] for i in range(max_docs)]
        notes.append(f"Sampled {len(docs):,} blocks evenly across the chat "
                     "to keep training quick")

    if len(docs) < 6:
        return TrainResult(notes=notes, error=(
            f"Only {len(docs)} conversation blocks had enough words to model. "
            "This chat is too short, or too much of it is media and stopwords."
        ))

    k_used = int(min(k, max(2, len(docs) // 3)))
    if k_used != k:
        notes.append(f"Reduced to k={k_used} topics — not enough blocks for {k}.")

    vec = TfidfVectorizer(max_df=0.85, min_df=2, sublinear_tf=True,
                          strip_accents="unicode", max_features=8000)
    try:
        X = vec.fit_transform(docs)
    except ValueError as e:
        return TrainResult(notes=notes, error=f"Vectorising failed: {e}")

    if X.shape[1] < k_used:
        return TrainResult(notes=notes, error=(
            f"Vocabulary too small ({X.shape[1]} terms) for {k_used} topics. "
            "Try a longer chat or fewer topics."
        ))

    notes.append(f"TF-IDF matrix: {X.shape[0]:,} blocks x {X.shape[1]:,} terms")

    nmf = NMF(n_components=k_used, random_state=seed, init="nndsvda",
              max_iter=400, l1_ratio=0.5, alpha_W=0.0)
    W = nmf.fit_transform(X)

    terms = np.array(vec.get_feature_names_out())
    top_words, top_weights = [], []
    for comp in nmf.components_:
        order = comp.argsort()[::-1][:n_top_words]
        top_words.append(terms[order].tolist())
        top_weights.append(comp[order].tolist())

    sizes = np.bincount(W.argmax(axis=1), minlength=k_used)
    notes.append(f"Reconstruction error: {nmf.reconstruction_err_:.3f}")
    for i in range(k_used):
        notes.append(f"Topic {i + 1} ({sizes[i]} blocks, "
                     f"{100 * sizes[i] / len(docs):.0f}%): "
                     f"{', '.join(top_words[i][:8])}")

    metrics = {
        "k": k_used, "n_docs": len(docs), "n_terms": int(X.shape[1]),
        "reconstruction_err": float(nmf.reconstruction_err_),
        "topic_sizes": sizes.tolist(), "block": block,
    }
    return TrainResult(
        model=M.TopicModel(vec, nmf, top_words, top_weights, metrics),
        payload={"vectorizer": vec, "nmf": nmf, "top_words": top_words,
                 "top_weights": top_weights, "metrics": metrics},
        notes=notes,
    )
