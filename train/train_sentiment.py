#!/usr/bin/env python3
"""Train a sentiment classifier on your own chat, with no labelled data.

The trick is **distant supervision**. Messages containing clearly positive
emoji are treated as positive examples and messages with clearly negative emoji
as negative ones. The emoji are then stripped from the text, so the classifier
is forced to learn sentiment from the *words* — and can go on to label the ~90%
of messages that carry no emoji at all.

This is the emoticon-supervision approach from Go, Bhayani & Huang (2009). It
needs no external corpus and it learns your group's own voice.

    python train/train_sentiment.py --chat data/sample_chat.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import parser as P               # noqa: E402
from src import text as T                 # noqa: E402

POSITIVE_EMOJI = set("😀😃😄😁😆😊🙂😍🥰😘🤗🤩😎👍👌🙏❤️💕💖💯🎉🥳😂🤣✨🔥💪👏🙌😇🥹")
NEGATIVE_EMOJI = set("😞😔😟😕🙁☹️😣😖😫😩😢😭😤😠😡🤬😰😨😱😥💔👎🤢🤮😒🙄😬😐")


def label_by_emoji(text: str) -> int | None:
    """1 positive, 0 negative, None if the message gives no clear signal."""
    found = set(T.extract_emoji(text))
    pos = len(found & POSITIVE_EMOJI)
    neg = len(found & NEGATIVE_EMOJI)
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, default=ROOT / "data" / "sample_chat.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sentiment.joblib")
    ap.add_argument("--min-examples", type=int, default=60)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    import joblib

    if not args.chat.exists():
        print(f"No chat file at {args.chat}", file=sys.stderr)
        return 1

    messages, report = P.parse_file(args.chat)
    print(f"Parsed: {report.summary()}")
    if not report.ok:
        print("Nothing parsed — is this a WhatsApp export?", file=sys.stderr)
        return 1

    texts, labels = [], []
    for m in messages:
        if not m.is_user or m.kind != P.KIND_TEXT:
            continue
        y = label_by_emoji(m.text)
        if y is None:
            continue
        stripped = T.strip_emoji(m.text).strip()
        if len(stripped.split()) < 2:   # nothing left to learn from
            continue
        texts.append(stripped)
        labels.append(y)

    y = np.array(labels)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"\nDistant supervision found {len(texts):,} labelled messages "
          f"({n_pos:,} positive, {n_neg:,} negative)")

    if len(texts) < args.min_examples or n_pos < 10 or n_neg < 10:
        print(
            f"\nNot enough emoji-labelled messages (need >= {args.min_examples} "
            "with at least 10 of each class).\n"
            "This chat does not use enough emoji for distant supervision. "
            "The rest of the app works fine without a sentiment model.",
            file=sys.stderr,
        )
        return 2

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=y
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
    acc = accuracy_score(y_te, pred)
    baseline = max(n_pos, n_neg) / len(y)

    print(f"\nHeld-out accuracy : {acc:.3f}")
    print(f"Majority baseline : {baseline:.3f}")
    print(f"{'BEATS' if acc > baseline else 'DOES NOT BEAT'} the baseline "
          f"by {acc - baseline:+.3f}")
    print("\n" + classification_report(y_te, pred, target_names=["negative", "positive"],
                                       zero_division=0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "metrics": {
            "accuracy": float(acc), "baseline": float(baseline),
            "n_train": len(X_tr), "n_test": len(X_te),
            "n_positive": n_pos, "n_negative": n_neg,
        },
    }, args.out)
    print(f"Saved {args.out}")

    print("\nCaveat: emoji-labelled messages are not a random sample of the chat — "
          "people reach for emoji in emotive messages. Treat this as a useful "
          "signal, not ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
