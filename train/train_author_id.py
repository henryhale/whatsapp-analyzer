#!/usr/bin/env python3
"""Train a "who wrote this?" classifier on your group chat.

Character n-grams rather than words, because authorship shows up in habits of
spelling, punctuation and abbreviation more than in vocabulary — "u" vs "you",
whether someone types "..." or "!!", how they spell "ok".

The confusion matrix is the interesting output: it shows who writes like whom.

    python train/train_author_id.py --chat data/sample_chat.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import parser as P               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, default=ROOT / "data" / "sample_chat.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "author_id.joblib")
    ap.add_argument("--min-messages", type=int, default=40,
                    help="drop people with fewer messages than this")
    ap.add_argument("--min-words", type=int, default=3,
                    help="drop very short messages — 'ok' identifies nobody")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    import joblib

    if not args.chat.exists():
        print(f"No chat file at {args.chat}", file=sys.stderr)
        return 1

    messages, report = P.parse_file(args.chat)
    print(f"Parsed: {report.summary()}")

    pairs = [
        (m.text, m.sender) for m in messages
        if m.is_user and m.kind == P.KIND_TEXT
        and len(m.text.split()) >= args.min_words
    ]
    counts = Counter(s for _, s in pairs)
    keep = {s for s, c in counts.items() if c >= args.min_messages}

    dropped = sorted(set(counts) - keep)
    if dropped:
        print(f"Skipping {len(dropped)} people with < {args.min_messages} "
              f"messages: {', '.join(dropped[:6])}"
              f"{' ...' if len(dropped) > 6 else ''}")

    if len(keep) < 2:
        print(
            f"\nNeed at least 2 people with {args.min_messages}+ messages "
            f"(found {len(keep)}). Lower --min-messages or use a bigger chat.",
            file=sys.stderr,
        )
        return 2

    X = [t for t, s in pairs if s in keep]
    y = np.array([s for _, s in pairs if s in keep])
    labels = sorted(keep)
    print(f"\nTraining on {len(X):,} messages from {len(labels)} people")
    for s in sorted(keep, key=lambda k: -counts[k]):
        print(f"  {s:<20} {counts[s]:>6,}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
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
    acc = accuracy_score(y_te, pred)
    baseline = max(Counter(y_te).values()) / len(y_te)
    chance = 1 / len(labels)

    print(f"\nHeld-out accuracy  : {acc:.3f}")
    print(f"Majority baseline  : {baseline:.3f}")
    print(f"Random chance      : {chance:.3f}")
    verdict = "BEATS" if acc > baseline else "DOES NOT BEAT"
    print(f"{verdict} the majority baseline by {acc - baseline:+.3f}")
    print("\n" + classification_report(y_te, pred, zero_division=0))

    cm = confusion_matrix(y_te, pred, labels=labels)

    # Who gets mistaken for whom — the genuinely interesting result.
    print("Most common confusions:")
    conf = [
        (labels[i], labels[j], cm[i, j])
        for i in range(len(labels)) for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    for a, b, n in sorted(conf, key=lambda x: -x[2])[:5]:
        print(f"  {a} mistaken for {b}: {n}")
    if not conf:
        print("  (none)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "labels": labels,
        "metrics": {
            "accuracy": float(acc), "baseline": float(baseline),
            "chance": float(chance), "confusion_matrix": cm.tolist(),
            "n_train": len(X_tr), "n_test": len(X_te),
        },
    }, args.out)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
