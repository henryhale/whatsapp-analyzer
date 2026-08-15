#!/usr/bin/env python3
"""Train a "who wrote this?" classifier on your group chat.

Character n-grams rather than words, because authorship shows up in habits of
spelling, punctuation and abbreviation more than in vocabulary — "u" vs "you",
whether someone types "..." or "!!", how they spell "ok".

The confusion matrix is the interesting output: it shows who writes like whom.

    python train/train_author_id.py --chat data/sample_chat.txt

The fit itself lives in `src/training.py`, which the Streamlit app calls too —
this script is the offline path, useful for committing a model or for training
on a chat you would rather not upload anywhere. In the app, the Authorship tab
trains the same model on the loaded chat at the press of a button.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import parser as P                # noqa: E402
from src import training as TR             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, default=ROOT / "data" / "sample_chat.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "author_id.joblib")
    ap.add_argument("--min-messages", type=int, default=40,
                    help="drop people with fewer messages than this")
    ap.add_argument("--min-words", type=int, default=3,
                    help="drop very short messages — 'ok' identifies nobody")
    ap.add_argument("--max-per-person", type=int, default=None,
                    help="cap the messages used per person (default: all)")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import joblib

    if not args.chat.exists():
        print(f"No chat file at {args.chat}", file=sys.stderr)
        return 1

    messages, report = P.parse_file(args.chat)
    print(f"Parsed: {report.summary()}")

    res = TR.train_author(
        TR.pairs_from_messages(messages),
        min_messages=args.min_messages, min_words=args.min_words,
        max_per_person=args.max_per_person, test_size=args.test_size,
        seed=args.seed,
    )
    print("\n" + res.log())
    if not res.ok:
        print(f"\n{res.error}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res.payload, args.out)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
