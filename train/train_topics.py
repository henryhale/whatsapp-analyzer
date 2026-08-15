#!/usr/bin/env python3
"""Discover the major topics in a group chat with TF-IDF + NMF.

NMF rather than LDA: on short, messy chat text it produces cleaner, more
readable topics and it is far quicker to fit on a CPU.

Chat messages are too short to model individually, so consecutive messages are
grouped into short conversation blocks first — a topic is a stretch of
conversation, not a single "ok".

    python train/train_topics.py --chat data/sample_chat.txt --k 8

The fit itself lives in `src/training.py`, which the Streamlit app calls too —
this script is the offline path, useful for committing a model or for training
on a chat you would rather not upload anywhere. In the app, the Topics tab
trains the same model on the loaded chat at the press of a button.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import parser as P                # noqa: E402
from src import text as T                  # noqa: E402
from src import training as TR             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, default=ROOT / "data" / "sample_chat.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "topics.joblib")
    ap.add_argument("--k", type=int, default=8, help="number of topics")
    ap.add_argument("--top-words", type=int, default=12)
    ap.add_argument("--block", type=int, default=TR.BLOCK,
                    help="messages grouped into one pseudo-document")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="cap the conversation blocks used (default: all)")
    ap.add_argument("--local-stopwords", action="store_true",
                    help="also filter common Luganda/Swahili function words")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import joblib

    if not args.chat.exists():
        print(f"No chat file at {args.chat}", file=sys.stderr)
        return 1

    messages, report = P.parse_file(args.chat)
    print(f"Parsed: {report.summary()}")

    res = TR.train_topics(
        TR.texts_from_messages(messages),
        k=args.k, block=args.block, n_top_words=args.top_words,
        stops=T.stopwords(include_local=args.local_stopwords),
        max_docs=args.max_docs, seed=args.seed,
    )
    print("\n" + res.log())
    if not res.ok:
        print(f"\n{res.error}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res.payload, args.out)
    print(f"\nSaved {args.out}")
    print("Topic labels get generated in the app, where the language model can "
          "turn these word lists into readable names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
