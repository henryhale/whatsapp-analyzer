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

The fit itself lives in `src/training.py`, which the Streamlit app calls too —
this script is the offline path, useful for committing a model or for training
on a chat you would rather not upload anywhere. In the app, the Sentiment tab
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

# Re-exported for anyone importing this script directly.
POSITIVE_EMOJI = TR.POSITIVE_EMOJI
NEGATIVE_EMOJI = TR.NEGATIVE_EMOJI
label_by_emoji = TR.label_by_emoji


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, default=ROOT / "data" / "sample_chat.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sentiment.joblib")
    ap.add_argument("--min-examples", type=int, default=60)
    ap.add_argument("--max-examples", type=int, default=None,
                    help="cap the labelled examples used (default: all)")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import joblib

    if not args.chat.exists():
        print(f"No chat file at {args.chat}", file=sys.stderr)
        return 1

    messages, report = P.parse_file(args.chat)
    print(f"Parsed: {report.summary()}")
    if not report.ok:
        print("Nothing parsed — is this a WhatsApp export?", file=sys.stderr)
        return 1

    res = TR.train_sentiment(
        TR.texts_from_messages(messages),
        min_examples=args.min_examples, max_examples=args.max_examples,
        test_size=args.test_size, seed=args.seed,
    )
    print("\n" + res.log())
    if not res.ok:
        print(f"\n{res.error}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res.payload, args.out)
    print(f"\nSaved {args.out}")
    print("\nCaveat: emoji-labelled messages are not a random sample of the chat — "
          "people reach for emoji in emotive messages. Treat this as a useful "
          "signal, not ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
