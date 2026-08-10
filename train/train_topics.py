#!/usr/bin/env python3
"""Discover the major topics in a group chat with TF-IDF + NMF.

NMF rather than LDA: on short, messy chat text it produces cleaner, more
readable topics and it is far quicker to fit on a CPU.

Chat messages are too short to model individually, so consecutive messages are
grouped into short conversation blocks first — a topic is a stretch of
conversation, not a single "ok".

    python train/train_topics.py --chat data/sample_chat.txt --k 8
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

BLOCK = 8   # messages per pseudo-document


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, default=ROOT / "data" / "sample_chat.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "topics.joblib")
    ap.add_argument("--k", type=int, default=8, help="number of topics")
    ap.add_argument("--top-words", type=int, default=12)
    ap.add_argument("--block", type=int, default=BLOCK,
                    help="messages grouped into one pseudo-document")
    ap.add_argument("--local-stopwords", action="store_true",
                    help="also filter common Luganda/Swahili function words")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    if not args.chat.exists():
        print(f"No chat file at {args.chat}", file=sys.stderr)
        return 1

    messages, report = P.parse_file(args.chat)
    print(f"Parsed: {report.summary()}")

    texts = [m.text for m in messages if m.is_user and m.kind == P.KIND_TEXT]
    if len(texts) < args.block * 4:
        print(f"Only {len(texts)} text messages — too few for topic modelling.",
              file=sys.stderr)
        return 2

    stops = T.stopwords(include_local=args.local_stopwords)
    docs = [
        " ".join(" ".join(T.tokenize(t, stops)) for t in texts[i:i + args.block])
        for i in range(0, len(texts), args.block)
    ]
    docs = [d for d in docs if len(d.split()) >= 5]
    print(f"Built {len(docs):,} conversation blocks from {len(texts):,} messages")

    k = int(min(args.k, max(2, len(docs) // 3)))
    if k != args.k:
        print(f"Reducing to k={k} topics — not enough blocks for {args.k}.")

    vec = TfidfVectorizer(max_df=0.85, min_df=2, sublinear_tf=True,
                          strip_accents="unicode", max_features=8000)
    try:
        X = vec.fit_transform(docs)
    except ValueError as e:
        print(f"Vectorising failed: {e}", file=sys.stderr)
        return 2

    if X.shape[1] < k:
        print(f"Vocabulary too small ({X.shape[1]} terms) for {k} topics.",
              file=sys.stderr)
        return 2

    print(f"TF-IDF matrix: {X.shape[0]:,} blocks x {X.shape[1]:,} terms")
    print(f"Fitting NMF with k={k}...")

    nmf = NMF(n_components=k, random_state=args.seed, init="nndsvda",
              max_iter=400, l1_ratio=0.5, alpha_W=0.0)
    W = nmf.fit_transform(X)

    terms = np.array(vec.get_feature_names_out())
    top_words, top_weights = [], []
    for i, comp in enumerate(nmf.components_):
        order = comp.argsort()[::-1][:args.top_words]
        top_words.append(terms[order].tolist())
        top_weights.append(comp[order].tolist())

    sizes = np.bincount(W.argmax(axis=1), minlength=k)
    print(f"\nReconstruction error: {nmf.reconstruction_err_:.3f}\n")
    for i in range(k):
        share = 100 * sizes[i] / len(docs)
        print(f"Topic {i + 1}  ({sizes[i]} blocks, {share:.0f}%)")
        print(f"  {', '.join(top_words[i][:8])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "vectorizer": vec, "nmf": nmf,
        "top_words": top_words, "top_weights": top_weights,
        "metrics": {
            "k": k, "n_docs": len(docs), "n_terms": int(X.shape[1]),
            "reconstruction_err": float(nmf.reconstruction_err_),
            "topic_sizes": sizes.tolist(), "block": args.block,
        },
    }, args.out)
    print(f"\nSaved {args.out}")
    print("Topic labels get generated in the app, where the language model can "
          "turn these word lists into readable names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
