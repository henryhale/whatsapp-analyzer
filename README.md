# WhatsApp Group Chat Analyzer

Statistics, topics, sentiment, authorship, and a small local LLM you can ask
questions about your own group chat.

**Runs without PyTorch**, which is what makes it deployable to Streamlit
Community Cloud. Training is scikit-learn; the language model runs on ONNX
Runtime.

## What it does

**Overview** — messages, people, active days, words, busiest day, longest
streak, longest silence. A most-active ranking (by messages, words, or average
message length) plus bar charts by hour, weekday and month, and an activity
timeline with a 7-day rolling average.

**People** — who talks when, reply-speed distributions, who starts
conversations after a silence, and top words and emoji per person.

**Summary** — key messages extracted verbatim (no model needed, cannot
hallucinate), plus an optional written summary via map-reduce over the chat.

**Topics** — NMF topic model over conversation blocks, optionally given
readable names by the language model.

**Sentiment** — trained on *your* chat with no labelled data (see below).

**Authorship** — a "who wrote this?" classifier, with a confusion matrix
showing who writes like whom, and a box to test your own sentence.

**Chat with your chats** — ask a question, get an answer grounded in retrieved
conversation windows, with every source shown.

## Quick start

```bash
streamlit run app.py          # works immediately with the bundled sample chat
```

Then train the three models to light up the remaining tabs:

```bash
python train/train_sentiment.py  --chat data/sample_chat.txt
python train/train_author_id.py  --chat data/sample_chat.txt
python train/train_topics.py     --chat data/sample_chat.txt --k 8
```

Point `--chat` at your own export to train on your real group. Each script
prints held-out accuracy against a stated baseline, so you can see whether the
model actually learned anything without opening the app.

### Getting your export

WhatsApp → open the chat → **⋮ → More → Export chat → Without media**. Both
iPhone and Android exports work, in any date locale. Upload the `.txt` or
`.zip` directly in the app.

## The three models

**Sentiment, without labelled data.** Messages containing clearly positive
emoji become positive examples, clearly negative emoji negative ones. The emoji
are then stripped, forcing the classifier to learn sentiment from words — so it
can label the ~90% of messages that carry no emoji. This is distant supervision
(Go, Bhayani & Huang 2009). *Caveat, stated in the app too: emoji-bearing
messages are not a random sample of a chat.*

**Authorship.** Character 3–5 grams rather than words, because writing identity
lives in spelling and punctuation habits — "u" vs "you", "..." vs "!!" — more
than in vocabulary.

**Topics.** TF-IDF + NMF over blocks of 8 consecutive messages. Individual chat
messages are far too short to model on their own; a topic is a stretch of
conversation.

## The language model

Optional, off by default. `SmolLM2-360M-Instruct` via ONNX Runtime, preferring
quantized weights (~360 MB rather than ~1.4 GB at fp32), with a 135M fallback
and a retrieval-only mode below that. Enable it in the sidebar; weights
download on first use and are cached.

**Be realistic about a 360M model.** It answers direct questions from retrieved
context. It does not reason. That is why retrieval does the real work, why
sources are always displayed, and why every feature stays useful with the model
switched off. On a 2-core CPU expect roughly 5–15 tokens/sec.

## Privacy

A group chat contains other people's messages and phone numbers. Exports are
parsed **in memory** and never uploaded anywhere; `data/.gitignore` keeps real
exports out of version control. Think carefully before pointing a public
Streamlit Cloud deployment at a real chat — the sample chat is synthetic and
safe for demos.

## Layout

```
src/parser.py      export parsing — two dialects, locale date inference,
                   invisible-character normalisation, multi-line messages
src/stats.py       descriptive statistics
src/text.py        tokenizing, stopwords (incl. optional Luganda/Swahili), emoji
src/summarize.py   extractive (always) and abstractive (optional) summaries
src/rag.py         conversation windows, TF-IDF retrieval, grounded prompts
src/llm.py         ONNX language model with graceful tiered fallback
src/models.py      loading trained artifacts
src/viz.py         Plotly charts
train/             three training scripts
data/sample_chat.txt  synthetic fixture, safe to share
```
