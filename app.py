#!/usr/bin/env python3
"""WhatsApp Group Chat Analyzer.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import llm as llm_mod
from src import models as models_mod
from src import parser as P
from src import rag as rag_mod
from src import stats as stats_mod
from src import summarize as summ_mod
from src import text as text_mod
from src import theme as theme_mod
from src import training as train_mod
from src import viz

SAMPLE = ROOT / "data" / "sample_chat.txt"

# Streamlit Community Cloud allows about a gigabyte of RAM, and the models are
# trained inside the app there, so training subsamples very large chats. The
# train/ scripts run uncapped by default.
CAP_SENTIMENT_EXAMPLES = 20_000
CAP_AUTHOR_PER_PERSON = 4_000
CAP_TOPIC_DOCS = 6_000

st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="wide")


# --- data loading -----------------------------------------------------------

@st.cache_data(show_spinner=False)
def _parse_bytes(raw: bytes, name: str):
    """Parse an uploaded export. Cached on content, so reruns are instant."""
    if name.lower().endswith(".zip"):
        import io, zipfile

        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                return None, None
            raw = z.read(names[0])
    messages, report = P.parse_text(raw.decode("utf-8", errors="replace"))
    return P.to_dataframe(messages), report


@st.cache_resource(show_spinner=False)
def _retriever(_df: pd.DataFrame, key: str) -> rag_mod.Retriever:
    return rag_mod.Retriever(rag_mod.build_windows(_df))


@st.cache_resource(show_spinner="Loading the language model (first run downloads it)...")
def _get_llm(model_id: str):
    m = llm_mod.LocalLLM(model_id)
    status = m.load()
    return (m, status) if status.available else (None, status)


def _fmt_minutes(m: float) -> str:
    if m < 1:
        return f"{m * 60:.0f}s"
    if m < 60:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def _chat_key(df: pd.DataFrame) -> str:
    """Identifies the loaded chat, so caches and buttons reset on a new upload."""
    return f"{len(df)}-{df['timestamp'].min()}-{df['timestamp'].max()}"


# --- training on demand -----------------------------------------------------
#
# The three sklearn models are small enough to fit inside a request, so the app
# trains them itself on the chat that is loaded rather than asking anyone to run
# a script. Results are cached per chat, so a fit happens once per session and
# every later rerun is instant. `train/*.py` remains the offline path and calls
# the same functions in src/training.py.

@st.cache_resource(show_spinner=False, max_entries=2)
def _train_topics(key: str, _df: pd.DataFrame, k: int, local_stops: bool):
    return train_mod.train_topics(
        train_mod.texts_from_dataframe(_df), k=k, max_docs=CAP_TOPIC_DOCS,
        stops=text_mod.stopwords(include_local=local_stops),
    )


@st.cache_resource(show_spinner=False, max_entries=2)
def _train_sentiment(key: str, _df: pd.DataFrame):
    return train_mod.train_sentiment(
        train_mod.texts_from_dataframe(_df), max_examples=CAP_SENTIMENT_EXAMPLES,
    )


@st.cache_resource(show_spinner=False, max_entries=2)
def _train_author(key: str, _df: pd.DataFrame):
    return train_mod.train_author(
        train_mod.pairs_from_dataframe(_df), max_per_person=CAP_AUTHOR_PER_PERSON,
    )


@st.cache_resource(show_spinner=False)
def _from_disk(kind: str):
    """Whatever the train/ scripts left in models/, if anything."""
    return {
        "topics": models_mod.load_topics,
        "sentiment": models_mod.load_sentiment,
        "author": models_mod.load_author,
    }[kind]()


@st.cache_resource(show_spinner=False, max_entries=2)
def _score(key: str, source: str, _sm, _texts: list[str]):
    """Sentiment for every message — cached, since every tab redraws on a rerun."""
    return _sm.predict(_texts)


def _flag(kind: str, key: str) -> str:
    return f"trained::{kind}::{key}"


def _training_log(res) -> None:
    if res is not None and res.notes:
        with st.expander("Training log"):
            st.code(res.log())


def _model_ui(kind: str, key: str, label: str, train_call, blurb: str):
    """Resolve the model a tab needs, training it on a button press.

    Returns (model, result); result is the training outcome when a fit ran in
    this session, otherwise None. A model left in models/ by the training
    scripts is used as a fallback, with the option to refit on this chat.
    """
    flag = _flag(kind, key)

    if st.session_state.get(flag):
        with st.spinner(f"Training the {label} model on this chat — "
                        "this takes a few seconds..."):
            res = train_call()
        st.session_state[f"status::{kind}::{key}"] = "ok" if res.ok else "failed"
        if res.ok:
            return res.model, res
        st.error(res.error)
        _training_log(res)
        return None, res

    disk = _from_disk(kind)
    if disk is not None:
        c1, c2 = st.columns([4, 1])
        c1.caption("Loaded from `models/`, trained offline — possibly on a "
                   "different chat.")
        if c2.button("Retrain here", key=f"retrain::{kind}::{key}",
                     use_container_width=True):
            st.session_state[flag] = True
            st.rerun()
        return disk, None

    st.info(blurb)
    if st.button(f"Train the {label} model", type="primary",
                 key=f"train::{kind}::{key}"):
        st.session_state[flag] = True
        st.rerun()
    return None, None


# --- sidebar ----------------------------------------------------------------

def sidebar():
    st.sidebar.title("💬 Chat Analyzer")
    st.sidebar.caption(
        "Your chat is parsed in memory and never uploaded anywhere."
    )

    up = st.sidebar.file_uploader(
        "WhatsApp export (.txt or .zip)", type=["txt", "zip"],
        help="In WhatsApp: open the chat → ⋮ → More → Export chat → Without media",
    )
    use_sample = st.sidebar.checkbox("Use the sample chat", value=up is None,
                                     disabled=up is not None)

    df = report = None
    if up is not None:
        df, report = _parse_bytes(up.getvalue(), up.name)
    elif use_sample and SAMPLE.exists():
        df, report = _parse_bytes(SAMPLE.read_bytes(), SAMPLE.name)

    st.sidebar.divider()
    st.sidebar.subheader("Language model")
    st.sidebar.caption(
        "Optional. Powers written summaries, topic names, and chat answers. "
        "Everything else works without it."
    )
    enable = st.sidebar.toggle("Enable generation", value=False)
    model_id = st.sidebar.selectbox(
        "Model", [m for m, _, _ in llm_mod.TIERS],
        format_func=lambda m: next(f"{l} ({s})" for x, l, s in llm_mod.TIERS if x == m),
        disabled=not enable,
    )

    llm = None
    if enable:
        llm, status = _get_llm(model_id)
        if status.available:
            st.sidebar.success(f"{status.label} · {status.detail}")
        else:
            st.sidebar.warning(status.error or "Model unavailable — retrieval only.")
    else:
        st.sidebar.info("Retrieval only")

    st.sidebar.divider()
    st.sidebar.subheader("Options")
    local_stops = st.sidebar.checkbox(
        "Filter Luganda/Swahili filler words", value=False,
        help="Useful for code-switched chats, where 'nga', 'kale', 'sawa' would "
             "otherwise dominate the top-words chart.",
    )

    return df, report, llm, local_stops


MODEL_KINDS = [("topics", "Topics"), ("sentiment", "Sentiment"),
               ("author", "Authorship")]


def sidebar_models(key: str) -> None:
    """Model status and a one-click 'train everything' button."""
    st.sidebar.divider()
    st.sidebar.subheader("Models")
    st.sidebar.caption(
        "Topics, sentiment and authorship are trained here, in this session, on "
        "the chat you loaded — nothing is saved and nothing is sent anywhere."
    )

    if st.sidebar.button("Train all three", use_container_width=True):
        for kind, _ in MODEL_KINDS:
            st.session_state[_flag(kind, key)] = True
        st.rerun()

    marks = {"ok": "✓", "failed": "✗", "pending": "…", "disk": "◆", "none": "·"}
    bits = []
    for kind, nice in MODEL_KINDS:
        status = st.session_state.get(f"status::{kind}::{key}")
        if status is None:
            if st.session_state.get(_flag(kind, key)):
                status = "pending"
            else:
                status = "disk" if _from_disk(kind) is not None else "none"
        bits.append(f"{marks[status]} {nice}")
    st.sidebar.caption(" · ".join(bits))


# --- tabs -------------------------------------------------------------------

def tab_overview(df, th, stops):
    h = stats_mod.headline(df)
    if not h:
        st.warning("No user messages found.")
        return

    c = st.columns(4)
    c[0].metric("Messages", f"{h['messages']:,}")
    c[1].metric("People", h["participants"])
    c[2].metric("Active days", f"{h['active_days']:,}",
                help=f"out of {h['days_span']:,} days spanned")
    c[3].metric("Words", f"{h['total_words']:,}")

    c = st.columns(4)
    c[0].metric("Busiest day", f"{h['busiest_day_count']:,}",
                help=str(h["busiest_day"]))
    c[1].metric("Avg / active day", f"{h['avg_per_active_day']:.0f}")
    c[2].metric("Media", f"{h['media']:,}")
    c[3].metric("Deleted", f"{h['deleted']:,}")

    s = stats_mod.streaks(df)
    st.caption(
        f"Longest daily streak: **{s['longest_streak_days']} days** "
        f"({s['streak_start']} → {s['streak_end']}) · "
        f"longest silence: **{h['longest_silence_hours']:.0f} hours** · "
        f"from {h['first']:%d %b %Y} to {h['last']:%d %b %Y}"
    )

    st.divider()
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Most active")
        pp = stats_mod.per_person(df, stops)
        metric = st.radio("Rank by", ["messages", "words", "avg_words"],
                          horizontal=True, label_visibility="collapsed",
                          format_func=lambda m: {"messages": "Messages",
                                                 "words": "Words",
                                                 "avg_words": "Avg words"}[m])
        st.plotly_chart(viz.most_active(pp, th, metric),
                        use_container_width=True, config={"displaylogo": False})

    with right:
        st.subheader("Activity over time")
        st.plotly_chart(viz.timeline(stats_mod.by_day(df), th),
                        use_container_width=True, config={"displaylogo": False})

    a, b, c3 = st.columns(3)
    with a:
        st.markdown("**By hour**")
        st.plotly_chart(viz.bar(stats_mod.by_hour(df), "hour", "messages", th,
                                xtitle="Hour", height=260),
                        use_container_width=True, config={"displaylogo": False})
    with b:
        st.markdown("**By weekday**")
        st.plotly_chart(viz.bar(stats_mod.by_weekday(df), "weekday", "messages", th,
                                height=260, slot=1),
                        use_container_width=True, config={"displaylogo": False})
    with c3:
        st.markdown("**By month**")
        st.plotly_chart(viz.bar(stats_mod.by_month(df), "month", "messages", th,
                                height=260, slot=2),
                        use_container_width=True, config={"displaylogo": False})

    with st.expander("Per-person table"):
        st.dataframe(pp.round(2), use_container_width=True, hide_index=True)


def tab_people(df, th, stops):
    pp = stats_mod.per_person(df, stops)
    if pp.empty:
        st.warning("No participants found.")
        return

    st.subheader("Who talks when")
    st.plotly_chart(
        viz.stacked_over_time(stats_mod.activity_by_person_over_time(df), th),
        use_container_width=True, config={"displaylogo": False})

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Reply speed")
        samples = stats_mod.response_time_samples(df)
        st.plotly_chart(viz.response_distribution(samples, th),
                        use_container_width=True, config={"displaylogo": False})
        rt = stats_mod.response_times(df)
        if not rt.empty:
            rt = rt.assign(median=rt["median_minutes"].map(_fmt_minutes))
            st.dataframe(rt[["sender", "median", "replies"]],
                         use_container_width=True, hide_index=True)
    with right:
        st.subheader("Who starts conversations")
        cs = stats_mod.conversation_starters(df)
        if cs.empty:
            st.info("Not enough gaps in the chat to identify conversation starts.")
        else:
            st.plotly_chart(viz.bar(cs, "sender", "starts", th, ytitle="Conversations started",
                                    height=340, slot=1),
                            use_container_width=True, config={"displaylogo": False})
            st.caption("A message counts as a start when it follows a gap of "
                       "an hour or more.")

    st.subheader("Words and emoji")
    person = st.selectbox("Person", ["Everyone"] + pp["sender"].tolist())
    sub = df if person == "Everyone" else df[df["sender"] == person]
    texts = sub.loc[sub["kind"] == "text", "text"].tolist()

    a, b = st.columns(2, gap="large")
    with a:
        tw = text_mod.top_words(texts, n=15, stops=stops)
        if tw:
            d = pd.DataFrame(tw, columns=["word", "count"])
            st.plotly_chart(viz.most_active(d.rename(columns={"word": "sender",
                                                              "count": "messages"}),
                                            th, "messages", height=420),
                            use_container_width=True, config={"displaylogo": False})
        else:
            st.info("No words left after filtering.")
    with b:
        te = text_mod.top_emoji(texts, n=15)
        if te:
            st.dataframe(pd.DataFrame(te, columns=["emoji", "count"]),
                         use_container_width=True, hide_index=True, height=420)
        else:
            st.info("No emoji in these messages.")


def tab_summary(df, th, llm):
    st.subheader("Summary")

    u = df[df["is_user"]]
    dates = sorted(set(u["date"]))
    if not dates:
        st.warning("Nothing to summarise.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        rng = st.select_slider(
            "Date range", options=dates,
            value=(dates[0], dates[-1]),
        ) if len(dates) > 1 else (dates[0], dates[0])
    with c2:
        n = st.slider("Key messages", 3, 20, 8)

    sub = df[(df["date"] >= rng[0]) & (df["date"] <= rng[1])]
    st.caption(f"{int(sub['is_user'].sum()):,} messages in range")

    st.markdown("#### Key messages")
    st.caption("Chosen by similarity to the centre of the conversation, spread "
               "across the period. These are real messages, quoted verbatim.")
    st.markdown(summ_mod.format_extractive(summ_mod.extractive(sub, n=n)))

    st.divider()
    st.markdown("#### Written summary")
    if llm is None:
        st.info("Enable the language model in the sidebar to generate a written "
                "summary. The key messages above need no model.")
    elif st.button("Generate summary", type="primary"):
        with st.spinner("Generating — a small model on CPU takes a moment..."):
            text, note = summ_mod.abstractive(sub, llm)
        if note:
            st.caption(note)
        st.markdown(text or "_Nothing generated._")


def tab_topics(df, th, llm, key, local_stops):
    st.subheader("Topics")
    st.caption(
        "TF-IDF and NMF over blocks of consecutive messages, because a topic is "
        "a stretch of conversation rather than a single message."
    )
    k = st.slider("Topics to look for", 4, 16, 8, key=f"k::{key}",
                  help="Refits when you change it.")

    tm, res = _model_ui(
        "topics", key, "topic",
        lambda: _train_topics(key, df, k, local_stops),
        blurb="Group the conversation into topics — takes a few seconds and "
              "runs right here.",
    )
    if tm is None:
        return

    st.markdown(f"**{tm.n_topics} topics discovered** — NMF over "
                f"{tm.metrics.get('n_docs', 0):,} conversation blocks of "
                f"{tm.metrics.get('block', 8)} messages each.")
    _training_log(res)

    # Labels belong to one particular fit, so they are keyed by it — a refit
    # with a different k must not inherit the previous names.
    label_key = f"topic_labels::{key}::{tm.n_topics}"
    if llm is not None and st.button("Name the topics with the language model"):
        with st.spinner("Labelling..."):
            st.session_state[label_key] = [
                summ_mod.label_topic(w, llm) for w in tm.top_words
            ]
    labels = st.session_state.get(label_key)

    sizes = tm.metrics.get("topic_sizes", [])
    cols = st.columns(2)
    for i, words in enumerate(tm.top_words):
        with cols[i % 2]:
            name = labels[i] if labels and i < len(labels) else ", ".join(words[:4])
            share = f" · {sizes[i]} blocks" if i < len(sizes) else ""
            st.markdown(f"**Topic {i + 1}: {name}**{share}")
            st.plotly_chart(
                viz.topic_bar(words[:8], tm.top_weights[i][:8], th, height=220),
                use_container_width=True, config={"displaylogo": False})


def tab_sentiment(df, th, key):
    st.subheader("Sentiment")
    st.caption(
        "Messages with clearly positive or negative emoji become the training "
        "labels; the emoji are then stripped, so the classifier has to learn "
        "sentiment from words — and can score the messages that carry no emoji."
    )

    sm, res = _model_ui(
        "sentiment", key, "sentiment",
        lambda: _train_sentiment(key, df),
        blurb="Learn this group's own positive and negative language from its "
              "emoji — takes a few seconds and runs right here.",
    )
    if sm is None:
        return

    m = sm.metrics
    c = st.columns(3)
    c[0].metric("Held-out accuracy", f"{m.get('accuracy', 0):.1%}")
    c[1].metric("Majority baseline", f"{m.get('baseline', 0):.1%}")
    c[2].metric("Labelled examples", f"{m.get('n_train', 0) + m.get('n_test', 0):,}")
    _training_log(res)

    u = df[df["is_user"] & (df["kind"] == "text")].copy()
    texts = u["text"].tolist()
    if not texts:
        st.warning("No text messages.")
        return

    with st.spinner("Scoring messages..."):
        u["positive"] = _score(key, "session" if res else "disk", sm, texts)

    st.subheader("Positivity by person")
    byp = (u.groupby("sender")["positive"].agg(["mean", "size"])
             .reset_index().rename(columns={"mean": "positive_share", "size": "messages"}))
    byp = byp.sort_values("positive_share", ascending=False)
    st.plotly_chart(
        viz.most_active(byp.assign(messages=byp["positive_share"] * 100), th,
                        "messages", height=320),
        use_container_width=True, config={"displaylogo": False})
    st.caption("Percentage of each person's messages predicted positive.")

    st.subheader("Over time")
    per = (u.set_index("timestamp").groupby(pd.Grouper(freq="W"))["positive"]
             .mean().reset_index())
    per.columns = ["period", "positive_share"]
    st.plotly_chart(viz.sentiment_over_time(per.dropna(), th),
                    use_container_width=True, config={"displaylogo": False})

    st.warning(
        "Emoji-labelled messages are not a random sample — people use emoji in "
        "emotive messages. This is a useful signal, not ground truth."
    )


def tab_authorship(df, th, key):
    st.subheader("Authorship")
    st.caption(
        "Character n-grams rather than words, because who wrote something shows "
        "up in habits of spelling and punctuation — \"u\" vs \"you\", \"...\" vs "
        "\"!!\" — more than in vocabulary."
    )

    am, res = _model_ui(
        "author", key, "authorship",
        lambda: _train_author(key, df),
        blurb="Learn each person's writing style and see who writes like whom — "
              "takes a few seconds and runs right here.",
    )
    if am is None:
        return

    m = am.metrics
    c = st.columns(3)
    c[0].metric("Accuracy", f"{m.get('accuracy', 0):.1%}")
    c[1].metric("Majority baseline", f"{m.get('baseline', 0):.1%}")
    c[2].metric("Random chance", f"{m.get('chance', 0):.1%}")
    _training_log(res)

    cm = m.get("confusion_matrix")
    if cm:
        st.subheader("Who writes like whom")
        st.caption("Each row is what someone actually wrote; each column is who "
                   "the model guessed. Off-diagonal cells are people with "
                   "similar writing styles.")
        st.plotly_chart(viz.confusion_matrix(np.array(cm), am.labels, th),
                        use_container_width=True, config={"displaylogo": False})

    st.subheader("Try it")
    txt = st.text_area("Type a message and see who it sounds like",
                       placeholder="e.g. can we sort the fuel money before saturday")
    if txt.strip():
        st.success(f"Sounds most like **{am.predict([txt])[0]}**")


def tab_chat(df, th, llm, key):
    st.subheader("Chat with your chats")
    st.caption(
        "Finds the most relevant stretches of conversation, then answers from "
        "them. Sources are always shown so you can check the answer."
    )

    retriever = _retriever(df, key)
    if not retriever.windows:
        st.warning("Not enough text messages to search.")
        return

    st.caption(f"{len(retriever.windows):,} conversation windows indexed")

    if llm is None:
        st.info(
            "Generation is off — you'll get the matching messages, not a written "
            "answer. Enable the language model in the sidebar for prose."
        )

    q = st.text_input("Ask something",
                      placeholder="What did everyone decide about the trip?")
    k = st.slider("Passages to retrieve", 2, 10, 5)

    if not q.strip():
        return

    with st.spinner("Searching..."):
        ans = rag_mod.answer(q, retriever, llm, k=k)

    if ans.note:
        st.caption(ans.note)
    if ans.generated and ans.text:
        st.markdown("#### Answer")
        st.markdown(ans.text)
        st.caption("Generated by a small local model from the sources below. "
                   "Verify against them.")
    elif ans.text:
        st.markdown(ans.text)

    if ans.hits:
        st.markdown("#### Sources")
        for h in ans.hits:
            with st.expander(f"{h.window.header()} · {', '.join(h.window.senders)} "
                             f"· relevance {h.score:.2f}"):
                st.text(h.window.as_text())


# --- main -------------------------------------------------------------------

def main() -> None:
    th = theme_mod.current()
    df, report, llm, local_stops = sidebar()

    st.title("WhatsApp Group Chat Analyzer")

    if df is None or df.empty or report is None or not report.ok:
        st.info(
            "**Upload a WhatsApp export to begin**, or tick *Use the sample chat* "
            "in the sidebar.\n\n"
            "To export: open the chat in WhatsApp → **⋮ → More → Export chat → "
            "Without media**. Both iPhone and Android exports work."
        )
        if report is not None and not report.ok:
            st.error(
                f"Parsed 0 messages from {report.total_lines:,} lines. "
                "This may not be a WhatsApp export."
            )
            if report.unparsed_samples:
                with st.expander("First unparsed lines"):
                    st.code("\n".join(report.unparsed_samples))
        return

    with st.expander(f"Parsed: {report.summary()}", expanded=False):
        c = st.columns(4)
        c[0].metric("Platform", report.platform)
        c[1].metric("System notices", f"{report.system_messages:,}")
        c[2].metric("Continuation lines", f"{report.continuation_lines:,}")
        c[3].metric("Unparsed lines", f"{report.unparsed_lines:,}")
        st.caption(f"Dates read as `{report.date_format}`, times as "
                   f"`{report.time_format}`. If those look wrong, the export "
                   "uses a locale the parser guessed incorrectly.")
        if report.unparsed_samples:
            st.code("\n".join(report.unparsed_samples))

    stops = text_mod.stopwords(include_local=local_stops)
    key = _chat_key(df)
    sidebar_models(key)

    tabs = st.tabs(["Overview", "People", "Summary", "Topics",
                    "Sentiment", "Authorship", "Chat with your chats"])
    with tabs[0]:
        tab_overview(df, th, stops)
    with tabs[1]:
        tab_people(df, th, stops)
    with tabs[2]:
        tab_summary(df, th, llm)
    with tabs[3]:
        tab_topics(df, th, llm, key, local_stops)
    with tabs[4]:
        tab_sentiment(df, th, key)
    with tabs[5]:
        tab_authorship(df, th, key)
    with tabs[6]:
        tab_chat(df, th, llm, key)


if __name__ == "__main__":
    main()
