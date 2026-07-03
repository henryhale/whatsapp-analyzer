"""Descriptive statistics over a parsed chat.

Everything here takes the DataFrame produced by `parser.to_dataframe` and
returns either a plain dict (for headline numbers) or a tidy DataFrame ready to
chart. No plotting happens in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import text as text_mod

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]

# A gap longer than this starts a new conversation rather than continuing one.
CONVERSATION_GAP_MINUTES = 60
# Replies slower than this are someone starting a new thread, not responding.
MAX_RESPONSE_MINUTES = 360


def user_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Real messages from real people — excludes system notices."""
    return df[df["is_user"]] if not df.empty else df


def headline(df: pd.DataFrame) -> dict:
    """The numbers that go across the top of the Overview tab."""
    u = user_messages(df)
    if u.empty:
        return {}

    per_day = u.groupby("date").size()
    days_span = (u["timestamp"].max() - u["timestamp"].min()).days + 1
    busiest_day = per_day.idxmax()

    # Longest stretch with no messages at all.
    ts = u["timestamp"].sort_values()
    gaps = ts.diff().dropna()
    longest_gap = gaps.max() if not gaps.empty else pd.Timedelta(0)

    return {
        "messages": int(len(u)),
        "participants": int(u["sender"].nunique()),
        "first": u["timestamp"].min(),
        "last": u["timestamp"].max(),
        "days_span": int(days_span),
        "active_days": int(u["date"].nunique()),
        "avg_per_active_day": float(per_day.mean()),
        "busiest_day": busiest_day,
        "busiest_day_count": int(per_day.max()),
        "media": int((df["kind"] == "media").sum()),
        "deleted": int((df["kind"] == "deleted").sum()),
        "edited": int(df["edited"].sum()),
        "system": int((df["kind"] == "system").sum()),
        "total_words": int(u["n_words"].sum()),
        "longest_silence_hours": float(longest_gap.total_seconds() / 3600),
    }


def per_person(df: pd.DataFrame, stops: set[str] | None = None) -> pd.DataFrame:
    """One row per participant, sorted by message count."""
    u = user_messages(df)
    if u.empty:
        return pd.DataFrame()

    rows = []
    for sender, g in u.groupby("sender"):
        texts = g.loc[g["kind"] == "text", "text"].tolist()
        emoji = [e for t in texts for e in text_mod.extract_emoji(t)]
        rows.append({
            "sender": sender,
            "messages": len(g),
            "words": int(g["n_words"].sum()),
            "avg_words": float(g["n_words"].mean()),
            "chars": int(g["n_chars"].sum()),
            "media": int((g["kind"] == "media").sum()),
            "deleted": int((g["kind"] == "deleted").sum()),
            "emoji": len(emoji),
            "emoji_per_msg": len(emoji) / max(1, len(g)),
            "vocabulary_richness": text_mod.vocabulary_richness(texts, stops),
            "active_days": int(g["date"].nunique()),
        })

    out = pd.DataFrame(rows).sort_values("messages", ascending=False)
    out["share"] = 100 * out["messages"] / out["messages"].sum()
    return out.reset_index(drop=True)


def by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per calendar day, with gaps filled so the line is honest."""
    u = user_messages(df)
    if u.empty:
        return pd.DataFrame(columns=["date", "messages", "rolling_7"])
    s = u.groupby("date").size()
    idx = pd.date_range(u["timestamp"].min().date(), u["timestamp"].max().date(), freq="D")
    s = s.reindex(idx.date, fill_value=0)
    out = pd.DataFrame({"date": pd.to_datetime(list(s.index)), "messages": s.values})
    out["rolling_7"] = out["messages"].rolling(7, min_periods=1).mean()
    return out


def by_hour(df: pd.DataFrame) -> pd.DataFrame:
    u = user_messages(df)
    if u.empty:
        return pd.DataFrame(columns=["hour", "messages"])
    s = u.groupby("hour").size().reindex(range(24), fill_value=0)
    return pd.DataFrame({"hour": s.index, "messages": s.values})


def by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    u = user_messages(df)
    if u.empty:
        return pd.DataFrame(columns=["weekday", "messages"])
    s = u.groupby("weekday").size().reindex(WEEKDAY_ORDER, fill_value=0)
    return pd.DataFrame({"weekday": s.index, "messages": s.values})


def by_month(df: pd.DataFrame) -> pd.DataFrame:
    u = user_messages(df)
    if u.empty:
        return pd.DataFrame(columns=["month", "messages"])
    s = u.groupby("month").size().sort_index()
    return pd.DataFrame({"month": s.index, "messages": s.values})


def hour_weekday_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    """Counts as a (7 x 24) grid for the activity heatmap."""
    u = user_messages(df)
    grid = np.zeros((7, 24), dtype=int)
    if u.empty:
        return grid, WEEKDAY_ORDER, list(range(24))
    pivot = u.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=WEEKDAY_ORDER, columns=range(24), fill_value=0)
    return pivot.values, WEEKDAY_ORDER, list(range(24))


def response_times(df: pd.DataFrame,
                   max_minutes: int = MAX_RESPONSE_MINUTES) -> pd.DataFrame:
    """How quickly each person replies to someone else.

    Only counts a message as a reply when the previous message came from a
    different person within `max_minutes` — otherwise it is a new thread, not a
    response, and would inflate the numbers with overnight gaps.
    """
    u = user_messages(df).sort_values("timestamp")
    if len(u) < 2:
        return pd.DataFrame(columns=["sender", "median_minutes", "mean_minutes", "replies"])

    prev_sender = u["sender"].shift(1)
    delta = (u["timestamp"] - u["timestamp"].shift(1)).dt.total_seconds() / 60
    is_reply = (prev_sender.notna()) & (prev_sender != u["sender"]) & (delta <= max_minutes)

    replies = pd.DataFrame({
        "sender": u["sender"][is_reply],
        "minutes": delta[is_reply],
    })
    if replies.empty:
        return pd.DataFrame(columns=["sender", "median_minutes", "mean_minutes", "replies"])

    out = replies.groupby("sender")["minutes"].agg(
        median_minutes="median", mean_minutes="mean", replies="size"
    ).reset_index()
    return out.sort_values("median_minutes")


def response_time_samples(df: pd.DataFrame,
                          max_minutes: int = MAX_RESPONSE_MINUTES) -> pd.DataFrame:
    """Raw per-reply gaps, for a distribution plot."""
    u = user_messages(df).sort_values("timestamp")
    if len(u) < 2:
        return pd.DataFrame(columns=["sender", "minutes"])
    prev_sender = u["sender"].shift(1)
    delta = (u["timestamp"] - u["timestamp"].shift(1)).dt.total_seconds() / 60
    m = (prev_sender.notna()) & (prev_sender != u["sender"]) & (delta <= max_minutes)
    return pd.DataFrame({"sender": u["sender"][m], "minutes": delta[m]})


def conversation_starters(df: pd.DataFrame,
                          gap_minutes: int = CONVERSATION_GAP_MINUTES) -> pd.DataFrame:
    """Who breaks the silence — first message after a gap."""
    u = user_messages(df).sort_values("timestamp")
    if u.empty:
        return pd.DataFrame(columns=["sender", "starts"])
    gap = (u["timestamp"] - u["timestamp"].shift(1)).dt.total_seconds() / 60
    starts = u[(gap.isna()) | (gap > gap_minutes)]
    out = starts.groupby("sender").size().reset_index(name="starts")
    return out.sort_values("starts", ascending=False).reset_index(drop=True)


def streaks(df: pd.DataFrame) -> dict:
    """Longest run of consecutive days with at least one message."""
    u = user_messages(df)
    if u.empty:
        return {"longest_streak_days": 0, "streak_start": None, "streak_end": None}

    days = sorted(set(u["date"]))
    best = cur = 1
    best_end = cur_start = days[0]
    best_start = days[0]
    for prev, day in zip(days, days[1:]):
        if (day - prev).days == 1:
            cur += 1
        else:
            cur, cur_start = 1, day
        if cur > best:
            best, best_start, best_end = cur, cur_start, day
    return {"longest_streak_days": best, "streak_start": best_start, "streak_end": best_end}


def activity_by_person_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Messages per person per period — long format, ready for a chart."""
    u = user_messages(df)
    if u.empty:
        return pd.DataFrame(columns=["period", "sender", "messages"])
    g = (u.set_index("timestamp")
           .groupby([pd.Grouper(freq=freq), "sender"])
           .size()
           .reset_index(name="messages"))
    g.columns = ["period", "sender", "messages"]
    return g
