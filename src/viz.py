"""Plotly charts for the chat analyzer.

Colour is assigned by the job it does: message counts are magnitude (one hue,
light to dark), people are identity (distinct hues, fixed order), and the
activity heatmap is magnitude again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import Theme, apply_layout, sequential_scale

BAR_RADIUS = 4  # rounded data-end, anchored at the baseline


def _axes(fig: go.Figure, theme: Theme, xtitle: str = "", ytitle: str = "",
          xgrid: bool = False, ygrid: bool = True) -> None:
    # Axis title styling lives under title.font; the old top-level `titlefont`
    # shorthand was removed in Plotly 6.
    title_font = dict(color=theme.muted, size=11)
    fig.update_xaxes(
        title=dict(text=xtitle, font=title_font),
        showgrid=xgrid, gridcolor=theme.grid,
        linecolor=theme.axis, zeroline=False,
        tickfont=dict(color=theme.muted, size=11),
    )
    fig.update_yaxes(
        title=dict(text=ytitle, font=title_font),
        showgrid=ygrid, gridcolor=theme.grid,
        linecolor=theme.axis, zeroline=False,
        tickfont=dict(color=theme.muted, size=11),
    )


def most_active(per_person: pd.DataFrame, theme: Theme, metric: str = "messages",
                height: int = 380) -> go.Figure:
    """Horizontal ranking of participants. Magnitude, so a single hue."""
    if per_person.empty:
        return empty(theme, "No messages")
    d = per_person.sort_values(metric, ascending=True)
    vals = d[metric].to_numpy(dtype=float)
    # Darker = larger, using the sequential ramp.
    ramp = theme.sequential
    lo, hi = vals.min(), vals.max()
    idx = np.zeros_like(vals, dtype=int) if hi <= lo else \
        np.clip(((vals - lo) / (hi - lo) * (len(ramp) - 1)).round().astype(int),
                0, len(ramp) - 1)
    colors = [ramp[i] for i in idx]

    label = {"messages": "Messages", "words": "Words",
             "avg_words": "Avg words per message"}.get(metric, metric)
    fmt = ",.1f" if metric == "avg_words" else ",.0f"

    fig = go.Figure(go.Bar(
        x=vals, y=d["sender"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,.1f}" if metric == "avg_words" else f"{v:,.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color=theme.text_secondary, size=11),
        hovertemplate=f"%{{y}}<br>{label}: %{{x:{fmt}}}<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    _axes(fig, theme, xtitle=label, xgrid=True, ygrid=False)
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=40, t=10, b=0), bargap=0.25)
    return fig


def bar(df: pd.DataFrame, x: str, y: str, theme: Theme, xtitle: str = "",
        ytitle: str = "Messages", height: int = 300,
        slot: int = 0) -> go.Figure:
    """Vertical bar chart for one series."""
    if df.empty:
        return empty(theme, "No data")
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y],
        marker=dict(color=theme.series[slot % len(theme.series)], line=dict(width=0)),
        hovertemplate="%{x}<br>%{y:,} messages<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    _axes(fig, theme, xtitle=xtitle, ytitle=ytitle)
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), bargap=0.2)
    return fig


def timeline(day_df: pd.DataFrame, theme: Theme, height: int = 320) -> go.Figure:
    """Messages per day with a 7-day rolling average on top."""
    if day_df.empty:
        return empty(theme, "No data")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=day_df["date"], y=day_df["messages"], name="Messages per day",
        marker=dict(color=theme.dim, line=dict(width=0)),
        hovertemplate="%{x|%d %b %Y}<br>%{y:,} messages<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=day_df["date"], y=day_df["rolling_7"], name="7-day average",
        mode="lines", line=dict(color=theme.series[0], width=2),
        hovertemplate="%{x|%d %b %Y}<br>avg %{y:.1f}<extra></extra>",
    ))
    _axes(fig, theme, ytitle="Messages")
    apply_layout(fig, theme, height=height)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
    return fig


def hour_weekday_heatmap(grid: np.ndarray, days: list[str], hours: list[int],
                         theme: Theme, height: int = 340) -> go.Figure:
    """When the group is awake. Magnitude, so the sequential ramp."""
    fig = go.Figure(go.Heatmap(
        z=grid, x=[f"{h:02d}" for h in hours], y=days,
        colorscale=sequential_scale(theme),
        xgap=2, ygap=2,
        hovertemplate="%{y} %{x}:00<br>%{z:,} messages<extra></extra>",
        colorbar=dict(
            title=dict(text="messages", font=dict(color=theme.muted, size=11)),
            tickfont=dict(color=theme.muted, size=10),
            thickness=10, len=0.7, outlinewidth=0,
        ),
    ))
    _axes(fig, theme, xtitle="Hour of day", xgrid=False, ygrid=False)
    fig.update_yaxes(autorange="reversed")
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def response_distribution(samples: pd.DataFrame, theme: Theme,
                          height: int = 340) -> go.Figure:
    """Reply-gap distribution per person, as box plots."""
    if samples.empty:
        return empty(theme, "Not enough back-and-forth to measure reply times")
    order = (samples.groupby("sender")["minutes"].median()
             .sort_values().index.tolist())
    fig = go.Figure()
    for i, sender in enumerate(order):
        vals = samples.loc[samples["sender"] == sender, "minutes"]
        fig.add_trace(go.Box(
            y=vals, name=str(sender),
            marker=dict(color=theme.series[i % len(theme.series)]),
            line=dict(width=2),
            boxpoints=False,
        ))
    _axes(fig, theme, ytitle="Minutes to reply")
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    fig.update_yaxes(type="log")
    return fig


def stacked_over_time(long_df: pd.DataFrame, theme: Theme,
                      height: int = 340, max_people: int = 3) -> go.Figure:
    """Messages per person over time.

    Capped at three people plus 'Other': beyond that, adjacent colours stop
    being reliably distinguishable for colourblind readers.
    """
    if long_df.empty:
        return empty(theme, "No data")
    totals = long_df.groupby("sender")["messages"].sum().sort_values(ascending=False)
    keep = list(totals.index[:max_people])
    d = long_df.copy()
    d["group"] = d["sender"].where(d["sender"].isin(keep), "Other")
    d = d.groupby(["period", "group"], as_index=False)["messages"].sum()

    fig = go.Figure()
    for i, name in enumerate(keep + (["Other"] if (d["group"] == "Other").any() else [])):
        g = d[d["group"] == name]
        color = theme.dim if name == "Other" else theme.series[i % len(theme.series)]
        fig.add_trace(go.Bar(
            x=g["period"], y=g["messages"], name=str(name),
            marker=dict(color=color, line=dict(width=2, color=theme.surface)),
            hovertemplate="%{x|%d %b %Y}<br>" + f"{name}: " + "%{y:,}<extra></extra>",
        ))
    _axes(fig, theme, ytitle="Messages")
    apply_layout(fig, theme, height=height)
    fig.update_layout(barmode="stack", margin=dict(l=0, r=0, t=10, b=0), bargap=0.15)
    return fig


def topic_bar(words: list[str], weights: list[float], theme: Theme,
              height: int = 260) -> go.Figure:
    """Top terms for one topic."""
    fig = go.Figure(go.Bar(
        x=weights[::-1], y=words[::-1], orientation="h",
        marker=dict(color=theme.series[0], line=dict(width=0)),
        hovertemplate="%{y}<br>weight %{x:.3f}<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    _axes(fig, theme, xtitle="Weight", xgrid=True, ygrid=False)
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), bargap=0.25)
    return fig


def confusion_matrix(cm: np.ndarray, labels: list[str], theme: Theme,
                     height: int = 420) -> go.Figure:
    """Row-normalised confusion matrix — who writes like whom."""
    total = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, np.maximum(total, 1)) * 100
    fig = go.Figure(go.Heatmap(
        z=pct, x=labels, y=labels,
        colorscale=sequential_scale(theme), zmin=0, zmax=100,
        xgap=2, ygap=2,
        text=[[f"{v:.0f}%" for v in row] for row in pct],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="actually %{y}<br>predicted %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(
            title=dict(text="% of row", font=dict(color=theme.muted, size=11)),
            tickfont=dict(color=theme.muted, size=10),
            thickness=10, len=0.7, outlinewidth=0,
        ),
    ))
    _axes(fig, theme, xtitle="Predicted", ytitle="Actual", xgrid=False, ygrid=False)
    fig.update_yaxes(autorange="reversed")
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def sentiment_over_time(df: pd.DataFrame, theme: Theme,
                        height: int = 320) -> go.Figure:
    """Share of positive messages over time — diverging around neutral."""
    if df.empty:
        return empty(theme, "No sentiment data")
    fig = go.Figure(go.Scatter(
        x=df["period"], y=df["positive_share"],
        mode="lines+markers",
        line=dict(color=theme.series[0], width=2),
        marker=dict(size=7, color=theme.series[0],
                    line=dict(width=2, color=theme.surface)),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.0%} positive<extra></extra>",
    ))
    fig.add_hline(y=0.5, line=dict(color=theme.axis, width=1, dash="dot"))
    _axes(fig, theme, ytitle="Share positive")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    apply_layout(fig, theme, height=height, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def empty(theme: Theme, message: str, height: int = 220) -> go.Figure:
    """Placeholder that says why there is nothing to show."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(color=theme.muted, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    apply_layout(fig, theme, height=height, legend=False)
    return fig
