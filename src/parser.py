"""Parse WhatsApp chat exports.

This is the part that actually decides whether the rest of the project works,
and it is fiddlier than it looks. Exports vary by platform, phone locale, and
WhatsApp version:

    Android   15/08/2026, 14:23 - Alice: hello
              8/15/26, 2:23 PM - Alice: hello
    iOS       [15/08/2026, 14:23:11] Alice: hello
              [8/15/26, 2:23:11 PM] Alice: hello

On top of that, real exports contain invisible characters that quietly break
naive regexes — U+200E left-to-right marks, U+202F narrow no-break spaces
before AM/PM, U+00A0 non-breaking spaces — and dd/mm vs mm/dd is genuinely
ambiguous until you find a day past the 12th.

The approach: normalise the invisible characters, try every known line-start
shape and keep whichever matches the most lines, then infer the date order from
the data rather than assuming a locale.

Deliberately stdlib-only so it can be tested without pandas installed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Invisible characters that appear in real exports and break naive matching.
_INVISIBLE = dict.fromkeys(
    [
        0x200E,  # left-to-right mark (iOS prefixes lines and attachments)
        0x200F,  # right-to-left mark
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embedding/override
        0x00AD,  # soft hyphen
        0xFEFF,  # BOM / zero-width no-break space
    ],
    None,
)
# Space-like characters normalised to a plain space.
_SPACES = {0x00A0: " ", 0x202F: " ", 0x2009: " ", 0x2007: " "}

_DATE = r"\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4}"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?"
_AMPM = r"(?:[APap]\.?[Mm]\.?)?"

# iOS wraps the stamp in brackets; Android separates it with a dash.
IOS_RE = re.compile(
    rf"^\[\s*(?P<date>{_DATE}),?\s+(?P<time>{_TIME})\s*(?P<ampm>{_AMPM})\s*\]\s*(?P<rest>.*)$"
)
ANDROID_RE = re.compile(
    rf"^(?P<date>{_DATE}),?\s+(?P<time>{_TIME})\s*(?P<ampm>{_AMPM})\s+-\s+(?P<rest>.*)$"
)
FORMATS = {"ios": IOS_RE, "android": ANDROID_RE}

# "Sender: text". Bounded length and no colon inside, so a message that merely
# contains a colon is not mistaken for a new sender.
SENDER_RE = re.compile(r"^(?P<sender>[^:\n]{1,120}?):\s(?P<text>.*)$", re.DOTALL)

MEDIA_MARKERS = (
    "<media omitted>", "<médias omis>", "image omitted", "video omitted",
    "audio omitted", "sticker omitted", "gif omitted", "document omitted",
    "contact card omitted", "this message was deleted", "you deleted this message",
)
ATTACHED_RE = re.compile(r"^<attached:\s*(?P<name>.+?)>\s*$", re.IGNORECASE)
DELETED_MARKERS = ("this message was deleted", "you deleted this message")
EDITED_SUFFIX_RE = re.compile(r"\s*<this message was edited>\s*$", re.IGNORECASE)

# Phrases that identify a system notice when no "Sender:" prefix is present.
SYSTEM_HINTS = (
    "end-to-end encrypted", "created group", "added", "removed", "left",
    "changed the subject", "changed this group's icon", "changed their phone number",
    "joined using this group's invite link", "changed the group description",
    "you're now an admin", "is now an admin", "no longer an admin",
    "messages and calls are end-to-end encrypted", "changed to",
    "security code changed", "pinned a message", "turned on disappearing messages",
)

KIND_TEXT, KIND_MEDIA, KIND_DELETED, KIND_SYSTEM = "text", "media", "deleted", "system"


def normalize(text: str) -> str:
    """Strip invisible marks and normalise exotic spaces and line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_INVISIBLE).translate(_SPACES)
    # Compose accented sender names consistently so "José" matches itself.
    return unicodedata.normalize("NFC", text)


# --- date handling ----------------------------------------------------------

_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y",
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%m-%d-%y",
    "%d.%m.%Y", "%m.%d.%Y", "%d.%m.%y", "%m.%d.%y",
]
_TIME_FORMATS_24 = ["%H:%M:%S", "%H:%M"]
_TIME_FORMATS_12 = ["%I:%M:%S %p", "%I:%M %p"]


def _day_first_hint(dates: list[str]) -> bool | None:
    """Decide dd/mm vs mm/dd from the data. None when genuinely ambiguous."""
    first_over_12 = second_over_12 = False
    for d in dates:
        parts = re.split(r"[./\-]", d)
        if len(parts) != 3:
            continue
        # An ISO-style leading year settles nothing about the other two.
        if len(parts[0]) == 4:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if a > 12:
            first_over_12 = True
        if b > 12:
            second_over_12 = True
    if first_over_12 and not second_over_12:
        return True
    if second_over_12 and not first_over_12:
        return False
    return None


def infer_datetime_format(dates: list[str], times: list[str],
                          ampms: list[str]) -> tuple[str, str]:
    """Pick the (date_fmt, time_fmt) pair that parses the most samples."""
    has_ampm = any(a for a in ampms)
    time_candidates = _TIME_FORMATS_12 if has_ampm else _TIME_FORMATS_24

    time_fmt = time_candidates[0]
    best_t = -1
    for tf in time_candidates:
        n = 0
        for t, a in zip(times, ampms):
            s = f"{t} {a.upper().replace('.', '')}".strip() if has_ampm else t
            try:
                datetime.strptime(s, tf)
                n += 1
            except ValueError:
                pass
        if n > best_t:
            best_t, time_fmt = n, tf

    day_first = _day_first_hint(dates)
    # Order candidates so the hinted convention wins ties.
    ordered = sorted(
        _DATE_FORMATS,
        key=lambda f: (
            0 if day_first is None
            else (0 if (f.startswith("%d") == day_first) else 1)
        ),
    )
    date_fmt, best_d = ordered[0], -1
    for df in ordered:
        n = 0
        for d in dates:
            try:
                datetime.strptime(d, df)
                n += 1
            except ValueError:
                pass
        if n > best_d:
            best_d, date_fmt = n, df
    return date_fmt, time_fmt


# --- messages ---------------------------------------------------------------

@dataclass
class Message:
    timestamp: datetime
    sender: str | None      # None for system notices
    text: str
    kind: str = KIND_TEXT
    edited: bool = False
    attachment: str | None = None

    @property
    def is_user(self) -> bool:
        return self.sender is not None and self.kind != KIND_SYSTEM


@dataclass
class ParseReport:
    """What the parser saw — surfaced in the UI so failures are visible."""

    platform: str = "unknown"
    date_format: str = ""
    time_format: str = ""
    total_lines: int = 0
    parsed_messages: int = 0
    system_messages: int = 0
    continuation_lines: int = 0
    unparsed_lines: int = 0
    unparsed_samples: list[str] = field(default_factory=list)
    bad_timestamps: int = 0

    @property
    def ok(self) -> bool:
        return self.parsed_messages > 0

    def summary(self) -> str:
        return (
            f"{self.platform} export · {self.parsed_messages:,} messages · "
            f"dates as {self.date_format} · {self.unparsed_lines:,} lines unparsed"
        )


def _classify(rest: str) -> tuple[str | None, str, str, str | None]:
    """Split a post-timestamp line into (sender, text, kind, attachment)."""
    m = SENDER_RE.match(rest)
    if not m:
        return None, rest.strip(), KIND_SYSTEM, None

    sender = m.group("sender").strip()
    text = m.group("text")

    # A "sender" that reads like a system phrase is a system notice whose text
    # happened to contain a colon.
    low = sender.lower()
    if any(h in low for h in SYSTEM_HINTS):
        return None, rest.strip(), KIND_SYSTEM, None

    attachment = None
    kind = KIND_TEXT
    stripped = text.strip()
    low_text = stripped.lower()

    am = ATTACHED_RE.match(stripped)
    if am:
        kind, attachment = KIND_MEDIA, am.group("name")
    elif any(mk in low_text for mk in DELETED_MARKERS):
        kind = KIND_DELETED
    elif any(mk in low_text for mk in MEDIA_MARKERS):
        kind = KIND_MEDIA

    return sender, text, kind, attachment


def detect_platform(lines: list[str], sample: int = 400) -> tuple[str, re.Pattern]:
    """Whichever line-start shape matches the most lines wins."""
    scores = {
        name: sum(1 for ln in lines[:sample] if rx.match(ln))
        for name, rx in FORMATS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "unknown", ANDROID_RE
    return best, FORMATS[best]


def parse_lines(lines: list[str]) -> tuple[list[Message], ParseReport]:
    """Parse normalised lines into messages."""
    report = ParseReport(total_lines=len(lines))
    platform, line_re = detect_platform(lines)
    report.platform = platform

    # Pass 1: split into (stamp, rest) records, folding continuations in.
    raw: list[tuple[str, str, str, str]] = []   # date, time, ampm, rest
    for ln in lines:
        m = line_re.match(ln)
        if m:
            raw.append((m.group("date"), m.group("time"),
                        m.group("ampm") or "", m.group("rest")))
        elif raw:
            # A line that does not start a message continues the previous one.
            d, t, a, rest = raw[-1]
            raw[-1] = (d, t, a, rest + "\n" + ln)
            report.continuation_lines += 1
        elif ln.strip():
            report.unparsed_lines += 1
            if len(report.unparsed_samples) < 5:
                report.unparsed_samples.append(ln[:120])

    if not raw:
        return [], report

    # Pass 2: work out how this export writes dates, then build messages.
    date_fmt, time_fmt = infer_datetime_format(
        [r[0] for r in raw[:500]], [r[1] for r in raw[:500]], [r[2] for r in raw[:500]]
    )
    report.date_format, report.time_format = date_fmt, time_fmt
    use_ampm = "%p" in time_fmt

    messages: list[Message] = []
    for d, t, a, rest in raw:
        stamp = f"{t} {a.upper().replace('.', '')}".strip() if use_ampm else t
        try:
            ts = datetime.strptime(f"{d} {stamp}", f"{date_fmt} {time_fmt}")
        except ValueError:
            report.bad_timestamps += 1
            continue

        edited = bool(EDITED_SUFFIX_RE.search(rest))
        if edited:
            rest = EDITED_SUFFIX_RE.sub("", rest)
        sender, text, kind, attachment = _classify(rest)
        if kind == KIND_SYSTEM:
            report.system_messages += 1
        messages.append(Message(
            timestamp=ts, sender=sender, text=text.strip(),
            kind=kind, edited=edited, attachment=attachment,
        ))

    report.parsed_messages = sum(1 for m in messages if m.is_user)
    messages.sort(key=lambda m: m.timestamp)
    return messages, report


def parse_text(text: str) -> tuple[list[Message], ParseReport]:
    return parse_lines(normalize(text).split("\n"))


def parse_file(path: str | Path) -> tuple[list[Message], ParseReport]:
    """Parse a .txt export, or a .zip containing one."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                raise ValueError("No .txt file inside the zip.")
            with z.open(names[0]) as f:
                return parse_text(f.read().decode("utf-8", errors="replace"))
    return parse_text(p.read_text(encoding="utf-8", errors="replace"))


def participants(messages: list[Message]) -> list[str]:
    """Senders ordered by message count, most active first."""
    c = Counter(m.sender for m in messages if m.is_user)
    return [s for s, _ in c.most_common()]


def to_dataframe(messages: list[Message]):
    """Messages as a pandas DataFrame (imported lazily to keep this stdlib-only)."""
    import pandas as pd

    df = pd.DataFrame([
        {
            "timestamp": m.timestamp,
            "sender": m.sender,
            "text": m.text,
            "kind": m.kind,
            "edited": m.edited,
            "attachment": m.attachment,
            "is_user": m.is_user,
        }
        for m in messages
    ])
    if df.empty:
        return df
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)
    return df
