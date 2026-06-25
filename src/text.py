"""Tokenizing, stopwords, and emoji handling for chat text.

Chat language is not prose: it is short, misspelt, full of emoji, and sprinkled
with "haha", "ok", and "lol". The stopword list below reflects that — a
standard English list alone leaves the top-words chart full of chat filler.
"""

from __future__ import annotations

import re
from collections import Counter

# Standard English function words.
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves", "s", "t", "don't", "im", "i'm", "its", "it's",
    "dont", "doesn't", "didn't", "isn't", "aren't", "won't", "can't",
}

# Chat-specific filler. Without these the "top words" chart is mostly laughter.
CHAT_STOPWORDS = {
    "ok", "okay", "ok", "lol", "haha", "hahaha", "hahahaha", "hehe", "yeah",
    "yes", "yep", "yup", "no", "nope", "nah", "hi", "hey", "hello", "thanks",
    "thank", "pls", "please", "u", "ur", "n", "k", "kk", "lmao", "lmfao",
    "omg", "wow", "hmm", "hmmm", "eh", "ah", "oh", "well", "like", "get",
    "got", "know", "think", "one", "go", "going", "come", "see", "good",
    "great", "nice", "sure", "let", "us", "also", "still", "even", "back",
    "make", "want", "need", "time", "day", "today", "tomorrow", "yesterday",
    "media", "omitted", "message", "deleted", "attached", "image", "video",
    "audio", "sticker", "gif", "document", "null",
}

# Common in Ugandan group chats — code-switching with Luganda/Swahili is normal.
LUGANDA_SWAHILI_STOPWORDS = {
    "nga", "nti", "ku", "mu", "ne", "era", "naye", "bwe", "ki", "kyi", "bambi",
    "ssebo", "nnyabo", "webale", "kale", "banange", "gyebale", "wangi",
    "sasa", "sawa", "poa", "asante", "karibu", "pole", "haraka", "bwana",
    "ndio", "hapana", "nini", "sana", "tu", "kwa", "na", "ya", "wa", "ni",
}

DEFAULT_STOPWORDS = ENGLISH_STOPWORDS | CHAT_STOPWORDS

WORD_RE = re.compile(r"[a-z']{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Emoji live in a handful of Unicode blocks; this covers the ones people use.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"
    "\U0001F900-\U0001F9FF"   # supplemental symbols (incl. most newer faces)
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"   # misc symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002764"              # heavy black heart
    "]",
    flags=re.UNICODE,
)


def stopwords(include_local: bool = False, extra: set[str] | None = None) -> set[str]:
    """Assemble a stopword set."""
    s = set(DEFAULT_STOPWORDS)
    if include_local:
        s |= LUGANDA_SWAHILI_STOPWORDS
    if extra:
        s |= {w.strip().lower() for w in extra if w.strip()}
    return s


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def extract_emoji(text: str) -> list[str]:
    return EMOJI_RE.findall(text or "")


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub(" ", text or "")


def tokenize(text: str, stops: set[str] | None = None,
             keep_urls: bool = False) -> list[str]:
    """Lowercase word tokens with URLs, emoji and stopwords removed."""
    t = (text or "").lower()
    if not keep_urls:
        t = strip_urls(t)
    t = strip_emoji(t)
    stops = DEFAULT_STOPWORDS if stops is None else stops
    return [w for w in WORD_RE.findall(t) if w not in stops and len(w) > 1]


def top_words(texts: list[str], n: int = 25, stops: set[str] | None = None) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        c.update(tokenize(t, stops))
    return c.most_common(n)


def top_emoji(texts: list[str], n: int = 15) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        c.update(extract_emoji(t))
    return c.most_common(n)


def vocabulary_richness(texts: list[str], stops: set[str] | None = None) -> float:
    """Type-token ratio: distinct words divided by total words.

    Higher means more varied language. It is sensitive to sample size, so only
    compare people with roughly similar message counts.
    """
    toks: list[str] = []
    for t in texts:
        toks.extend(tokenize(t, stops))
    return len(set(toks)) / len(toks) if toks else 0.0
