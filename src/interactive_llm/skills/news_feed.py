"""Fresh headlines from Google News RSS (key-less). Used for news questions: web-search news indexes were weeks old."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

MAX_HEADLINES = 6
_BASE = "https://news.google.com/rss"
_LOCALE = "hl=en-US&gl=US&ceid=US:en"
# Words in a request that map to one of Google News' own topic sections (better than a keyword search for broad topics).
SECTIONS = {
    "BUSINESS": r"business|finance|financial|economy|economic|markets?|stocks?|wall street",
    "TECHNOLOGY": r"tech|technology|ai|artificial intelligence|gadgets?|software",
    "SPORTS": r"sports?",
    "SCIENCE": r"science|space",
    "HEALTH": r"health|medical|medicine",
    "ENTERTAINMENT": r"entertainment|celebrity|celebrities|movies?|music|film",
    "WORLD": r"world|international|global",
    "NATION": r"national|us|u\.s\.|america|american|domestic",
}
_FILLER_RE = re.compile(r"\b(?:news|headlines?|stories|story|latest|top|today'?s?|today|now|right now|current|breaking|new|recent|the|on|about|for|me|"
                        r"any|some|of|in|what'?s|what is|happening|going on|with)\b", re.I)


def age_text(published: datetime, now: datetime | None = None) -> str:
    """Human wording for how long ago `published` was: "35 minutes ago", "5 hours ago", "2 days ago"."""
    seconds = max(0, int(((now or datetime.now(timezone.utc)) - published).total_seconds()))
    count, unit = ((max(1, seconds // 60), "minute") if seconds < 3600 else (seconds // 3600, "hour") if seconds < 86400
                   else (seconds // 86400, "day"))
    return f"{count} {unit}{'s' if count != 1 else ''} ago"


def feed_url(query: str) -> str:
    """Pick the Google News feed for a request: the top stories, a topic section, or a search for the remaining words."""
    topic = _FILLER_RE.sub(" ", query).strip()
    topic = re.sub(r"\s+", " ", topic)
    if not topic:
        return f"{_BASE}?{_LOCALE}"
    for section, words in SECTIONS.items():
        if re.fullmatch(f"(?:{words})", topic, re.I):
            return f"{_BASE}/headlines/section/topic/{section}?{_LOCALE}"
    return f"{_BASE}/search?q={urllib.parse.quote(topic + ' when:3d')}&{_LOCALE}"


def parse_feed(xml_text: str, now: datetime | None = None) -> list[dict]:
    """Parse an RSS document into ``[{"title", "body", "url", "source", "headline": True}]``, newest first."""
    rows = []
    for item in ET.fromstring(xml_text).findall(".//item"):
        title = (item.findtext("title") or "").strip()
        source = item.findtext("source") or ""
        if " - " in title:                                   # Google appends " - Publisher" to every title
            title, _, suffix = title.rpartition(" - ")
            source = source or suffix
        try:
            published = parse_date(item.findtext("pubDate"))
        except (TypeError, ValueError):
            continue
        rows.append({"title": title, "source": source, "url": item.findtext("link") or "", "headline": True,
                     "body": f"{source}, {age_text(published, now)}", "published": published})
    rows.sort(key=lambda row: row["published"], reverse=True)
    return rows


def parse_date(text: str | None) -> datetime:
    """Parse an RSS pubDate (RFC 822) into an aware datetime."""
    stamp = parsedate_to_datetime(text or "")
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def fetch_news(query: str) -> list[dict]:
    """Return up to six fresh headlines for `query` (empty list if the feed gives nothing)."""
    request = urllib.request.Request(feed_url(query), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=6) as page:
        rows = parse_feed(page.read(2_000_000).decode("utf-8", "replace"))
    seen, unique = set(), []
    for row in rows:                                         # the same story often appears from several outlets
        key = row["title"].lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique[:MAX_HEADLINES]
