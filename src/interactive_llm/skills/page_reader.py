"""Fetch a web page and return the few passages most relevant to a query (plain HTTP, no browser).

Search snippets are one or two sentences, too thin to answer from. Reading the top result pages gives the model real
content. Page text is untrusted data: it is length-limited here and labelled as untrusted wherever it is shown to the model.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError

MAX_PAGE_BYTES = 600_000       # read at most this much of a page
PAGE_TIMEOUT_S = 4.0           # per-request socket timeout
TOTAL_DEADLINE_S = 6.0         # overall time allowed for reading all pages of one search
EXCERPT_CHARS = 1200           # relevant text kept per page
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
_DROP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe", "button")
_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was", "what", "whats", "who", "how", "and", "or",
              "me", "my", "tell", "about", "today", "now", "current", "currently", "latest", "please", "with", "at", "it"}


def is_public_http_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves to public addresses (never localhost, LAN or link-local).

    Ambiguous URLs are refused outright. Python and the fetching libraries disagree about where the host ends when a URL
    holds a backslash or a user:password part ("http://127.0.0.1:8000\\@public.example/"): Python sees the public host,
    the fetcher connects to 127.0.0.1. Known limit: the name is resolved here and again when connecting, so a DNS
    rebinding host that alternates answers is not caught.
    """
    if any(char == "\\" or char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url):
        return False
    try:
        parts = urllib.parse.urlparse(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or "@" in parts.netloc:
            return False
        parts.port                                    # raises ValueError for a malformed port
    except ValueError:
        return False
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)}
    except OSError:
        return False
    return bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)


class _SafeRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse redirects to non-public addresses (a public page cannot bounce the server onto your network)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_public_http_url(newurl):
            raise URLError("redirect to a non-public address blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode(raw: bytes, content_type: str) -> str:
    """Decode page bytes using the declared charset, falling back to UTF-8 with replacement."""
    match = re.search(r"charset=([\w-]+)", content_type, re.I)
    try:
        return raw.decode(match.group(1) if match else "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def page_blocks(html: str) -> list[str]:
    """Split an HTML document into readable text blocks (paragraphs, list items, headings, table cells)."""
    import lxml.html
    try:
        tree = lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return []
    for bad in tree.xpath("//" + "|//".join(_DROP_TAGS)):
        bad.drop_tree()
    blocks, seen = [], set()
    for node in tree.xpath("//h1|//h2|//h3|//p|//li|//td|//dd|//blockquote|//div[not(.//div) and not(.//p)]|//span[not(.//span)]"):
        text = re.sub(r"\s+", " ", node.text_content()).strip()
        if len(text) >= 25 and text not in seen:
            seen.add(text)
            blocks.append(text)
    return blocks


def best_passages(blocks: list[str], query: str, limit: int = EXCERPT_CHARS) -> str:
    """Pick the blocks that best match the query words (numbers get a bonus: prices, scores, temperatures)."""
    words = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if word not in _STOPWORDS and len(word) > 1}
    scored = []
    for position, block in enumerate(blocks):
        lowered = block.lower()
        hits = sum(1 for word in words if word in lowered)
        score = hits * 3 + (1 if re.search(r"\d", block) else 0) - position * 0.002
        if hits:
            scored.append((score, position, block))
    chosen, used = [], 0
    for _, position, block in sorted(scored, reverse=True):
        if used + len(block) > limit and chosen:
            continue
        chosen.append((position, block[:limit]))
        used += len(block)
        if used >= limit:
            break
    return " ".join(block for _, block in sorted(chosen))


MAX_REDIRECTS = 5


def _fetch_markdown(url: str, client=None) -> str:
    """GET `url` with a browser-like TLS client, following redirects ourselves so every hop must be a public address.

    The client library follows redirects by default with no address check, which would let a public page bounce this
    server onto localhost or the LAN. Returns "" for a blocked hop, a non-200 answer or any error.
    """
    if client is None:
        import primp
        client = primp.Client(impersonate="random", impersonate_os="random", timeout=PAGE_TIMEOUT_S, follow_redirects=False)
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not is_public_http_url(current):
            return ""
        response = client.get(current, follow_redirects=False)
        if 300 <= response.status_code < 400:
            location = response.headers.get("location") or response.headers.get("Location")
            if not location:
                return ""
            current = urllib.parse.urljoin(current, location)
            continue
        return response.text_markdown[:MAX_PAGE_BYTES] if response.status_code == 200 else ""
    return ""        # too many redirects


def read_page_with_ddgs(url: str, query: str, client=None) -> str:
    """Second reader: a browser-like TLS client (the one ddgs uses), which gets past some blocks plain HTTP cannot."""
    try:
        content = _fetch_markdown(url, client)
    except Exception:
        return ""
    blocks = [line.strip() for line in re.split(r"\n+", content or "") if len(line.strip()) >= 25]
    return best_passages(blocks, query)


def read_page(url: str, query: str) -> str:
    """Passages of `url` relevant to `query`: plain HTTP first, then ddgs's extractor if that got nothing ("" if both fail)."""
    if not is_public_http_url(url):
        return ""
    return read_page_http(url, query) or read_page_with_ddgs(url, query)


def read_page_http(url: str, query: str) -> str:
    """Plain-HTTP reader (see `read_page`); "" if blocked, not HTML, slow or non-public."""
    opener = urllib.request.build_opener(_SafeRedirects)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    try:
        with opener.open(request, timeout=PAGE_TIMEOUT_S) as page:
            if "html" not in page.headers.get("Content-Type", "html").lower():
                return ""
            raw = page.read(MAX_PAGE_BYTES)
            html = _decode(raw, page.headers.get("Content-Type", ""))
    except Exception:
        return ""
    return best_passages(page_blocks(html), query)


def read_pages(rows: list[dict], query: str, count: int = 3) -> list[str]:
    """Read the first `count` result pages in parallel within one overall deadline; "" for pages that failed."""
    urls = [row.get("url", "") for row in rows[:count]]
    excerpts = [""] * len(urls)
    pool = ThreadPoolExecutor(max_workers=max(1, len(urls)), thread_name_prefix="page")
    futures = [pool.submit(read_page, url, query) for url in urls]
    deadline = time.monotonic() + TOTAL_DEADLINE_S
    for index, future in enumerate(futures):
        try:
            excerpts[index] = future.result(timeout=max(0.05, deadline - time.monotonic()))
        except Exception:
            pass
    pool.shutdown(wait=False, cancel_futures=True)
    return excerpts
