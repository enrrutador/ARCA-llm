from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen


@dataclass(slots=True)
class WebResult:
    title: str
    url: str
    snippet: str
    text: str = ""
    source: str = "web"


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


class WebEvidenceClient:
    """Bounded HTTP evidence client. Retrieved text is never executable."""

    def __init__(self, timeout: float = 8.0, max_bytes: int = 1_000_000):
        self.timeout, self.max_bytes = timeout, max_bytes
        self.headers = {"User-Agent": "ARCA-llm/0.3"}

    def _get(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("only absolute http(s) URLs are allowed")
        with urlopen(Request(url, headers=self.headers), timeout=self.timeout) as response:
            content_type = response.headers.get("content-type", "")
            if content_type and not any(x in content_type for x in ("text/", "json", "xml")):
                raise ValueError(f"unsupported content type: {content_type}")
            return response.read(self.max_bytes).decode("utf-8", errors="replace")

    def fetch(self, url: str) -> WebResult:
        raw = self._get(url)
        parser = _TextParser(); parser.feed(raw)
        text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
        return WebResult(urlparse(url).netloc, url, text[:400], text, urlparse(url).netloc)

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        if not query.strip():
            return []
        raw = self._get("https://lite.duckduckgo.com/lite/?q=" + quote_plus(query))
        # DuckDuckGo Lite exposes result links as ordinary anchors. Keep this
        # intentionally conservative: failed parsing means no evidence, not guesses.
        links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.I | re.S)
        results: list[WebResult] = []
        for href, title_html in links:
            title = re.sub(r"<[^>]+>", " ", html.unescape(title_html)).strip()
            if not title or "duckduckgo.com" in href:
                continue
            url = urljoin("https://lite.duckduckgo.com", href)
            results.append(WebResult(title, url, title, source="duckduckgo"))
            if len(results) >= limit:
                break
        return results
