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

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            value = " ".join(data.split())
            if value:
                self.parts.append(value)


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebResult] = []
        self._href: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_result = False
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "") or ""
        if tag == "a" and "result-link" in classes:
            self._href = attrs_dict.get("href")
            self._title, self._snippet = [], []
            self._in_result = True
            self._in_title = True
        elif self._in_result and tag in {"div", "a"} and "result-snippet" in classes:
            self._in_title = False
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if self._in_result and tag == "a" and self._in_title:
            self._in_title = False
        if self._in_result and tag == "div" and self._in_snippet:
            self._in_snippet = False
            if self._href and self._title:
                self.results.append(WebResult(" ".join(self._title), self._href, " ".join(self._snippet)))
            self._href = None
            self._in_result = False

    def handle_data(self, data: str) -> None:
        text = " ".join(html.unescape(data).split())
        if self._in_title and text:
            self._title.append(text)
        elif self._in_snippet and text:
            self._snippet.append(text)


class WebEvidenceClient:
    """Small stdlib-only web client. Retrieved text is evidence, never executable control."""

    def __init__(self, timeout: float = 8.0, max_bytes: int = 1_000_000) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.headers = {"User-Agent": "ARCA-llm/0.1 (+local cognitive research)"}

    def _get(self, url: str) -> str:
        request = Request(url, headers=self.headers)
        with urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get("content-type", "")
            if content_type and not any(x in content_type for x in ("text/", "json", "xml")):
                raise ValueError(f"unsupported content type: {content_type}")
            return response.read(self.max_bytes).decode("utf-8", errors="replace")

    def fetch(self, url: str) -> WebResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("only absolute http(s) URLs are allowed")
        raw = self._get(url)
        parser = _TextParser()
        parser.feed(raw)
        text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
        return WebResult(parsed.netloc, url, text[:400], text, source=url)

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        if not query.strip():
            return []
        raw = self._get("https://lite.duckduckgo.com/lite/?q=" + quote_plus(query))
        parser = _SearchParser()
        parser.feed(raw)
        results = []
        for result in parser.results[:limit]:
            results.append(WebResult(result.title, urljoin("https://lite.duckduckgo.com", result.url), result.snippet, source="duckduckgo"))
        return results
