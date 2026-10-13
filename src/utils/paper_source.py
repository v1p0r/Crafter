"""Verified official-paper full text used as a gap-resolution fallback.

The normal pipeline consumes MinerU partitions.  Those partitions can be
incomplete (most notably, a PDF parse can stop at the bibliography and omit
appendices).  This module provides a deliberately narrow fallback to arXiv's
official HTML endpoint and converts the HTML into the same lightweight section
shape used by the rest of the pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import aiohttp


logger = logging.getLogger("paper_source")

ARXIV_HTML_BASE_URL = "https://arxiv.org/html"
ARXIV_HTML_MIN_BYTES = 10_000
ARXIV_HTML_MAX_BYTES = 25_000_000
ARXIV_FETCH_RETRY_DELAYS = (2, 5)

_ARXIV_ID_RE = re.compile(
    r"(?<![\d.])(?P<identifier>\d{4}\.\d{4,5})(?:v\d+)?(?![\d.])",
    flags=re.IGNORECASE,
)
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_TAGS = frozenset({"p", "li", "tr", "figcaption", "pre", "blockquote"})
_IGNORED_TAGS = frozenset({"nav", "script", "style", "noscript", "form", "svg", "footer"})


def extract_arxiv_id(value: object) -> str | None:
    """Extract a modern arXiv identifier without accepting arbitrary digits."""
    match = _ARXIV_ID_RE.search(str(value or ""))
    return match.group("identifier") if match else None


def _normalized_text(parts: list[str], *, preserve_lines: bool = False) -> str:
    text = "".join(parts).replace("\xa0", " ")
    if preserve_lines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\|\s*\|", "|", text)
    return text.strip(" |")


def _inline_text(value: str) -> list[dict[str, str]]:
    return [{"type": "text", "content": value}]


def _section_role(title: str, classes: set[str], previous_role: str) -> str:
    normalized = title.strip().casefold()
    if (
        "ltx_title_bibliography" in classes
        or normalized in {"references", "bibliography", "works cited"}
    ):
        return "references"
    if "ltx_title_appendix" in classes or re.match(r"^appendix\s+[a-z]\b", normalized):
        return "appendix"
    if previous_role == "appendix" and re.match(r"^[a-z](?:\.\d+)*\b", normalized):
        return "appendix"
    return "body"


class _ArxivContentParser(HTMLParser):
    """Extract section-scoped prose/table rows while suppressing MathML noise."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.document_title = ""
        self.sections: list[dict[str, Any]] = []
        self.current_section: dict[str, Any] | None = None
        self.current_role = "body"
        self.heading_tag: str | None = None
        self.heading_classes: set[str] = set()
        self.heading_parts: list[str] = []
        self.block_tag: str | None = None
        self.block_parts: list[str] = []
        self.ignored_depth = 0
        self.math_depth = 0
        self.seen_article = False

    @staticmethod
    def _attributes(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key: value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = self._attributes(attrs)
        classes = set(attributes.get("class", "").split())

        if tag == "article":
            self.seen_article = True
        if self.ignored_depth:
            if tag in _IGNORED_TAGS:
                self.ignored_depth += 1
            return
        if tag in _IGNORED_TAGS:
            self.ignored_depth = 1
            return
        if not self.seen_article:
            return

        if self.math_depth:
            self.math_depth += 1
            return
        if tag == "math":
            alttext = attributes.get("alttext", "").strip()
            if alttext:
                self._append_text(f" ${alttext}$ ")
            self.math_depth = 1
            return

        if tag in _HEADING_TAGS:
            self._finish_block()
            self.heading_tag = tag
            self.heading_classes = classes
            self.heading_parts = []
            return
        if tag in _BLOCK_TAGS and self.block_tag is None and self.heading_tag is None:
            self.block_tag = tag
            self.block_parts = []
            return
        if tag == "br":
            self._append_text("\n")
        elif tag in {"td", "th"} and self.block_tag == "tr" and self.block_parts:
            self.block_parts.append(" | ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in {"br", "img", "meta", "link", "hr", "input"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.ignored_depth:
            if tag in _IGNORED_TAGS:
                self.ignored_depth -= 1
            return
        if not self.seen_article:
            return
        if self.math_depth:
            self.math_depth -= 1
            return
        if self.heading_tag == tag:
            self._finish_heading()
            return
        if self.block_tag == tag:
            self._finish_block()
        if tag == "article":
            self._finish_block()
            self.seen_article = False

    def handle_data(self, data: str) -> None:
        if self.ignored_depth or self.math_depth or not self.seen_article:
            return
        self._append_text(data)

    def close(self) -> None:
        super().close()
        self._finish_block()
        if self.heading_tag:
            self._finish_heading()

    def _append_text(self, value: str) -> None:
        if self.heading_tag:
            self.heading_parts.append(value)
        elif self.block_tag:
            self.block_parts.append(value)

    def _finish_heading(self) -> None:
        tag = self.heading_tag
        title = _normalized_text(self.heading_parts)
        classes = self.heading_classes
        self.heading_tag = None
        self.heading_classes = set()
        self.heading_parts = []
        if not title:
            return
        if tag == "h1" and ("ltx_title_document" in classes or not self.document_title):
            self.document_title = title
            return
        role = _section_role(title, classes, self.current_role)
        self.current_role = role
        self.current_section = {
            "title": title,
            "section_role": role,
            "items": [],
        }
        self.sections.append(self.current_section)

    def _finish_block(self) -> None:
        tag = self.block_tag
        text = _normalized_text(self.block_parts, preserve_lines=tag == "pre")
        self.block_tag = None
        self.block_parts = []
        if len(text) < 2:
            return
        # Some valid arXiv conversions render the document title as the first
        # centered/bold article paragraph instead of ``h1.ltx_title_document``.
        # Capture only that first pre-section paragraph. Callers still verify
        # it against expected title metadata before trusting the source.
        if not self.document_title and self.current_section is None and tag == "p":
            if 8 <= len(text) <= 500:
                self.document_title = text
                return
        if self.current_section is None:
            self.current_section = {
                "title": "Preamble",
                "section_role": "body",
                "items": [],
            }
            self.sections.append(self.current_section)
        if tag == "tr":
            text = f"[Table row] {text}"
        elif tag == "figcaption":
            text = f"[Figure caption] {text}"
        elif tag == "pre":
            text = f"[Code]\n{text}"
        self.current_section["items"].append({
            "type": "paragraph",
            "content": {"paragraph_content": _inline_text(text)},
        })


def parse_arxiv_html(html_text: str) -> dict[str, Any]:
    """Parse arXiv HTML into a title and MinerU-compatible sections."""
    parser = _ArxivContentParser()
    parser.feed(str(html_text or ""))
    parser.close()
    return {"title": parser.document_title, "partitions": parser.sections}


def validate_arxiv_html(raw: bytes, *, expected_sha256: str | None = None) -> tuple[str, str]:
    """Perform format-aware integrity checks and return decoded text + digest."""
    if not ARXIV_HTML_MIN_BYTES <= len(raw) <= ARXIV_HTML_MAX_BYTES:
        raise ValueError(f"arXiv HTML has invalid size: {len(raw)} bytes")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("cached arXiv HTML checksum mismatch")
    text = raw.decode("utf-8")
    lowered = text.casefold()
    if "<html" not in lowered or "</html>" not in lowered or "<article" not in lowered:
        raise ValueError("arXiv response is not a complete article HTML document")
    parsed = parse_arxiv_html(text)
    if not parsed["title"] or not parsed["partitions"]:
        raise ValueError("arXiv HTML contains no parseable title or paper sections")
    return text, digest


def _read_cached_html(cache_dir: Path, arxiv_id: str, url: str) -> dict[str, Any] | None:
    html_path = cache_dir / "paper.html"
    metadata_path = cache_dir / "metadata.json"
    if (
        cache_dir.is_symlink()
        or html_path.is_symlink()
        or metadata_path.is_symlink()
        or not html_path.is_file()
        or not metadata_path.is_file()
    ):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("arxiv_id") != arxiv_id or metadata.get("source_url") != url:
            return None
        text, digest = validate_arxiv_html(
            html_path.read_bytes(), expected_sha256=str(metadata.get("sha256") or ""),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, TypeError):
        return None
    return {
        "arxiv_id": arxiv_id,
        "source_url": url,
        "sha256": digest,
        "html": text,
        "cache_path": str(html_path),
        "from_cache": True,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError(f"refusing to write through symlinked cache directory: {path.parent}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


async def fetch_arxiv_html(arxiv_id: str, cache_root: str | Path) -> dict[str, Any] | None:
    """Fetch and cache verified official arXiv HTML, failing closed on errors."""
    clean_id = extract_arxiv_id(arxiv_id)
    if clean_id != arxiv_id:
        return None
    url = f"{ARXIV_HTML_BASE_URL}/{clean_id}"
    cache_dir = Path(cache_root) / clean_id
    cached = _read_cached_html(cache_dir, clean_id, url)
    if cached:
        return cached

    last_error: Exception | None = None
    for attempt in range(len(ARXIV_FETCH_RETRY_DELAYS) + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=45)
            headers = {"User-Agent": "ptc-gap-agent/1.0 (official-paper fallback)"}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status != 200:
                        raise ValueError(f"arXiv returned HTTP {response.status}")
                    raw = await response.read()
            text, digest = validate_arxiv_html(raw)
            parsed = parse_arxiv_html(text)
            metadata = {
                "schema_version": 1,
                "arxiv_id": clean_id,
                "source_url": url,
                "sha256": digest,
                "size_bytes": len(raw),
                "title": parsed["title"],
            }
            _atomic_write(cache_dir / "paper.html", raw)
            _atomic_write(
                cache_dir / "metadata.json",
                (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
            )
            return {
                **metadata,
                "html": text,
                "cache_path": str(cache_dir / "paper.html"),
                "from_cache": False,
            }
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, UnicodeError, ValueError) as exc:
            last_error = exc
            if attempt < len(ARXIV_FETCH_RETRY_DELAYS):
                await asyncio.sleep(ARXIV_FETCH_RETRY_DELAYS[attempt])
    logger.warning("Official arXiv HTML fetch failed closed for %s: %s", clean_id, last_error)
    return None
