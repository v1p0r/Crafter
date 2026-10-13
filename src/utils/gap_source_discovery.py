"""Discover referenced implementation repositories directly from paper sources.

The discovery path is deliberately conservative:

* only GitHub repository URLs that occur in the target/reference paper are
  considered;
* caller-supplied target-paper codebase URLs are blacklisted before scope
  matching, cloning, checkout validation, or source inspection;
* repositories classified as target implementations are never materialized,
  even when the explicit blacklist is incomplete;
* every URL must be bound to the target paper or to one exact referenced gap;
* a checkout is treated as untrusted data and is never executed;
* accepted repositories are clean, symlink-free Git checkouts whose origin and
  recorded commit are written to a relocatable manifest.

This module does not perform fuzzy GitHub search.  When a paper contains no
repository link, discovery fails closed instead of guessing a same-named repo.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse


MAX_AUTO_REPOSITORIES = 12
MAX_DISCOVERY_CANDIDATES = 32
DEFAULT_CLONE_TIMEOUT_SECONDS = 180

_GITHUB_TEXT_RE = re.compile(
    r"(?:(?:https?://)?(?:www\.)?github\.com)\s*/\s*"
    r"(?P<owner>[A-Za-z0-9_.-]+)\s*/\s*(?P<repo>[A-Za-z0-9_.-]+)",
    flags=re.IGNORECASE,
)
_CODE_MARKER_RE = re.compile(
    r"\b(?:our\s+code|source\s+code|code\s+(?:is\s+)?(?:publicly\s+)?available|"
    r"official\s+(?:code|implementation)|implementation\s+(?:is\s+)?available|"
    r"github\s+repository|codebase|repository)\b",
    flags=re.IGNORECASE,
)
_REFERENCE_TITLES = frozenset({"references", "bibliography", "works cited"})
_IGNORED_HTML_TAGS = frozenset({
    "footer", "form", "nav", "noscript", "script", "style", "svg",
})
_HTML_BLOCK_TAGS = frozenset({"blockquote", "figcaption", "li", "p", "pre", "tr"})
_HTML_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_TITLE_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "large", "language", "model", "models", "of", "on", "the",
    "to", "via", "with",
})
_IDENTITY_FILES = (
    "README.md", "README.rst", "README.txt", "README", "CITATION.cff",
    "CITATION.bib", "pyproject.toml", "setup.py",
)


def _normalized_words(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _significant_words(value: object) -> set[str]:
    return {
        word for word in _normalized_words(value)
        if len(word) >= 2 and word not in _TITLE_STOPWORDS
    }


def _compact(value: object) -> str:
    return "".join(_normalized_words(value))


def canonical_github_repository_url(value: object) -> str:
    """Return an HTTPS repository-root URL, rejecting org/user-only links."""
    raw = html.unescape(unquote(str(value or ""))).strip()
    raw = re.sub(r"\s*/\s*", "/", raw)
    if raw.startswith("git@github.com:"):
        raw = "https://github.com/" + raw.split(":", 1)[1]
    if re.match(r"^(?:www\.)?github\.com/", raw, flags=re.IGNORECASE):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return ""
    if (parsed.hostname or "").casefold() not in {"github.com", "www.github.com"}:
        return ""
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return ""
    owner = re.sub(r"[^A-Za-z0-9_.-]", "", parts[0]).strip(".")
    repo = re.sub(r"[^A-Za-z0-9_.-]", "", parts[1]).strip(".")
    if repo.casefold().endswith(".git"):
        repo = repo[:-4].strip(".")
    if not owner or not repo:
        return ""
    # These are GitHub site routes, not repository owners.
    if owner.casefold() in {
        "about", "apps", "collections", "customer-stories", "enterprise",
        "events", "explore", "features", "issues", "marketplace", "new",
        "notifications", "orgs", "pricing", "pulls", "search", "settings",
        "sponsors", "topics", "trending",
    }:
        return ""
    return f"https://github.com/{owner}/{repo}"


def github_repository_identity(value: object) -> str:
    canonical = canonical_github_repository_url(value)
    if not canonical:
        return ""
    parts = urlparse(canonical).path.strip("/").split("/")
    return "/".join(part.casefold() for part in parts[:2])


def normalize_github_repository_blacklist(values: Iterable[object]) -> list[str]:
    """Canonicalize an explicit repository blacklist or reject invalid entries."""
    if isinstance(values, (str, bytes)):
        values = [values]
    normalized_by_identity: dict[str, str] = {}
    invalid: list[str] = []
    for value in values:
        canonical = canonical_github_repository_url(value)
        identity = github_repository_identity(canonical)
        if not canonical or not identity:
            invalid.append(str(value))
            continue
        normalized_by_identity.setdefault(
            identity, f"https://github.com/{identity}"
        )
    if invalid:
        raise ValueError(
            "invalid_target_codebase_blacklist_url:" + ",".join(invalid)
        )
    return [normalized_by_identity[key] for key in sorted(normalized_by_identity)]


def extract_github_repository_urls(text: object) -> list[str]:
    """Extract repository roots, tolerating common PDF/OCR spaces around `/`."""
    value = html.unescape(str(text or ""))
    urls: list[str] = []
    seen: set[str] = set()
    for match in _GITHUB_TEXT_RE.finditer(value):
        candidate = canonical_github_repository_url(
            f"https://github.com/{match.group('owner')}/{match.group('repo')}"
        )
        identity = github_repository_identity(candidate)
        if candidate and identity and identity not in seen:
            urls.append(candidate)
            seen.add(identity)
    return urls


def _nested_text(value: object) -> str:
    """Read text from any MinerU item, including footnotes and page footers."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_nested_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_nested_text(item) for item in value.values())
    return ""


def _is_reference_section(section: dict[str, Any]) -> bool:
    role = str(section.get("section_role") or "").casefold()
    if role:
        return role == "references"
    title = re.sub(r"^\d+(?:\.\d+)*\s+", "", str(section.get("title") or "")).strip()
    return title.casefold() in _REFERENCE_TITLES


def repository_candidates_from_partitions(
    partitions: list[dict[str, Any]],
    *,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract auditable GitHub candidates from MinerU-style partitions."""
    candidates: list[dict[str, Any]] = []
    for section_index, section in enumerate(partitions):
        if not isinstance(section, dict):
            continue
        items = section.get("items") or []
        units: list[tuple[int, int | None, str, str]] = []
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "list":
                list_items = (item.get("content") or {}).get("list_items") or []
                for list_index, list_item in enumerate(list_items):
                    text = re.sub(r"\s+", " ", _nested_text(list_item)).strip()
                    if text:
                        units.append((item_index, list_index, item_type, text))
            else:
                text = re.sub(r"\s+", " ", _nested_text(item)).strip()
                if text:
                    units.append((item_index, None, item_type, text))
        is_references = _is_reference_section(section)
        for unit_index, (item_index, list_index, item_type, text) in enumerate(units):
            urls = extract_github_repository_urls(text)
            if not urls:
                continue
            if is_references:
                context = text
            else:
                # Footnote URLs are often separated from the sentence that
                # introduces them. One neighboring item on either side is a
                # bounded amount of context and does not span whole sections.
                start = max(0, unit_index - 1)
                end = min(len(units), unit_index + 2)
                context = " ".join(unit[3] for unit in units[start:end])
            for url in urls:
                suffix = f"/list/{list_index:04d}" if list_index is not None else ""
                candidates.append({
                    "url": url,
                    "identity": github_repository_identity(url),
                    "anchor_text": "",
                    "link_context": text[:2000],
                    "item_type": item_type,
                    "context": context[:4000],
                    "section_title": str(section.get("title") or ""),
                    "section_role": "references" if is_references else "body",
                    "source_path": (
                        f"partition/{section_index:03d}/item/{item_index:04d}{suffix}"
                    ),
                    **source,
                })
    return candidates


class _ArticleLinkParser(HTMLParser):
    """Collect GitHub links and containing-block context inside `<article>`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_article = False
        self.ignored_depth = 0
        self.current_section = ""
        self.heading_tag: str | None = None
        self.heading_parts: list[str] = []
        self.block_tag: str | None = None
        self.block_parts: list[str] = []
        self.block_links: list[tuple[str, str]] = []
        self.anchor_href: str | None = None
        self.anchor_parts: list[str] = []
        self.links: list[dict[str, str]] = []

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.casefold(): value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag == "article":
            self.in_article = True
            return
        if not self.in_article:
            return
        if self.ignored_depth:
            if tag in _IGNORED_HTML_TAGS:
                self.ignored_depth += 1
            return
        if tag in _IGNORED_HTML_TAGS:
            self.ignored_depth = 1
            return
        if tag in _HTML_HEADING_TAGS:
            self._finish_block()
            self.heading_tag = tag
            self.heading_parts = []
            return
        if tag in _HTML_BLOCK_TAGS and self.block_tag is None:
            self.block_tag = tag
            self.block_parts = []
            self.block_links = []
            return
        if tag == "a":
            href = self._attrs(attrs).get("href", "")
            if canonical_github_repository_url(href):
                self.anchor_href = href
                self.anchor_parts = []

    def handle_data(self, data: str) -> None:
        if not self.in_article or self.ignored_depth:
            return
        if self.heading_tag:
            self.heading_parts.append(data)
        if self.block_tag:
            self.block_parts.append(data)
        if self.anchor_href is not None:
            self.anchor_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if not self.in_article:
            return
        if self.ignored_depth:
            if tag in _IGNORED_HTML_TAGS:
                self.ignored_depth -= 1
            return
        if tag == "a" and self.anchor_href is not None:
            self.block_links.append((self.anchor_href, "".join(self.anchor_parts).strip()))
            self.anchor_href = None
            self.anchor_parts = []
            return
        if self.heading_tag == tag:
            self.current_section = re.sub(r"\s+", " ", "".join(self.heading_parts)).strip()
            self.heading_tag = None
            self.heading_parts = []
            return
        if self.block_tag == tag:
            self._finish_block()
        if tag == "article":
            self._finish_block()
            self.in_article = False

    def close(self) -> None:
        super().close()
        self._finish_block()

    def _finish_block(self) -> None:
        if not self.block_tag:
            return
        context = re.sub(r"\s+", " ", "".join(self.block_parts)).strip()
        # Visible URLs are needed when the conversion did not preserve an href.
        visible = [(url, "") for url in extract_github_repository_urls(context)]
        for href, anchor in self.block_links + visible:
            url = canonical_github_repository_url(href)
            if url:
                self.links.append({
                    "url": url,
                    "anchor_text": anchor,
                    "context": context[:4000],
                    "section_title": self.current_section,
                    "section_role": (
                        "references"
                        if re.sub(r"^\d+(?:\.\d+)*\s+", "", self.current_section)
                        .strip().casefold() in _REFERENCE_TITLES
                        else "body"
                    ),
                })
        self.block_tag = None
        self.block_parts = []
        self.block_links = []


def repository_candidates_from_arxiv_html(
    html_text: str,
    *,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract links only from the paper article, never arXiv site footers."""
    parser = _ArticleLinkParser()
    parser.feed(str(html_text or ""))
    parser.close()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(parser.links):
        identity = github_repository_identity(item["url"])
        key = (identity, item["context"], item["section_title"])
        if not identity or key in seen:
            continue
        seen.add(key)
        candidates.append({
            **item,
            "identity": identity,
            "source_path": f"official_html/link/{index:04d}",
            **source,
        })
    return candidates


def _reference_associations(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for gap in gaps:
        referenced = gap.get("referenced_paper") or {}
        title = str(referenced.get("title") or "").strip()
        arxiv_id = str(
            (gap.get("source_candidate") or {}).get("arxiv_id")
            or referenced.get("arxiv_id") or ""
        ).strip()
        if not title:
            continue
        key = (_compact(title), arxiv_id.casefold())
        row = grouped.setdefault(key, {
            "role": "referenced_implementation",
            "paper_title": title,
            "arxiv_id": arxiv_id,
            "ref_ids": [],
            "paper_titles": [title],
            "target_names": [],
        })
        ref_id = str(gap.get("ref_id") or "").strip()
        target_name = str(gap.get("target_name") or "").strip()
        if ref_id and ref_id not in row["ref_ids"]:
            row["ref_ids"].append(ref_id)
        if target_name and target_name not in row["target_names"]:
            row["target_names"].append(target_name)
    return list(grouped.values())


def _association_score(candidate: dict[str, Any], association: dict[str, Any]) -> tuple[int, list[str]]:
    broad_context = " ".join((
        str(candidate.get("context") or ""),
        str(candidate.get("anchor_text") or ""),
        str(candidate.get("section_title") or ""),
    ))
    association_role = str(association.get("role") or "")
    # Neighboring body text helps recover a target-repository footnote, but it
    # must not bind that URL to a merely adjacent cited method. References need
    # to match the link's own sentence/list row.
    context = (
        " ".join((
            str(candidate.get("link_context") or candidate.get("context") or ""),
            str(candidate.get("anchor_text") or ""),
            str(candidate.get("section_title") or ""),
        ))
        if association_role == "referenced_implementation"
        else broad_context
    )
    normalized_context = " ".join(_normalized_words(context))
    title = str(association.get("paper_title") or "")
    normalized_title = " ".join(_normalized_words(title))
    title_words = _significant_words(title)
    context_words = _significant_words(context)
    identity = str(candidate.get("identity") or "")
    repo_name = identity.split("/", 1)[-1]
    score = 0
    reasons: list[str] = []

    if normalized_title and normalized_title in normalized_context:
        score += 10
        reasons.append("exact_paper_title_in_link_context")
    elif title_words:
        overlap = title_words & context_words
        coverage = len(overlap) / len(title_words)
        if len(overlap) >= 3 and coverage >= 0.6:
            score += 6
            reasons.append("strong_paper_title_context_overlap")
        elif len(overlap) >= 2 and coverage >= 0.35:
            score += 3
            reasons.append("paper_title_context_overlap")

    arxiv_id = str(association.get("arxiv_id") or "").casefold()
    if arxiv_id and arxiv_id in context.casefold():
        score += 10
        reasons.append("arxiv_id_in_link_context")

    for target_name in association.get("target_names") or []:
        normalized_target = " ".join(_normalized_words(target_name))
        if normalized_target and normalized_target in normalized_context:
            score += 6
            reasons.append("target_name_in_link_context")
            break
        if _compact(target_name) and _compact(target_name) == _compact(repo_name):
            score += 5
            reasons.append("target_name_matches_repository")
            break

    if _compact(repo_name) and _compact(repo_name) in _compact(title):
        score += 4
        reasons.append("repository_name_matches_paper_title")

    marker = bool(_CODE_MARKER_RE.search(context))
    if marker:
        score += 4
        reasons.append("implementation_link_context")

    document_role = str(candidate.get("document_role") or "")
    section_role = str(candidate.get("section_role") or "")
    if (
        document_role == "target_paper"
        and association_role == "target_implementation"
        and section_role != "references"
    ):
        # A body/footnote link is worth inspecting, but is not enough for final
        # acceptance without repository-side identity evidence.
        score += 2
        reasons.append("direct_target_paper_body_link")
    return score, reasons


def associate_repository_candidates(
    candidates: list[dict[str, Any]],
    *,
    gaps: list[dict[str, Any]],
    target_title: str,
    target_arxiv_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind each paper link to one explicit target/reference scope."""
    target = {
        "role": "target_implementation",
        "paper_title": target_title,
        "arxiv_id": target_arxiv_id,
        "ref_ids": [],
        "paper_titles": [],
        "target_names": [],
    }
    references = _reference_associations(gaps)
    associated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates[:MAX_DISCOVERY_CANDIDATES]:
        document_role = str(candidate.get("document_role") or "target_paper")
        if document_role == "referenced_paper":
            source_title = _compact(candidate.get("paper_title"))
            source_arxiv = str(candidate.get("arxiv_id") or "").casefold()
            options = [
                row for row in references
                if (source_title and _compact(row.get("paper_title")) == source_title)
                or (source_arxiv and str(row.get("arxiv_id") or "").casefold() == source_arxiv)
            ]
        else:
            options = [target, *references]
        scored = []
        for option in options:
            score, reasons = _association_score(candidate, option)
            scored.append((score, _compact(option.get("paper_title")), option, reasons))
        scored.sort(key=lambda row: (-row[0], row[1]))
        if not scored or scored[0][0] < 2:
            rejected.append({
                "url": candidate.get("url"),
                "source_path": candidate.get("source_path"),
                "reason": "no_paper_or_gap_scope_match",
            })
            continue
        best = scored[0]
        if len(scored) > 1 and scored[1][0] == best[0] and scored[1][1] != best[1]:
            rejected.append({
                "url": candidate.get("url"),
                "source_path": candidate.get("source_path"),
                "reason": "ambiguous_paper_scope",
                "score": best[0],
            })
            continue
        associated.append({
            **candidate,
            "association": best[2],
            "paper_link_score": best[0],
            "paper_link_reasons": best[3],
        })

    # Keep the strongest occurrence of one repository/scope pair. Repeated
    # footer links otherwise cause needless clone/validation work.
    best_by_scope: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in associated:
        association = item["association"]
        key = (
            str(item.get("identity") or ""),
            str(association.get("role") or ""),
            _compact(association.get("paper_title")),
        )
        current = best_by_scope.get(key)
        if current is None or int(item["paper_link_score"]) > int(current["paper_link_score"]):
            best_by_scope[key] = item
    return list(best_by_scope.values()), rejected


def _git_value(repository: Path, *args: str, timeout: int = 30) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _validate_checkout(repository: Path, expected_url: str) -> dict[str, str]:
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("repository_missing_or_symlink")
    if any(path.is_symlink() for path in repository.rglob("*")):
        raise ValueError("repository_contains_symlink")
    commit = _git_value(repository, "rev-parse", "HEAD")
    tree = _git_value(repository, "rev-parse", "HEAD^{tree}")
    status = _git_value(repository, "status", "--porcelain", "--untracked-files=all")
    origin = _git_value(repository, "remote", "get-url", "origin")
    if status:
        raise ValueError("repository_not_clean")
    if github_repository_identity(origin) != github_repository_identity(expected_url):
        raise ValueError("origin_url_mismatch")
    _git_value(repository, "fsck", "--connectivity-only", "--no-dangling", timeout=60)
    return {"commit": commit, "tree": tree, "origin": origin}


def _clone_or_reuse_repository(
    url: str,
    destination: Path,
    *,
    timeout: int,
) -> tuple[dict[str, str], bool]:
    """Materialize a normal directory without replacing an existing checkout."""
    if destination.exists() or destination.is_symlink():
        return _validate_checkout(destination, url), True
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("repository_root_is_symlink")
    temporary_root = Path(tempfile.mkdtemp(prefix=".paper-repo-", dir=destination.parent))
    checkout = temporary_root / "checkout"
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--single-branch", "--no-tags",
                url, str(checkout),
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        metadata = _validate_checkout(checkout, url)
        if destination.exists() or destination.is_symlink():
            raise ValueError("repository_destination_appeared_during_clone")
        checkout.replace(destination)
        return metadata, False
    finally:
        # This removes only the uniquely named temporary directory created
        # immediately above; accepted checkouts have already been moved out.
        shutil.rmtree(temporary_root, ignore_errors=True)


def _repository_identity_text(repository: Path) -> str:
    parts: list[str] = []
    for filename in _IDENTITY_FILES:
        path = repository / filename
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)[:4_000_000]


def _repository_match_score(
    repository: Path,
    candidate: dict[str, Any],
) -> tuple[int, list[str]]:
    association = candidate["association"]
    identity_text = _repository_identity_text(repository)
    normalized_docs = " ".join(_normalized_words(identity_text))
    title = str(association.get("paper_title") or "")
    normalized_title = " ".join(_normalized_words(title))
    title_words = _significant_words(title)
    doc_words = _significant_words(identity_text)
    score = 0
    reasons: list[str] = []
    if normalized_title and normalized_title in normalized_docs:
        score += 10
        reasons.append("exact_paper_title_in_repository_metadata")
    elif title_words:
        overlap = title_words & doc_words
        coverage = len(overlap) / len(title_words)
        if len(overlap) >= 3 and coverage >= 0.6:
            score += 6
            reasons.append("strong_paper_title_repository_overlap")
        elif len(overlap) >= 2 and coverage >= 0.4:
            score += 3
            reasons.append("paper_title_repository_overlap")
    arxiv_id = str(association.get("arxiv_id") or "").casefold()
    if arxiv_id and arxiv_id in identity_text.casefold():
        score += 10
        reasons.append("arxiv_id_in_repository_metadata")
    repo_name = str(candidate.get("identity") or "").split("/", 1)[-1]
    for target_name in association.get("target_names") or []:
        compact_target = _compact(target_name)
        if compact_target and (
            compact_target == _compact(repo_name) or compact_target in _compact(identity_text)
        ):
            score += 5
            reasons.append("target_name_matches_repository")
            break
    if _compact(repo_name) and _compact(repo_name) in _compact(title):
        score += 4
        reasons.append("repository_name_matches_paper_title")
    return score, reasons


def _repository_slug(identity: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", identity.replace("/", "--")).strip("_")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("refusing_to_write_manifest_through_symlink")
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def discover_repository_manifest(
    *,
    output_dir: str | Path,
    target_partitions: list[dict[str, Any]],
    target_title: str,
    gaps: list[dict[str, Any]],
    target_codebase_blacklist: Iterable[str],
    target_arxiv_id: str = "",
    html_documents: list[dict[str, Any]] | None = None,
    excluded_repository_urls: Iterable[str] = (),
    clone_timeout: int = DEFAULT_CLONE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Discover, verify, clone, and manifest non-target repositories."""
    normalized_blacklist = normalize_github_repository_blacklist(
        target_codebase_blacklist
    )
    if not normalized_blacklist:
        raise ValueError("target_codebase_blacklist_required")
    blacklisted_identities = {
        github_repository_identity(url) for url in normalized_blacklist
    }
    output_root = Path(output_dir)
    source_root = output_root / "gap_sources"
    repository_root = source_root / "repositories"
    manifest_path = source_root / "auto_gap_sources.json"
    target_source = {
        "document_role": "target_paper",
        "paper_title": target_title,
        "arxiv_id": target_arxiv_id,
        "source_url": "local_partitions",
        "source_sha256": "",
    }
    candidates = repository_candidates_from_partitions(
        target_partitions, source=target_source,
    )
    documents_used = [{
        "document_role": "target_paper",
        "paper_title": target_title,
        "arxiv_id": target_arxiv_id,
        "source_url": "local_partitions",
        "candidate_count": len(candidates),
    }]
    for document in html_documents or []:
        html_text = str(document.get("html") or "")
        if not html_text and document.get("html_path"):
            path = Path(str(document["html_path"]))
            if path.is_symlink() or not path.is_file():
                continue
            try:
                html_text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
        source = {
            key: document.get(key, "") for key in (
                "document_role", "paper_title", "arxiv_id", "source_url", "source_sha256",
            )
        }
        extracted = repository_candidates_from_arxiv_html(html_text, source=source)
        candidates.extend(extracted)
        documents_used.append({**source, "candidate_count": len(extracted)})

    # Exact URL identity de-duplication happens before scope matching, but the
    # same URL may retain several contexts until the strongest scope is known.
    unique_candidates: list[dict[str, Any]] = []
    seen_candidates: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (
            str(candidate.get("identity") or ""),
            str(candidate.get("document_role") or ""),
            str(candidate.get("source_path") or ""),
        )
        if key not in seen_candidates:
            unique_candidates.append(candidate)
            seen_candidates.add(key)

    # This is the primary contamination boundary. A blacklisted identity is
    # removed before association so a target repository cannot evade the rule
    # by being misclassified as a referenced implementation. No checkout path
    # is touched for these candidates.
    eligible_candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in unique_candidates:
        identity = str(candidate.get("identity") or "")
        if identity in blacklisted_identities:
            rejected.append({
                "url": candidate.get("url"),
                "identity": identity,
                "source_path": candidate.get("source_path"),
                "reason": "target_codebase_blacklisted",
                "blocked_before": "scope_matching_and_clone",
            })
        else:
            eligible_candidates.append(candidate)

    associated, scope_rejected = associate_repository_candidates(
        eligible_candidates,
        gaps=gaps,
        target_title=target_title,
        target_arxiv_id=target_arxiv_id,
    )
    rejected.extend(scope_rejected)
    excluded = {
        github_repository_identity(url) for url in excluded_repository_urls
        if github_repository_identity(url)
    }
    definitions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    accepted_identities: set[str] = set()
    for candidate in sorted(
        associated,
        key=lambda item: (-int(item.get("paper_link_score") or 0), str(item.get("identity"))),
    ):
        identity = str(candidate.get("identity") or "")
        association = candidate["association"]
        if str(association.get("role") or "") == "target_implementation":
            rejected.append({
                "url": candidate.get("url"),
                "identity": identity,
                "source_path": candidate.get("source_path"),
                "reason": "target_implementation_forbidden",
                "blocked_before": "clone",
            })
            continue
        if identity in excluded:
            rejected.append({"url": candidate.get("url"), "reason": "already_configured"})
            continue
        if identity in accepted_identities:
            rejected.append({"url": candidate.get("url"), "reason": "duplicate_repository"})
            continue
        if len(definitions) >= MAX_AUTO_REPOSITORIES:
            rejected.append({"url": candidate.get("url"), "reason": "repository_limit"})
            continue
        slug = _repository_slug(identity)
        destination = repository_root / slug
        try:
            git_metadata, reused = _clone_or_reuse_repository(
                str(candidate["url"]), destination, timeout=clone_timeout,
            )
            repository_score, repository_reasons = _repository_match_score(destination, candidate)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            rejected.append({
                "url": candidate.get("url"),
                "reason": "clone_or_validation_failed",
                "detail": str(exc)[:300],
            })
            continue
        paper_score = int(candidate.get("paper_link_score") or 0)
        accepted_match = (
            paper_score >= 8
            or (paper_score >= 2 and repository_score >= 6)
            or (paper_score >= 5 and repository_score >= 4)
        )
        if not accepted_match:
            rejected.append({
                "url": candidate.get("url"),
                "reason": "repository_identity_not_verified",
                "paper_link_score": paper_score,
                "repository_score": repository_score,
            })
            continue
        relative_path = destination.relative_to(source_root).as_posix()
        definition = {
            "name": slug,
            "role": association["role"],
            "path": relative_path,
            "url": candidate["url"],
            "commit": git_metadata["commit"],
            "ref_ids": association.get("ref_ids") or [],
            "paper_titles": association.get("paper_titles") or [],
            "target_names": association.get("target_names") or [],
            "discovery": {
                "source_url": candidate.get("source_url"),
                "source_sha256": candidate.get("source_sha256"),
                "source_path": candidate.get("source_path"),
                "section_title": candidate.get("section_title"),
                "context": str(candidate.get("context") or "")[:1200],
                "paper_link_score": paper_score,
                "paper_link_reasons": candidate.get("paper_link_reasons") or [],
                "repository_score": repository_score,
                "repository_reasons": repository_reasons,
            },
        }
        definitions.append(definition)
        accepted.append({
            "identity": identity,
            "url": candidate["url"],
            "role": association["role"],
            "paper_title": association.get("paper_title"),
            "commit": git_metadata["commit"],
            "tree": git_metadata["tree"],
            "reused": reused,
            "paper_link_score": paper_score,
            "repository_score": repository_score,
        })
        accepted_identities.add(identity)

    payload = {
        "schema_version": 1,
        "generated_by": "paper_repository_discovery",
        "repositories": definitions,
        "discovery": {
            "repository_access_policy": {
                "schema_version": 1,
                "target_codebase_blacklist": normalized_blacklist,
                "target_codebase_identities": sorted(blacklisted_identities),
                "enforced_before": [
                    "scope_matching",
                    "clone",
                    "checkout_validation",
                    "source_inspection"
                ],
                "target_implementations_allowed": False,
            },
            "documents": documents_used,
            "candidate_count": len(unique_candidates),
            "associated_count": len(associated),
            "accepted_count": len(accepted),
            "rejected": rejected,
        },
    }
    _atomic_json(manifest_path, payload)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "candidate_count": len(unique_candidates),
        "associated_count": len(associated),
        "accepted": accepted,
        "rejected": rejected,
        "documents": documents_used,
        "repository_access_policy": payload["discovery"]["repository_access_policy"],
    }
