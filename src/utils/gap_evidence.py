"""Deterministic evidence spans for target-paper gap resolution."""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.paper_content import item_to_text


DEFAULT_SPAN_CHARS = 600
REPOSITORY_SPAN_CHARS = 900
REPOSITORY_MAX_FILE_BYTES = 512_000
REPOSITORY_NOTEBOOK_MAX_FILE_BYTES = 8_000_000
REPOSITORY_TEXT_SUFFIXES = frozenset({
    ".ipynb", ".json", ".jsonl", ".md", ".py", ".rst", ".sh", ".toml", ".txt",
    ".yaml", ".yml",
})
REPOSITORY_EXCLUDED_PARTS = frozenset({
    ".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__",
    "build", "dist", "historical_trajectories", "node_modules", "results",
    "run_metrics", "trajectories",
})

_RETRIEVAL_STOPWORDS = {
    "a", "all", "also", "an", "and", "any", "are", "as", "at", "be",
    "by", "complete", "detail", "details", "did", "do", "exact", "for",
    "from", "full", "how", "if", "implementation", "in", "including",
    "information", "into", "is", "it", "method", "of", "on", "or", "other",
    "paper", "parameter", "parameters", "provide", "required", "specific",
    "that", "the", "their", "these", "this", "to", "used", "using", "was",
    "were", "what", "when", "whether", "which", "with",
}


def _is_references_section(section: dict) -> bool:
    role = section.get("section_role")
    if role is not None:
        return role == "references"
    title = str(section.get("title") or "").casefold()
    return any(word in title for word in ("references", "bibliography"))


def _plain_source_text(value: str) -> str:
    """Render table HTML as row text while leaving scientific content intact."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"</t[dh]\s*>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    words = piece.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and current_length + added > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_evidence_text(text: str, max_chars: int) -> list[str]:
    """Split into small auditable spans, preferring rows and sentences."""
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])\b", line)
        for piece in pieces:
            piece = piece.strip()
            if len(piece) < 12:
                if chunks and len(chunks[-1]) + len(piece) + 1 <= max_chars:
                    chunks[-1] += " " + piece
                continue
            chunks.extend(_split_long_piece(piece, max_chars))
    return chunks


def build_paper_evidence_spans(
    partitions: list[dict],
    *,
    max_chars: int = DEFAULT_SPAN_CHARS,
    include_references: bool = False,
) -> list[dict[str, Any]]:
    """Create stable sentence/row spans from parsed paper content.

    Bibliography entries are excluded by default because a citation is not
    implementation evidence. IDs are deterministic for a fixed partition file.
    """
    spans: list[dict[str, Any]] = []
    for section_index, section in enumerate(partitions):
        if not isinstance(section, dict):
            continue
        if not include_references and _is_references_section(section):
            continue
        section_title = str(section.get("title") or "")
        section_role = str(section.get("section_role") or "")
        for item_index, item in enumerate(section.get("items", []) or []):
            if not isinstance(item, dict):
                continue
            rendered = item_to_text(item, include_table_html=True)
            plain = _plain_source_text(rendered)
            for chunk_index, chunk in enumerate(_split_evidence_text(plain, max_chars)):
                spans.append({
                    "span_id": f"s{section_index:03d}_i{item_index:03d}_c{chunk_index:02d}",
                    "section_index": section_index,
                    "section_title": section_title,
                    "section_role": section_role,
                    "item_type": item.get("type"),
                    "text": chunk,
                })
    return add_contextual_evidence_spans(spans, max_chars=max_chars * 2)


_PAPER_SPAN_ID_RE = re.compile(
    r"^(?P<prefix>.*?s(?P<section>\d{3}))_i(?P<item>\d{3})_c(?P<chunk>\d{2})$"
)


def add_contextual_evidence_spans(
    spans: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_SPAN_CHARS * 2,
) -> list[dict[str, Any]]:
    """Add bounded cross-item spans when parsing split one logical statement.

    PDF/HTML figures sometimes interrupt a sentence (for example, ``These DNAs
    are directly compared`` ... figure ... ``with the query embeddings``).
    Equations and their immediately following explanation are likewise much
    easier to audit together.  The added span contains only exact source text,
    records its component span IDs, never crosses a paper section/source, and
    leaves all original stable spans untouched.
    """
    original = [dict(span) for span in spans]
    grouped: dict[tuple[str, str, str], list[tuple[int, int, dict[str, Any]]]] = {}
    for span in original:
        if span.get("component_span_ids"):
            continue
        match = _PAPER_SPAN_ID_RE.fullmatch(str(span.get("span_id") or ""))
        if not match:
            continue
        if str(span.get("item_type") or "").casefold() in {
            "figure", "image", "title",
        }:
            continue
        key = (
            match.group("prefix"),
            str(span.get("source_type") or ""),
            str(span.get("repository") or ""),
        )
        grouped.setdefault(key, []).append((
            int(match.group("item")), int(match.group("chunk")), span,
        ))

    contextual: list[dict[str, Any]] = []
    for (prefix, _, _), rows in grouped.items():
        rows.sort(key=lambda row: (row[0], row[1], str(row[2].get("span_id"))))
        for left, right in zip(rows, rows[1:]):
            left_item, left_chunk, left_span = left
            right_item, right_chunk, right_span = right
            if left_item == right_item:
                continue
            left_text = str(left_span.get("text") or "").strip()
            right_text = str(right_span.get("text") or "").strip()
            if not left_text or not right_text:
                continue
            left_incomplete = not re.search(r"[.!?][\]\)\"']?$", left_text)
            equation_then_explanation = (
                left_text.startswith(("$$", "[Table row]"))
                and re.match(r"^(?:This|That|These|Those|Here|Thus|Hence)\b", right_text)
            )
            right_continues = bool(re.match(
                r"^(?:[a-z]|and\b|or\b|where\b|which\b|with\b)",
                right_text,
            ))
            if not ((left_incomplete and right_continues) or equation_then_explanation):
                continue
            combined = f"{left_text}\n{right_text}"
            if len(combined) > max_chars:
                continue
            contextual.append({
                **left_span,
                "span_id": (
                    f"{prefix}_ctx_i{left_item:03d}c{left_chunk:02d}"
                    f"_i{right_item:03d}c{right_chunk:02d}"
                ),
                "item_type": "context_window",
                "text": combined,
                "component_span_ids": [
                    str(left_span["span_id"]), str(right_span["span_id"]),
                ],
            })
    return original + contextual


def build_repository_evidence_spans(
    repository: str | Path,
    *,
    max_chars: int = REPOSITORY_SPAN_CHARS,
    max_file_bytes: int = REPOSITORY_MAX_FILE_BYTES,
    id_prefix: str = "repo",
) -> list[dict[str, Any]]:
    """Create stable path/line spans from an official source repository.

    Notebook outputs are deliberately excluded: only markdown and code cell
    sources can serve as implementation evidence.  ``id_prefix`` lets callers
    combine several repositories without silently colliding span IDs.
    """
    root = Path(repository)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(id_prefix)).strip("_") or "repo"
    spans: list[dict[str, Any]] = []
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in REPOSITORY_TEXT_SUFFIXES
        and not (set(path.relative_to(root).parts) & REPOSITORY_EXCLUDED_PARTS)
        and path.stat().st_size <= (
            REPOSITORY_NOTEBOOK_MAX_FILE_BYTES
            if path.suffix.casefold() == ".ipynb"
            else max_file_bytes
        )
    )

    def append_text_spans(
        *,
        text: str,
        file_index: int,
        relative: str,
        span_stem: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        current: list[str] = []
        current_start = 1
        current_length = 0
        chunk_index = 0
        lines = text.splitlines()

        def flush(end_line: int) -> None:
            nonlocal current, current_start, current_length, chunk_index
            content = "\n".join(current).strip()
            if len(content) >= 12:
                spans.append({
                    "span_id": f"{safe_prefix}_f{file_index:04d}_{span_stem}c{chunk_index:04d}",
                    "section_title": relative,
                    "source_path": relative,
                    "line_start": current_start,
                    "line_end": end_line,
                    "source_type": "official_repository",
                    "text": content,
                    **(extra or {}),
                })
                chunk_index += 1
            current = []
            current_length = 0

        for line_number, line in enumerate(lines, start=1):
            line = line.rstrip()
            added = len(line) + (1 if current else 0)
            if current and current_length + added > max_chars:
                flush(line_number - 1)
                current_start = line_number
            if len(line) > max_chars:
                if current:
                    flush(line_number - 1)
                for piece in _split_long_piece(line, max_chars):
                    current = [piece]
                    current_start = line_number
                    current_length = len(piece)
                    flush(line_number)
                current_start = line_number + 1
                continue
            if not current:
                current_start = line_number
            current.append(line)
            current_length += added
        if current:
            flush(len(lines))

    for file_index, path in enumerate(files):
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        if path.suffix.casefold() == ".ipynb":
            try:
                notebook = json.loads(text)
            except (TypeError, ValueError):
                continue
            cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
            for cell_index, cell in enumerate(cells):
                if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown"}:
                    continue
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(str(part) for part in source)
                elif not isinstance(source, str):
                    continue
                append_text_spans(
                    text=source,
                    file_index=file_index,
                    relative=relative,
                    span_stem=f"n{cell_index:04d}_",
                    extra={
                        "cell_index": cell_index,
                        "cell_type": cell.get("cell_type"),
                    },
                )
            continue
        append_text_spans(
            text=text,
            file_index=file_index,
            relative=relative,
            span_stem="",
        )
    return spans


def _retrieval_tokens(value: object) -> list[str]:
    original = str(value or "")
    # Preserve mixed-case scientific acronyms before CamelCase splitting.
    # Otherwise ``CoE_R`` becomes only ``co``, ``e``, ``r`` and can no longer
    # match prose spelling ``CoE-R``.  Keep both the compact identifier pieces
    # and their CamelCase-expanded form; Counter/BM25 handles duplicates.
    compact_tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", original)
        if len(token) >= 2
    ]
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", original).casefold()
    for source, target in {
        "τ": "tau", "λ": "lambda", "γ": "gamma", "κ": "kappa",
        "×": "times", "ε": "epsilon",
    }.items():
        text = text.replace(source, f" {target} ")
    expanded_tokens = [
        token for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 2 and token not in _RETRIEVAL_STOPWORDS
    ]
    return expanded_tokens + [
        token for token in compact_tokens
        if token not in _RETRIEVAL_STOPWORDS and token not in expanded_tokens
    ]


def rank_evidence_spans(
    gap: dict,
    spans: list[dict],
    *,
    top_k: int = 40,
) -> list[dict]:
    """Rank target-paper spans with deterministic BM25-like lexical evidence."""
    query_parts = (
        gap.get("target_name"), gap.get("description"), gap.get("what_is_needed"),
        gap.get("context_quote"),
    )
    query_tokens = _retrieval_tokens(" ".join(str(value or "") for value in query_parts))
    query_counts = Counter(query_tokens)
    if not query_counts:
        return []

    document_tokens = [
        _retrieval_tokens(" ".join((
            str(span.get("source_path") or ""),
            str(span.get("section_title") or ""),
            str(span.get("text") or ""),
        )))
        for span in spans
    ]
    document_frequency: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequency.update(set(tokens))
    document_count = max(len(spans), 1)
    average_length = (
        sum(len(tokens) for tokens in document_tokens) / document_count
        if document_tokens else 1
    )

    target_tokens = set(_retrieval_tokens(gap.get("target_name")))
    internal_label = str(gap.get("internal_ref") or "").casefold()
    internal_label = re.sub(r"^(?:appendix|section)\s+", "", internal_label).strip()
    ranked: list[tuple[float, str, dict]] = []
    for span, tokens in zip(spans, document_tokens):
        counts = Counter(tokens)
        length = max(len(tokens), 1)
        score = 0.0
        for token, query_frequency in query_counts.items():
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency.get(token, 0)
            inverse_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            saturation = frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / max(average_length, 1))
            )
            score += inverse_frequency * saturation * min(query_frequency, 3)

        section_tokens = set(_retrieval_tokens(span.get("section_title")))
        text_token_set = set(tokens)
        score += 2.5 * len(target_tokens & section_tokens)
        score += 1.25 * len(target_tokens & text_token_set)
        if target_tokens and target_tokens <= (section_tokens | text_token_set):
            score += 5.0
        if internal_label:
            section_title = str(span.get("section_title") or "").casefold()
            normalized_title = re.sub(r"^(?:appendix|section)\s+", "", section_title).strip()
            if (
                normalized_title == internal_label
                or normalized_title.startswith(internal_label + ".")
                or normalized_title.startswith(internal_label + " ")
            ):
                score += 12.0
        if score > 0:
            ranked.append((score, str(span.get("span_id")), span))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [
        {**span, "retrieval_score": round(score, 6)}
        for score, _, span in ranked[:top_k]
    ]


def gap_retrieval_queries(gap: dict) -> list[str]:
    """Build focused, deterministic queries for each material need in a gap."""
    target = str(gap.get("target_name") or "").strip()
    needed = str(gap.get("what_is_needed") or "").strip()
    queries = [
        target,
        " ".join(str(value) for value in (target, gap.get("description")) if value),
        " ".join(value for value in (target, needed) if value),
    ]
    # Numbered lists and semicolon-delimited requirements are common in the
    # generated gaps.  Ranking them independently prevents one easy need from
    # crowding every other need out of the candidate window.
    atomic_parts = re.split(
        r"\s*(?:;|\n|(?=\(\d+\)\s*)|(?=\d+[.)]\s+))\s*",
        needed,
    )
    for part in atomic_parts:
        part = re.sub(r"^\(?\d+\)?[.)]?\s*", "", part).strip(" ,.;:")
        if part:
            queries.append(part)
            queries.append(" ".join(value for value in (target, part) if value))
    internal_ref = str(gap.get("internal_ref") or "").strip()
    if internal_ref:
        queries.append(" ".join(value for value in (target, internal_ref, needed) if value))
    context = str(gap.get("context_quote") or "").strip()
    if context:
        queries.append(" ".join(value for value in (target, context) if value))

    # Definitions alone rarely establish an implementation choice.  For gaps
    # asking how a method was run, reserve retrieval opportunities for the
    # three places that usually carry the missing concrete evidence: an actual
    # entrypoint/command, configuration defaults, and downstream evaluation or
    # output consumers.  These remain lexical queries (not trusted claims).
    implementation_text = " ".join((
        str(gap.get("category") or ""),
        str(gap.get("description") or ""),
        needed,
    )).casefold()
    implementation_markers = (
        "baseline", "config", "hyperparameter", "implementation", "preprocess",
        "procedure", "setting", "variant",
    )
    if target and any(marker in implementation_text for marker in implementation_markers):
        queries.extend((
            f"{target} main entrypoint run command script CLI arguments",
            f"{target} config default hyperparameter option setting",
            f"{target} evaluation score metric output result consumer",
            f"{target} preprocessing token aggregation pooling data loader",
        ))

    unique: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for query in queries:
        key = tuple(_retrieval_tokens(query))
        if key and key not in seen:
            unique.append(query)
            seen.add(key)
    return unique


def material_coverage_relevance(gap: dict, covered_needs: list[object]) -> int:
    """Score covered rows against the authoritative ``what_is_needed`` text.

    LLM decompositions can inflate a candidate by splitting peripheral context
    (for example, "retail domain confirmed" and "API count") into many rows.
    This bounded lexical score instead rewards rows that overlap the actual
    requested implementation details.  It ranks candidates only; it never
    turns evidence into a validated claim.
    """
    material = set(_retrieval_tokens(gap.get("what_is_needed")))
    if not material:
        return 0
    score = 0
    for need in covered_needs:
        overlap = material & set(_retrieval_tokens(need))
        score += min(len(overlap), 6)
    return score


def rank_evidence_spans_multi_query(
    gap: dict,
    spans: list[dict],
    *,
    top_k: int = 80,
    per_query: int = 16,
    rrf_constant: int = 30,
    extra_queries: list[str] | None = None,
    preferred_paths: list[str] | None = None,
    preferred_sources: list[tuple[str, str]] | None = None,
    neighbor_radius: int = 1,
) -> list[dict]:
    """Fuse per-need rankings so candidate evidence covers the whole gap.

    Reciprocal-rank fusion is deliberately deterministic and does not require
    an embedding service.  It rewards a span that is useful for several needs
    while reserving candidates from every atomic query.
    """
    if top_k <= 0 or per_query <= 0:
        return []
    span_by_id = {str(span.get("span_id")): span for span in spans if span.get("span_id")}
    fused: Counter[str] = Counter()
    best_scores: dict[str, float] = {}
    query_hits: Counter[str] = Counter()
    rankings: list[list[dict]] = []
    queries = gap_retrieval_queries(gap) + [
        str(query) for query in (extra_queries or []) if str(query).strip()
    ]
    unique_queries: list[str] = []
    seen_queries: set[tuple[str, ...]] = set()
    for query in queries:
        key = tuple(_retrieval_tokens(query))
        if key and key not in seen_queries:
            unique_queries.append(query)
            seen_queries.add(key)
    for query in unique_queries:
        query_gap = {
            # ``query`` already includes the target for target-aware variants.
            # Leaving this empty lets the need-only variants rank an exact
            # implementation span above generic README mentions of the target.
            "target_name": "",
            "description": query,
            "what_is_needed": "",
            "context_quote": "",
            "internal_ref": gap.get("internal_ref"),
        }
        ranked = rank_evidence_spans(query_gap, spans, top_k=per_query)
        rankings.append(ranked)
        for rank, span in enumerate(ranked, start=1):
            span_id = str(span["span_id"])
            fused[span_id] += 1.0 / (rrf_constant + rank)
            query_hits[span_id] += 1
            best_scores[span_id] = max(
                best_scores.get(span_id, 0.0), float(span.get("retrieval_score") or 0.0),
            )

    # Reserve most of the window for diverse seed spans, then use the rest for
    # adjacent source context.  Code/config meaning often crosses a chunk
    # boundary (definition followed by invocation, key followed by its value).
    seed_limit = max(1, min(top_k, int(top_k * 0.72)))
    selected_ids: list[str] = []
    selected: set[str] = set()

    # Retrieval planning may identify a likely config/source path even when
    # prose and symbol names differ.  Exact path matching is intentional and
    # fail-closed; fuzzy path matches could reintroduce the original mismatch.
    # Keep this reservation bounded so atomic-query diversity remains intact.
    preferred = {str(path) for path in (preferred_paths or []) if str(path)}
    preferred_pairs = {
        (str(repository), str(path))
        for repository, path in (preferred_sources or [])
        if str(repository) and str(path)
    }
    preferred_budget = (
        min(seed_limit, max(1, seed_limit // 3))
        if preferred or preferred_pairs else 0
    )
    if preferred_budget:
        source_keys: list[tuple[str | None, str]] = [
            (repository, path) for repository, path in sorted(preferred_pairs)
        ]
        source_keys.extend((None, path) for path in sorted(preferred))
        per_path_limit = max(1, min(8, preferred_budget // max(len(source_keys), 1)))
        for repository, path in source_keys:
            candidates = [
                span_id for span_id, span in span_by_id.items()
                if str(span.get("source_path") or "") == path
                and (
                    repository is None
                    or str(span.get("repository") or "") == repository
                )
                and span_id not in selected
            ]
            candidates.sort(key=lambda span_id: (
                -fused.get(span_id, 0.0),
                -query_hits.get(span_id, 0),
                -best_scores.get(span_id, 0.0),
                span_id,
            ))
            for span_id in candidates[:per_path_limit]:
                fused.setdefault(span_id, 0.0)
                query_hits.setdefault(span_id, 0)
                best_scores.setdefault(span_id, 0.0)
                selected_ids.append(span_id)
                selected.add(span_id)
                if len(selected_ids) >= preferred_budget:
                    break
            if len(selected_ids) >= preferred_budget:
                break

    reserve_per_query = max(1, min(4, seed_limit // max(len(rankings), 1)))
    for ranked in rankings:
        for span in ranked[:reserve_per_query]:
            span_id = str(span["span_id"])
            if span_id not in selected:
                selected_ids.append(span_id)
                selected.add(span_id)
                if len(selected_ids) >= seed_limit:
                    break
        if len(selected_ids) >= seed_limit:
            break

    ranked_ids = sorted(
        (span_id for span_id in fused if span_id not in selected),
        key=lambda span_id: (
            -fused[span_id], -query_hits[span_id], -best_scores[span_id], span_id,
        ),
    )

    for span_id in ranked_ids:
        if len(selected_ids) >= seed_limit:
            break
        if span_id not in selected:
            selected_ids.append(span_id)
            selected.add(span_id)

    # Add neighboring spans from the same file (or same paper section), while
    # preserving source order and never crossing repository/file boundaries.
    if neighbor_radius > 0 and len(selected_ids) < top_k:
        index_by_id = {
            str(span.get("span_id")): index
            for index, span in enumerate(spans) if span.get("span_id")
        }

        def same_source(left: dict, right: dict) -> bool:
            left_path = str(left.get("source_path") or "")
            right_path = str(right.get("source_path") or "")
            if left_path or right_path:
                return bool(left_path) and left_path == right_path
            return (
                left.get("section_index") == right.get("section_index")
                and left.get("source_type") == right.get("source_type")
            )

        seeds = list(selected_ids)
        for distance in range(1, neighbor_radius + 1):
            for span_id in seeds:
                index = index_by_id.get(span_id)
                if index is None:
                    continue
                for neighbor_index in (index - distance, index + distance):
                    if not 0 <= neighbor_index < len(spans):
                        continue
                    neighbor = spans[neighbor_index]
                    neighbor_id = str(neighbor.get("span_id") or "")
                    if (
                        neighbor_id
                        and neighbor_id not in selected
                        and same_source(span_by_id[span_id], neighbor)
                    ):
                        selected_ids.append(neighbor_id)
                        selected.add(neighbor_id)
                        fused.setdefault(neighbor_id, 0.0)
                        query_hits.setdefault(neighbor_id, 0)
                        best_scores.setdefault(neighbor_id, 0.0)
                        if len(selected_ids) >= top_k:
                            break
                if len(selected_ids) >= top_k:
                    break
            if len(selected_ids) >= top_k:
                break

    # If neighboring context did not consume the window, fill from the fused
    # global ranking without duplicating earlier reservations.
    if len(selected_ids) < top_k:
        remaining = sorted(
            (span_id for span_id in fused if span_id not in selected),
            key=lambda span_id: (
                -fused[span_id], -query_hits[span_id], -best_scores[span_id], span_id,
            ),
        )
        selected_ids.extend(remaining[:top_k - len(selected_ids)])
    return [
        {
            **span_by_id[span_id],
            "retrieval_score": round(best_scores[span_id], 6),
            "retrieval_fused_score": round(float(fused[span_id]), 8),
            "retrieval_query_hits": query_hits[span_id],
        }
        for span_id in selected_ids
        if span_id in span_by_id
    ]


def select_candidate_spans(
    gaps: list[dict],
    spans: list[dict],
    *,
    per_gap: int = 40,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Return a de-duplicated candidate corpus and per-gap allowed span IDs."""
    span_by_id = {str(span["span_id"]): span for span in spans}
    candidate_ids: dict[str, list[str]] = {}
    union_ids: set[str] = set()
    for gap in gaps:
        ranked = rank_evidence_spans(gap, spans, top_k=per_gap)
        ids = [str(span["span_id"]) for span in ranked]
        candidate_ids[str(gap.get("id"))] = ids
        union_ids.update(ids)
    candidates = [span_by_id[span_id] for span_id in sorted(union_ids)]
    return candidates, candidate_ids


def apply_span_judgment(
    gap: dict,
    judgment: dict | None,
    spans: list[dict],
    *,
    tier: float = 0.25,
    source_type: str = "target_paper",
) -> dict:
    """Apply a span-ID judgment and fail closed on unknown/missing spans."""
    result = dict(gap)
    judgment = judgment if isinstance(judgment, dict) else {}
    requested_status = str(judgment.get("status") or "insufficient").casefold()
    span_by_id = {
        str(span.get("span_id")): span
        for span in spans
        if isinstance(span, dict) and span.get("span_id") and span.get("text")
    }
    raw_evidence = judgment.get("evidence") or []
    if isinstance(raw_evidence, (str, dict)):
        raw_evidence = [raw_evidence]

    evidence: list[dict] = []
    invalid_span_ids: list[str] = []
    for item in raw_evidence:
        if isinstance(item, dict):
            span_id = str(item.get("span_id") or "")
            supports = str(item.get("supports") or "")
        else:
            span_id = str(item or "")
            supports = ""
        span = span_by_id.get(span_id)
        if span:
            evidence_item = {
                "span_id": span_id,
                "quote": span["text"],
                "supports": supports,
                "section_title": span.get("section_title"),
            }
            for provenance_key in (
                "source_type", "source_path", "source_url", "source_sha256",
                "repository", "repository_url", "repository_commit",
                "repository_role", "line_start", "line_end", "component_span_ids",
            ):
                if span.get(provenance_key) is not None:
                    evidence_item[provenance_key] = span[provenance_key]
            evidence.append(evidence_item)
        elif span_id:
            invalid_span_ids.append(span_id)

    covered = [str(value) for value in (judgment.get("covered_needs") or []) if value]
    missing = [str(value) for value in (judgment.get("missing_needs") or []) if value]
    if invalid_span_ids:
        missing.append("validator returned unknown evidence span IDs")

    if requested_status == "resolved" and evidence and not missing and not invalid_span_ids:
        status = "resolved"
        coverage = "complete"
    elif requested_status in {"resolved", "partial"} and evidence:
        status = "partial"
        coverage = "partial"
    else:
        status = "source_found"
        coverage = "insufficient"

    result["resolution_status"] = status
    result["tier"] = tier
    result["confidence"] = "high" if status == "resolved" else "medium"
    result["resolution_evidence"] = {
        "validated": status in {"resolved", "partial"},
        "validator": "minimax_span_gate",
        "requested_status": requested_status,
        "source_type": source_type,
        "coverage": coverage,
        "evidence": evidence,
        "covered_needs": covered,
        "missing_needs": missing,
        "invalid_span_ids": invalid_span_ids,
        "rationale": str(judgment.get("rationale") or ""),
    }
    if status in {"resolved", "partial"}:
        source_sections = list(dict.fromkeys(
            str(item.get("section_title") or "") for item in evidence
            if item.get("section_title")
        ))
        result["resolution"] = {
            "title": "Verified implementation evidence",
            "source_type": source_type,
            "source_sections": source_sections,
            "source_spans": evidence,
            "internal_text": "\n\n".join(item["quote"] for item in evidence),
            "from_target_paper": all(
                str(item.get("repository_role") or item.get("source_type") or "")
                in {"target_paper", "target_paper_official_html"}
                for item in evidence
            ),
        }
        if status == "partial":
            result["unresolved_note"] = "Target paper provides verified partial evidence"
    else:
        result.pop("resolution", None)
        result["unresolved_note"] = "No implementation evidence was validated in the target paper"
    return result


def apply_span_coverage_judgment(
    gap: dict,
    judgment: dict | None,
    spans: list[dict],
    *,
    tier: float = 0.25,
    source_type: str = "target_paper",
) -> dict:
    """Derive resolution status from per-need supported/missing coverage rows."""
    judgment = judgment if isinstance(judgment, dict) else {}
    coverage_rows = judgment.get("coverage") or []
    if isinstance(coverage_rows, dict):
        coverage_rows = [coverage_rows]

    evidence: list[dict] = []
    covered: list[str] = []
    missing: list[str] = []
    for row in coverage_rows:
        if not isinstance(row, dict):
            continue
        need = str(row.get("need") or "unspecified requested detail")
        state = str(row.get("status") or "missing").casefold()
        span_ids = row.get("span_ids") or []
        if isinstance(span_ids, str):
            span_ids = [span_ids]
        span_ids = [str(span_id) for span_id in span_ids if span_id]
        if state == "supported" and span_ids:
            covered.append(need)
            evidence.extend({"span_id": span_id, "supports": need} for span_id in span_ids)
        else:
            missing.append(need)

    # A full claim requires an explicit completeness assertion in addition to
    # every enumerated row being supported. Missing or unknown span IDs will
    # still downgrade it inside apply_span_judgment.
    coverage_complete = judgment.get("coverage_complete") is True and bool(covered) and not missing
    status = "resolved" if coverage_complete else ("partial" if covered else "insufficient")
    translated = {
        "status": status,
        "evidence": evidence,
        "covered_needs": covered,
        "missing_needs": missing,
        "rationale": judgment.get("rationale", ""),
    }
    result = apply_span_judgment(
        gap,
        translated,
        spans,
        tier=tier,
        source_type=source_type,
    )
    result["resolution_evidence"]["coverage_rows"] = coverage_rows
    result["resolution_evidence"]["coverage_complete_asserted"] = (
        judgment.get("coverage_complete") is True
    )
    return result
