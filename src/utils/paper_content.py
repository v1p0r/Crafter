"""Shared utilities for working with MinerU-parsed paper content and gap context.

Consolidates `_item_to_text` and `_extract_inline` — previously duplicated across
gap_agent, planning_agent, and sub_planning_agent with inconsistent item-type
coverage (the reason the gap agent initially couldn't see the references list).
"""

from __future__ import annotations


def extract_inline(content_list: list[dict]) -> str:
    """Extract plain text from an inline content list (text + equation_inline)."""
    parts = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(item.get("content", ""))
        elif item.get("type") == "equation_inline":
            parts.append(f"${item.get('content', '')}$")
    return "".join(parts)


def item_to_text(item: dict, include_table_html: bool = False) -> str:
    """Render a single MinerU content item as plain text.

    Covers every item type observed in MinerU output: title, paragraph,
    equation_interline, algorithm, table, image, list (reference lists).

    Set ``include_table_html=True`` to embed the table's HTML after the caption
    (needed by planning/sub-planning agents that want table contents, not the
    gap agent which only needs the caption).
    """
    item_type = item.get("type")
    content = item.get("content", {}) or {}

    if item_type == "title":
        return "### " + extract_inline(content.get("title_content", []))
    if item_type == "paragraph":
        return extract_inline(content.get("paragraph_content", []))
    if item_type == "equation_interline":
        return f"$${content.get('math_content', '')}$$"
    if item_type == "algorithm":
        caption = extract_inline(content.get("algorithm_caption", []))
        algo = extract_inline(content.get("algorithm_content", []))
        return f"[Algorithm: {caption}]\n{algo}"
    if item_type == "table":
        caption = extract_inline(content.get("table_caption", []))
        if include_table_html:
            html = content.get("html", "")
            return f"[Table: {caption}]\n{html}"
        return f"[Table: {caption}]"
    if item_type == "image":
        caption = extract_inline(content.get("image_caption", []))
        return f"[Figure: {caption}]"
    if item_type == "list":
        lines = []
        for li in content.get("list_items", []) or []:
            if not isinstance(li, dict):
                continue
            text = extract_inline(li.get("item_content", []))
            if text.strip():
                lines.append(f"- {text}")
        return "\n".join(lines)
    return ""


def format_gaps_for_context(
    gaps: list[dict],
    header: str = "Supplementary context from referenced papers:",
    abstract_chars: int = 400,
) -> str:
    """Format a list of resolved gaps as a context block for an LLM prompt.

    Each gap appears as ``### {category}: {description}`` followed by TLDR
    and (truncated) abstract from its resolution. Returns an empty string if
    the gaps list is empty (so callers can concatenate unconditionally).
    """
    # Fail closed for legacy caches and source-only retrievals. A non-empty
    # ``resolution`` field is insufficient without an explicit evidence gate.
    gaps = [
        gap for gap in gaps
        if gap.get("resolution_status") in {"resolved", "partial"}
        and (gap.get("resolution_evidence") or {}).get("validated") is True
        and gap.get("resolution")
    ]
    if not gaps:
        return ""
    lines = [f"\n\n{header}"]
    for g in gaps:
        res = g.get("resolution") or {}
        status = g.get("resolution_status", "")
        lines.append(
            f"\n### {g.get('category', '')} [{status}]: {g.get('description', '')}"
        )
        evidence = g.get("resolution_evidence") or {}
        evidence_items = evidence.get("evidence") or []
        for item in evidence_items:
            if isinstance(item, dict) and item.get("quote"):
                lines.append(
                    f"Verified evidence ({item.get('supports', 'requested detail')}): "
                    f"{item['quote']}"
                )
        missing = evidence.get("missing_needs") or []
        if missing:
            lines.append("Still missing: " + "; ".join(str(item) for item in missing))
        tldr = res.get("tldr")
        if tldr:
            lines.append(f"TLDR: {tldr}")
        abstract = res.get("abstract")
        if abstract:
            lines.append(f"Abstract: {abstract[:abstract_chars]}")
        # Appendix-based (internal) resolutions carry the source text directly.
        internal_text = res.get("internal_text")
        if internal_text:
            title = res.get("title", "")
            label = f" ({title})" if title else ""
            lines.append(f"From appendix{label}: {internal_text[:abstract_chars]}")
        if res.get("from_registry"):
            lines.append(f"Registry source: {res.get('source', '')}")
            if res.get("standard_transforms"):
                lines.append(f"Standard transforms: {res['standard_transforms']}")
    return "\n".join(lines)
