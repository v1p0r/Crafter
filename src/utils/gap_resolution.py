"""Evidence and appendix helpers for trustworthy gap resolution."""

from __future__ import annotations

import html
import hashlib
import re
import unicodedata
from typing import Any

from src.utils.paper_content import item_to_text


APPENDIX_TEXT_LIMIT = 40_000
ACTIONABLE_STATUSES = frozenset({"resolved", "partial"})


def _clean_section_label(value: str) -> str:
    cleaned = str(value or "").strip()
    cleaned = re.sub(r"^(?:appendix|section)\s+", "", cleaned, flags=re.IGNORECASE)
    match = re.match(r"^([A-Za-z](?:\.\d+)*)", cleaned)
    return match.group(1).casefold() if match else ""


def _section_label(title: str) -> str:
    cleaned = re.sub(r"^(?:appendix|section)\s+", "", str(title or "").strip(), flags=re.IGNORECASE)
    match = re.match(r"^([A-Za-z](?:\.\d+)*)(?=\s|$)", cleaned)
    return match.group(1).casefold() if match else ""


def aggregate_appendix_sections(
    internal_ref: str,
    partitions: list[dict],
    *,
    max_chars: int = APPENDIX_TEXT_LIMIT,
) -> dict[str, Any] | None:
    """Return one appendix section together with all of its descendants.

    For example, ``Appendix A`` includes A, A.1, A.2, ... while ``B.2``
    includes B.2, B.2.1, ... but not B.3. Table HTML is retained because the
    values needed to close a gap are often absent from the table caption.
    """
    label = _clean_section_label(internal_ref)
    if not label:
        return None

    appendix_sections = [
        section for section in partitions
        if isinstance(section, dict) and section.get("section_role") == "appendix"
    ]
    matched = [
        section for section in appendix_sections
        if (_section_label(section.get("title", "")) == label
            or _section_label(section.get("title", "")).startswith(label + "."))
    ]

    # Compatibility fallback for older partitions whose appendix titles were
    # not normalized but still begin with the requested label.
    if not matched:
        for section in appendix_sections:
            title = str(section.get("title", "")).lstrip().casefold()
            if title.startswith(label):
                next_character = title[len(label):len(label) + 1]
                if next_character in ("", ".", " "):
                    matched.append(section)

    if not matched:
        return None

    chunks: list[str] = []
    titles: list[str] = []
    for section in matched:
        rendered = "\n".join(
            text
            for text in (
                item_to_text(item, include_table_html=True)
                for item in section.get("items", [])
            )
            if text
        ).strip()
        if not rendered:
            continue
        titles.append(str(section.get("title", "")))
        chunks.append(rendered)

    if not chunks:
        return None

    complete_text = "\n\n".join(chunks)
    internal_text = complete_text[:max_chars]
    return {
        "title": titles[0] if titles else f"Appendix {label.upper()}",
        "source_sections": titles,
        "internal_text": internal_text,
        "content_sha256": hashlib.sha256(internal_text.encode("utf-8")).hexdigest(),
        "text_truncated": len(complete_text) > len(internal_text),
        "from_appendix": True,
    }


def _normalized_evidence_text(value: object) -> str:
    """Canonicalize presentation markup without semantic fuzzy matching."""
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    replacements = {
        "×": " times ", "∈": " in ", "≈": " approx ",
        "γ": " gamma ", "κ": " kappa ", "≤": " le ", "≥": " ge ",
        "→": " to ", "−": "-", "–": "-", "—": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    latex_commands = {
        "times": "times", "in": "in", "approx": "approx",
        "gamma": "gamma", "kappa": "kappa", "leq": "le", "geq": "ge",
        "rightarrow": "to", "to": "to",
        "text": "", "mathrm": "", "mathbf": "", "mathit": "",
        "operatorname": "", "left": "", "right": "", "hat": "",
        "bar": "",
    }
    text = re.sub(
        r"\\([A-Za-z]+)",
        lambda match: f" {latex_commands.get(match.group(1), match.group(1))} ",
        text,
    )
    # OCR occasionally inserts spaces between every digit in displayed math
    # (e.g. ``1 2 8`` for ``128``). This changes presentation only.
    text = re.sub(
        r"(?<!\d)(\d(?:\s+\d)+)(?!\d)",
        lambda match: match.group(0).replace(" ", ""),
        text,
    )
    text = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE)
    return " ".join(text.split())


def evidence_quote_is_present(quote: object, source_text: str) -> bool:
    """Check that a purported exact quote is substantively present in source."""
    normalized_quote = _normalized_evidence_text(quote)
    normalized_source = _normalized_evidence_text(source_text)
    return len(normalized_quote) >= 12 and normalized_quote in normalized_source


def apply_appendix_judgment(gap_result: dict, judgment: dict | None) -> dict:
    """Apply a validator judgment, failing closed on missing/fabricated evidence."""
    result = dict(gap_result)
    candidate = result.get("source_candidate") or {}
    source_text = str(candidate.get("internal_text") or "")
    judgment = judgment if isinstance(judgment, dict) else {}
    requested_status = str(judgment.get("status") or "insufficient").casefold()
    raw_evidence = judgment.get("evidence") or []
    if isinstance(raw_evidence, (str, dict)):
        raw_evidence = [raw_evidence]

    evidence: list[dict] = []
    invalid_evidence: list[dict] = []
    for item in raw_evidence:
        if isinstance(item, dict):
            quote = str(item.get("quote") or "")
            supports = str(item.get("supports") or "")
        else:
            quote = str(item or "")
            supports = ""
        if evidence_quote_is_present(quote, source_text):
            evidence.append({"quote": quote, "supports": supports})
        elif quote:
            invalid_evidence.append({"quote": quote, "supports": supports})

    covered = [str(item) for item in (judgment.get("covered_needs") or []) if item]
    missing = [str(item) for item in (judgment.get("missing_needs") or []) if item]
    has_verified_evidence = bool(evidence)
    if invalid_evidence:
        # A model may return several quotes and slightly rewrite one of them.
        # Keep only mechanically verified support and downgrade completeness.
        verified_supports = [item["supports"] for item in evidence if item.get("supports")]
        if verified_supports:
            covered = verified_supports
        for item in invalid_evidence:
            unsupported = item.get("supports") or "one claimed implementation detail"
            note = f"unverified evidence: {unsupported}"
            if note not in missing:
                missing.append(note)

    if (
        requested_status == "resolved"
        and has_verified_evidence
        and not missing
        and not invalid_evidence
    ):
        status = "resolved"
        coverage = "complete"
    elif requested_status in {"resolved", "partial"} and has_verified_evidence:
        status = "partial"
        coverage = "partial"
    else:
        status = "source_found"
        coverage = "insufficient"

    result["resolution_status"] = status
    result["resolution_evidence"] = {
        "validated": status in ACTIONABLE_STATUSES,
        "validator": "minimax_evidence_gate",
        "requested_status": requested_status,
        "source_type": candidate.get("source_type", "appendix"),
        "coverage": coverage,
        "evidence": evidence,
        "covered_needs": covered,
        "missing_needs": missing,
        "invalid_evidence_quotes": [item["quote"] for item in invalid_evidence],
        "invalid_evidence": invalid_evidence,
        "rationale": str(judgment.get("rationale") or ""),
    }
    if status in ACTIONABLE_STATUSES:
        result["resolution"] = candidate
        if status == "partial":
            result["unresolved_note"] = "Appendix provides verified partial evidence; material details remain missing"
    else:
        result.pop("resolution", None)
        result["unresolved_note"] = "Appendix source found, but it did not pass the implementation-evidence gate"
    return result


def validated_registry_result(gap: dict, resolution: dict) -> dict:
    """Build a resolved result for registry metadata within its declared scope."""
    return {
        **gap,
        "resolution": resolution,
        "resolution_status": "resolved",
        "resolution_evidence": {
            "validated": True,
            "validator": "dataset_registry_capability_gate",
            "source_type": "dataset_registry",
            "coverage": "complete",
            "evidence": [{
                "registry_key": resolution.get("dataset_key"),
                "matched_alias": resolution.get("matched_alias"),
                "supports": resolution.get("resolution_scope"),
            }],
            "covered_needs": ["canonical dataset registry metadata"],
            "missing_needs": [],
        },
        "tier": 0,
        "confidence": "high",
    }


def is_gap_actionable(gap: dict) -> bool:
    evidence = gap.get("resolution_evidence") or {}
    return (
        gap.get("resolution_status") in ACTIONABLE_STATUSES
        and evidence.get("validated") is True
        and bool(gap.get("resolution"))
    )


def is_gap_resolved(gap: dict) -> bool:
    evidence = gap.get("resolution_evidence") or {}
    return (
        gap.get("resolution_status") == "resolved"
        and evidence.get("validated") is True
        and evidence.get("coverage") == "complete"
        and bool(gap.get("resolution"))
    )
