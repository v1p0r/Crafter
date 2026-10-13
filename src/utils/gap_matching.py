"""Pure matching and routing helpers for gap analysis.

Keeping these helpers independent of LLM clients makes their conservative
matching rules cheap and deterministic to test.
"""

import copy
import re
from difflib import SequenceMatcher

from src.utils.dataset_registry import find_datasets_in_text


# Fallback routing only. Explicit file ownership, requirement IDs, and target
# names take precedence when the corresponding metadata is present.
GAP_TARGET_KEYWORDS = {
    "baseline_method": ("baseline",),
    "dataset_detail": ("data", "dataset", "loader"),
    "referenced_architecture": (
        "model", "architecture", "network", "encoder", "backbone",
    ),
    "missing_hyperparameter": (
        "config", "train", "experiment", "model", "method", "optimizer", "runner",
    ),
    "algorithm_gap": (
        "algorithm", "experiment", "method", "model", "objective", "loss", "solver",
    ),
}

# A static dataset registry can supply canonical source URLs, standard split
# sizes, class counts, and standard preprocessing. It cannot establish
# paper-specific choices such as exact subsets, prompts, seeds, or protocols.
_UNSUPPORTED_REGISTRY_REQUESTS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bannotat(?:e|ed|ion|ions|ing)\b",
    r"\b(?:judge|prompt template|exact prompt)\b",
    r"\b(?:random seed|random sampl(?:e|es|ing))\b",
    r"\b(?:filtering|selection) criteria\b",
    r"\b(?:which|exact|specific) (?:version|subset|split|text field)\b",
    r"\b(?:api|tool signature|stopping criteria)\b",
    r"\b(?:pair construction|preference pair|binariz\w*|routing accuracy)\b",
    r"\b(?:evaluation|validation) protocol\b",
))


def _normalized_identifier(value: object) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _normalized_requirement_id(value: object) -> str:
    text = str(value).casefold()
    text = re.sub(r"\bequation\b", "eq", text)
    text = re.sub(r"\bfigure\b", "fig", text)
    text = re.sub(r"\balgorithm\b", "alg", text)
    text = re.sub(r"\btable\b", "tab", text)
    return _normalized_identifier(text)


_REFERENCE_PATTERNS = (
    ("eq", re.compile(r"\b(?:eq(?:uation)?)\.?\s*\(?([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)\)?", re.IGNORECASE)),
    ("fig", re.compile(r"\b(?:fig(?:ure)?)\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("tab", re.compile(r"\b(?:tab(?:le)?)\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("alg", re.compile(r"\b(?:alg(?:orithm)?)\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("sec", re.compile(r"(?:\bsection\s*|§\s*)([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("def", re.compile(r"\b(?:def(?:inition)?)\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("thm", re.compile(r"\b(?:thm|theorem)\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("prop", re.compile(r"\b(?:prop(?:osition)?)\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
    ("lemma", re.compile(r"\blemma\.?\s*([a-z](?:\.\d+)+|[a-z]?\d+(?:\.\d+)*)", re.IGNORECASE)),
)

_GENERIC_TARGET_TOKENS = {
    "additional", "algorithm", "benchmark", "complete", "configuration",
    "data", "dataset", "detail", "details", "evaluation", "exact", "full",
    "environment", "implementation", "mechanism", "method", "methodology",
    "missing", "paper", "parameter", "parameters", "pipeline", "prior",
    "selection", "setup", "specific", "specification", "specifications",
    "training", "used", "work",
}


def _canonical_requirement_ids(values: object) -> set[str]:
    identifiers: set[str] = set()
    for value in _string_list(values):
        for prefix, pattern in _REFERENCE_PATTERNS:
            identifiers.update(
                prefix + match.group(1).casefold().lstrip("0")
                for match in pattern.finditer(value)
            )
    return identifiers


def _target_tokens(value: object) -> set[str]:
    text = str(value or "").replace("τ", "tau").replace("Τ", "tau")
    raw_chunks = re.findall(r"[A-Za-z0-9]+", text)
    tokens: set[str] = set()
    for chunk in raw_chunks:
        tokens.add(chunk.casefold())
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", chunk)
        tokens.update(part.casefold() for part in split.split())
    return {
        token for token in tokens
        if len(token) >= 3 and token not in _GENERIC_TARGET_TOKENS
    }


def _tokens_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) >= 5:
        prefix_length = 0
        for left_char, right_char in zip(left, right):
            if left_char != right_char:
                break
            prefix_length += 1
        if prefix_length >= 6:
            return True
        if SequenceMatcher(None, left, right).ratio() >= 0.86:
            return True
    return False


def _matched_target_tokens(target_name: object, target: str) -> set[str]:
    wanted = _target_tokens(target_name)
    available = _target_tokens(target)
    return {
        token for token in wanted
        if any(_tokens_equivalent(token, candidate) for candidate in available)
    }


def _normalized_file_path(value: object) -> str:
    return str(value or "").replace("\\", "/").removeprefix("./").casefold()


def _string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item)
    return ()


def gap_relevant_to_target(
    gap: dict,
    target: str,
    extra_keywords: tuple[str, ...] = (),
    *,
    requirement_ids: tuple[str, ...] = (),
    target_files: tuple[str, ...] = (),
) -> bool:
    """Return whether a resolved gap applies to a module or generated file.

    Routing is ordered from strongest to weakest evidence: explicit file
    ownership, requirement-ID overlap, target-name overlap, then category
    keywords. Optional fields are compatible with a future requirements stage.
    """
    # Once planning has attached target_files, including an explicit empty
    # list, it is authoritative. This prevents broad category fallback from
    # reintroducing routing noise.
    if "target_files" in gap:
        explicit_files = _string_list(gap.get("target_files"))
        candidate_files = target_files or (target.split(maxsplit=1)[0],)
        normalized_candidates = {_normalized_file_path(path) for path in candidate_files}
        return any(_normalized_file_path(path) in normalized_candidates for path in explicit_files)

    gap_requirement_ids = _canonical_requirement_ids(gap.get("requirement_ids"))
    target_requirement_ids = _canonical_requirement_ids(requirement_ids)
    if gap_requirement_ids and target_requirement_ids:
        if gap_requirement_ids & target_requirement_ids:
            return True

    if _matched_target_tokens(gap.get("target_name"), target):
        return True

    # Category-only matching was responsible for most broad injections. Keep
    # the argument for API compatibility, but require an explicit opt-in.
    if gap.get("allow_category_fallback") is True:
        target_lower = target.casefold()
        category = gap.get("category", "")
        keywords = GAP_TARGET_KEYWORDS.get(category, ()) + extra_keywords
        return any(keyword.casefold() in target_lower for keyword in keywords)
    return False


def route_gap_to_architecture(
    gap: dict,
    architecture: dict,
    *,
    max_files: int = 4,
) -> tuple[list[str], list[dict]]:
    """Map one gap to concrete non-orchestrator files with auditable reasons."""
    gap_requirements = _canonical_requirement_ids(gap.get("requirement_ids"))
    target_tokens = _target_tokens(gap.get("target_name"))
    category = str(gap.get("category") or "")
    requirement_candidates: list[tuple[float, str, dict]] = []
    semantic_candidates: list[tuple[float, str, dict]] = []

    for file_spec in architecture.get("files", []):
        if not isinstance(file_spec, dict) or file_spec.get("orchestrator"):
            continue
        path = str(file_spec.get("path") or "")
        if not path:
            continue
        purpose = str(file_spec.get("purpose") or "")
        target = f"{path} {purpose}"
        file_requirements = _canonical_requirement_ids(file_spec.get("implements"))
        requirement_overlap = sorted(gap_requirements & file_requirements)
        matched_tokens = sorted(_matched_target_tokens(gap.get("target_name"), target))
        matched_path_tokens = sorted(_matched_target_tokens(gap.get("target_name"), path))
        matched_purpose_tokens = sorted(_matched_target_tokens(gap.get("target_name"), purpose))

        role_bonus = 0.0
        path_lower = path.casefold()
        if category == "dataset_detail" and (
            path_lower.startswith("data/") or "dataset" in target.casefold()
        ):
            role_bonus = 2.0
        elif category == "baseline_method" and path_lower.startswith("baselines/"):
            role_bonus = 2.0
        elif category == "referenced_architecture" and (
            path_lower.startswith("models/") or "model" in purpose.casefold()
        ):
            role_bonus = 2.0
        elif category == "missing_hyperparameter" and "config" in target.casefold():
            role_bonus = 1.0

        evidence = {
            "file": path,
            "requirement_ids": requirement_overlap,
            "matched_target_tokens": matched_tokens,
            "matched_path_tokens": matched_path_tokens,
        }
        if requirement_overlap:
            score = 20.0 + 3.0 * len(requirement_overlap) + role_bonus
            evidence["score"] = score
            requirement_candidates.append((score, path, evidence))
        semantic_signal = (
            len(matched_tokens) >= 2
            or any(len(token) >= 8 for token in matched_path_tokens)
            or (bool(matched_tokens) and role_bonus >= 2.0)
            or (len(target_tokens) == 1 and bool(matched_tokens))
        )
        if semantic_signal:
            coverage = len(matched_tokens) / max(len(target_tokens), 1)
            top_directory = path_lower.split("/", 1)[0]
            directory_bonus = 3.0 if _matched_target_tokens(
                gap.get("target_name"), top_directory
            ) else 0.0
            score = (
                10.0
                + 5.0 * len(matched_path_tokens)
                + 3.0 * len(matched_purpose_tokens)
                + 3.0 * coverage
                + role_bonus
                + directory_bonus
            )
            evidence = dict(evidence)
            evidence["score"] = round(score, 4)
            semantic_candidates.append((score, path, evidence))

    selected: list[tuple[float, str, dict]] = []
    # Target semantics usually identify the implementation owner more directly
    # than an affected table/figure ID, so select semantic candidates first.
    for candidate in sorted(semantic_candidates, reverse=True):
        if len(selected) >= max_files:
            break
        if candidate[1] not in {item[1] for item in selected}:
            selected.append(candidate)
    for candidate in sorted(requirement_candidates, reverse=True):
        if len(selected) >= max_files:
            break
        if candidate[1] not in {item[1] for item in selected}:
            selected.append(candidate)

    selected.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in selected], [item[2] for item in selected]


def attach_target_files_to_context(supplementary: dict, architecture: dict) -> dict:
    """Attach post-planning file ownership to every gap collection."""
    updated = copy.deepcopy(supplementary or {})
    all_gaps = list(updated.get("all_gaps") or [])
    if not all_gaps:
        seen: set[str] = set()
        for collection_name in (
            "resolved_gaps", "partial_gaps", "source_gaps", "unresolved_gaps",
        ):
            for gap in updated.get(collection_name, []) or []:
                key = str(gap.get("id") or repr(gap))
                if key not in seen:
                    seen.add(key)
                    all_gaps.append(gap)

    routed_by_id: dict[str, dict] = {}
    reason_counts: dict[str, int] = {"requirement_id": 0, "target_name": 0}
    for gap in all_gaps:
        routed = dict(gap)
        target_files, evidence = route_gap_to_architecture(routed, architecture)
        routed["target_files"] = target_files
        routed["routing_status"] = "routed" if target_files else "unrouted"
        routed["routing_evidence"] = evidence
        for item in evidence:
            if item.get("requirement_ids"):
                reason_counts["requirement_id"] += 1
            if item.get("matched_target_tokens"):
                reason_counts["target_name"] += 1
        routed_by_id[str(routed.get("id") or repr(routed))] = routed

    updated["all_gaps"] = list(routed_by_id.values())
    for collection_name in (
        "actionable_gaps", "resolved_gaps", "partial_gaps", "source_gaps",
        "unresolved_gaps", "missing_source_gaps",
    ):
        if collection_name in updated:
            updated[collection_name] = [
                routed_by_id.get(str(gap.get("id") or repr(gap)), gap)
                for gap in updated.get(collection_name, []) or []
            ]
    updated["routing_summary"] = {
        "total_gaps": len(routed_by_id),
        "routed_gaps": sum(bool(gap.get("target_files")) for gap in routed_by_id.values()),
        "unrouted_gaps": sum(not gap.get("target_files") for gap in routed_by_id.values()),
        "gap_file_routes": sum(len(gap.get("target_files", [])) for gap in routed_by_id.values()),
        "route_evidence_counts": reason_counts,
        "category_fallback_routes": 0,
    }
    return updated


def scan_text_for_datasets(text: str) -> dict[str, dict]:
    """Scan prose for complete, bounded dataset aliases."""
    return {
        result["full_name"]: {"from_registry": True, **result}
        for result in find_datasets_in_text(text)
    }


def _registry_can_answer(gap: dict) -> bool:
    requested = " ".join(str(value or "") for value in (
        gap.get("description"),
        gap.get("what_is_needed"),
    ))
    return not any(pattern.search(requested) for pattern in _UNSUPPORTED_REGISTRY_REQUESTS)


def resolve_dataset_gap(gap: dict) -> dict | None:
    """Resolve only one explicit dataset and only details the registry contains."""
    if not _registry_can_answer(gap):
        return None

    target_name = gap.get("target_name") or gap.get("dataset_name")
    if target_name:
        named = scan_text_for_datasets(str(target_name))
        if len(named) == 1:
            return {"resolution_scope": "canonical_registry_metadata", **next(iter(named.values()))}
        return None

    searchable = " ".join(str(value or "") for value in (
        gap.get("description"),
        gap.get("what_is_needed"),
        gap.get("context_quote"),
        gap.get("referenced_paper", {}).get("title", ""),
    ))
    found = scan_text_for_datasets(searchable)
    if len(found) == 1:
        return {"resolution_scope": "canonical_registry_metadata", **next(iter(found.values()))}
    return None
