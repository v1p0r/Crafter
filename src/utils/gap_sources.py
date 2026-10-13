"""Load explicitly scoped, immutable reference repositories for GapAgent.

The pipeline never guesses that one repository applies to every gap. A source
manifest must bind referenced implementations by reference ID, paper title, or
target name. Target-paper implementations are forbidden because exposing the
codebase being reproduced would contaminate paper-to-code evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from src.utils.gap_evidence import build_repository_evidence_spans
from src.utils.gap_source_discovery import github_repository_identity


logger = logging.getLogger("gap_sources")

MAX_SOURCE_REPOSITORIES = 16
ALLOWED_REPOSITORY_ROLES = frozenset({
    "referenced_implementation",
})


def _normalized_words(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def repository_applies_to_gap(definition: dict, gap: dict) -> bool:
    """Return True only for an explicit source-to-gap scope match."""
    role = str(definition.get("role") or "")
    if role != "referenced_implementation":
        return False

    allowed_ref_ids = {
        str(value).strip().casefold()
        for value in definition.get("ref_ids", []) or [] if str(value).strip()
    }
    gap_ref_id = str(gap.get("ref_id") or "").strip().casefold()
    if gap_ref_id and gap_ref_id in allowed_ref_ids:
        return True

    referenced = gap.get("referenced_paper") or {}
    gap_title = _normalized_words(referenced.get("title"))
    source_titles = {
        _normalized_words(value)
        for value in definition.get("paper_titles", []) or [] if _normalized_words(value)
    }
    if gap_title and gap_title in source_titles:
        return True

    target = _normalized_words(gap.get("target_name"))
    source_targets = {
        _normalized_words(value)
        for value in definition.get("target_names", []) or [] if _normalized_words(value)
    }
    return bool(target and target in source_targets)


def _git_value(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    return result.stdout.strip()


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gap_source_manifest(
    manifest_path: str | Path,
    *,
    target_codebase_blacklist: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a source manifest and return provenance-rich evidence spans.

    Repositories must be clean Git checkouts at a concrete commit. Symlinked
    roots or any symlink within the checkout are rejected rather than followed.
    """
    blacklist_values = [str(value) for value in target_codebase_blacklist]
    blacklist_identities = {
        github_repository_identity(value) for value in blacklist_values
    }
    if not blacklist_values or "" in blacklist_identities:
        return [], {
            "loaded": False,
            "reason": (
                "target_codebase_blacklist_required"
                if not blacklist_values
                else "target_codebase_blacklist_invalid"
            ),
        }

    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file():
        return [], {"loaded": False, "reason": "manifest_missing_or_symlink"}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [], {"loaded": False, "reason": "manifest_invalid_json"}
    definitions = payload.get("repositories", []) if isinstance(payload, dict) else []
    if not isinstance(definitions, list):
        return [], {"loaded": False, "reason": "repositories_not_a_list"}

    all_spans: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw_definition in enumerate(definitions[:MAX_SOURCE_REPOSITORIES]):
        if not isinstance(raw_definition, dict):
            rejected.append({"index": index, "reason": "definition_not_an_object"})
            continue
        definition = dict(raw_definition)
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(definition.get("name") or "")).strip("_")
        role = str(definition.get("role") or "")
        path_value = str(definition.get("path") or "")
        expected_url = str(definition.get("url") or "")
        expected_identity = github_repository_identity(expected_url)
        repository = Path(path_value)
        if not repository.is_absolute():
            repository = manifest.parent / repository
        reason = ""
        if not name:
            reason = "missing_name"
        elif not expected_identity:
            reason = "missing_or_invalid_repository_url"
        elif expected_identity in blacklist_identities:
            reason = "target_codebase_blacklisted"
        elif role == "target_implementation":
            reason = "target_implementation_forbidden"
        elif role not in ALLOWED_REPOSITORY_ROLES:
            reason = "invalid_role"
        elif repository.is_symlink() or not repository.is_dir():
            reason = "repository_missing_or_symlink"
        if reason:
            rejected.append({"index": index, "name": name, "reason": reason})
            continue

        # Read only Git origin metadata first. This catches a manifest whose
        # declared URL is harmless but whose local path actually points at the
        # blacklisted target checkout before walking or reading its source tree.
        try:
            origin = _git_value(repository, "remote", "get-url", "origin")
        except (OSError, subprocess.SubprocessError):
            rejected.append({"index": index, "name": name, "reason": "git_validation_failed"})
            continue
        origin_identity = github_repository_identity(origin)
        if origin_identity in blacklist_identities:
            rejected.append({
                "index": index,
                "name": name,
                "reason": "target_codebase_blacklisted_origin",
            })
            continue
        if expected_identity != origin_identity:
            rejected.append({"index": index, "name": name, "reason": "origin_url_mismatch"})
            continue
        if any(path.is_symlink() for path in repository.rglob("*")):
            rejected.append({
                "index": index, "name": name, "reason": "repository_contains_symlink",
            })
            continue
        try:
            commit = _git_value(repository, "rev-parse", "HEAD")
            status = _git_value(repository, "status", "--porcelain", "--untracked-files=all")
        except (OSError, subprocess.SubprocessError):
            rejected.append({"index": index, "name": name, "reason": "git_validation_failed"})
            continue
        expected_commit = str(definition.get("commit") or "").strip()
        if status:
            rejected.append({"index": index, "name": name, "reason": "repository_not_clean"})
            continue
        if expected_commit and expected_commit != commit:
            rejected.append({"index": index, "name": name, "reason": "commit_mismatch"})
            continue

        spans = build_repository_evidence_spans(repository, id_prefix=name)
        definition_scope = {
            key: definition.get(key, [])
            for key in ("ref_ids", "paper_titles", "target_names")
        }
        all_spans.extend({
            **span,
            "repository": name,
            "repository_url": expected_url,
            "repository_commit": commit,
            "repository_role": role,
            "repository_scope": definition_scope,
        } for span in spans)
        accepted.append({
            "name": name,
            "role": role,
            "path": path_value,
            "url": expected_url,
            "commit": commit,
            "span_count": len(spans),
            "source_files": len({span["source_path"] for span in spans}),
            **definition_scope,
        })

    return all_spans, {
        "loaded": True,
        "schema_version": payload.get("schema_version", 1),
        "manifest_file": manifest.name,
        "manifest_sha256": _manifest_hash(manifest),
        "repository_access_policy": {
            "schema_version": 1,
            "target_codebase_blacklist": blacklist_values,
            "target_codebase_identities": sorted(blacklist_identities),
            "target_implementations_allowed": False,
        },
        "accepted": accepted,
        "rejected": rejected,
        "truncated": len(definitions) > MAX_SOURCE_REPOSITORIES,
    }


def repository_spans_for_gap(spans: list[dict], gap: dict) -> list[dict]:
    """Filter loaded spans using their manifest-declared gap scope."""
    grouped: dict[str, list[dict]] = {}
    for span in spans:
        grouped.setdefault(str(span.get("repository") or ""), []).append(span)
    selected: list[dict] = []
    for repository_spans in grouped.values():
        first = repository_spans[0]
        definition = {
            "role": first.get("repository_role"),
            **(first.get("repository_scope") or {}),
        }
        if repository_applies_to_gap(definition, gap):
            selected.extend(repository_spans)
    return selected
