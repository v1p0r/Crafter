"""Gap Analysis Agent — Stage 2.5 of the PTC pipeline.

Identifies information gaps in the target paper via one LLM call that also
parses the bibliography.  Each gap is linked to an exact reference entry so
we search Semantic Scholar with the real title, not an LLM guess.
"""

import asyncio
import logging
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.memory.global_memory import GlobalMemory
from src.utils.gap_evidence import (
    apply_span_coverage_judgment,
    build_paper_evidence_spans,
    gap_retrieval_queries,
    material_coverage_relevance,
    rank_evidence_spans_multi_query,
)
from src.utils.gap_matching import (
    resolve_dataset_gap,
    scan_text_for_datasets,
)
from src.utils.gap_resolution import (
    aggregate_appendix_sections,
    apply_appendix_judgment,
    is_gap_actionable,
    is_gap_resolved,
    validated_registry_result,
)
from src.utils.gap_source_discovery import (
    discover_repository_manifest,
    normalize_github_repository_blacklist,
)
from src.utils.gap_sources import (
    load_gap_source_manifest,
    repository_spans_for_gap,
)
from src.utils.llm_client import call_minimax, parse_json
from src.utils.paper_content import item_to_text
from src.utils.paper_source import extract_arxiv_id, fetch_arxiv_html, parse_arxiv_html
from src.utils import semantic_scholar_client as s2

logger = logging.getLogger("gap_agent")

# Title similarity threshold for verifying S2 search results
TITLE_VERIFY_THRESHOLD = 0.55
APPENDIX_VALIDATION_BATCH_SIZE = 10
APPENDIX_VALIDATION_RETRY_DELAYS = (2, 5)
EVIDENCE_VALIDATION_CONCURRENCY = 4
EVIDENCE_VALIDATION_RETRY_DELAYS = (2, 5)
EVIDENCE_REQUEST_TIMEOUT_SECONDS = 300
EVIDENCE_TOP_K = 90
EVIDENCE_PER_QUERY = 16

GAP_IDENTIFICATION_SYSTEM = """\
You are an expert ML research analyst. You will receive:
1. The BODY of a research paper (methods, experiments, etc.)
2. The REFERENCES / BIBLIOGRAPHY section

Your job has TWO parts:

## Part 1 — Parse the references
Extract every reference entry into a structured list. For each, extract:
- ref_id: the citation key as it appears in the paper (e.g. "23", "Hu et al., 2021")
- title: the EXACT title from the bibliography (copy it verbatim)
- authors: author names
- year: publication year
- arxiv_id: arXiv identifier if present (e.g. "2106.09685")

## Part 2 — Identify implementation gaps
Find every place where the paper references external work that is NEEDED to \
implement the paper's method, experiments, or evaluation, but does not provide \
sufficient detail for a developer to write working code.

Categorize each gap as one of:
1. referenced_architecture: External model/architecture the paper uses or builds on
2. baseline_method: A comparison method the paper evaluates against but does not describe
3. dataset_detail: A dataset where preprocessing/splits/augmentation details are missing
4. missing_hyperparameter: A hyperparameter referenced from another paper
5. algorithm_gap: An algorithm step that references another paper or is garbled/incomplete

For each gap, provide:
- category: one of the five types above
- target_name: the exact name of the missing dataset, model, baseline, or
algorithm as written in the paper; use null when no explicit name is present
- requirement_ids: exact paper-local identifiers directly affected by the gap
(e.g. ["Eq. 4", "Algorithm 1", "Figure 2"]); use [] when none are stated
- description: what information is missing (1-2 sentences)
- context_quote: the exact text from the paper where this gap appears
- ref_id: which reference entry this gap refers to (from Part 1). Use null if \
the gap is internal or has no external reference.
- internal_ref: if the paper delegates details to its own appendix or a specific \
internal section (e.g. "see Appendix B for the derivation", "hyperparameters in \
Appendix A.1"), the appendix/section identifier as it appears in the paper \
(e.g. "Appendix B", "A.1", "A"). Use null if the gap does not point to an \
internal section.
- importance: "critical" | "high" | "low"
- what_is_needed: specifically what information would resolve this gap

Do NOT flag standard well-known components (Adam, ReLU, BatchNorm, dropout, etc.).

Output ONLY valid JSON:
{
  "references": [
    {"ref_id": "23", "title": "Exact Title From Bibliography", "authors": "...", "year": 2021, "arxiv_id": "2106.09685"}
  ],
  "gaps": [
    {
      "id": "gap_001",
      "category": "baseline_method",
      "target_name": "Exact baseline name",
      "requirement_ids": ["Table 1"],
      "description": "...",
      "context_quote": "...",
      "ref_id": "23",
      "internal_ref": null,
      "importance": "critical",
      "what_is_needed": "..."
    }
  ]
}"""

APPENDIX_VALIDATION_SYSTEM = """\
You are a strict evidence validator for implementation gaps in a research paper.

You receive appendix sources and gaps that point to them. For each gap, decide
whether the supplied appendix text itself contains the information requested in
`what_is_needed`. Use no outside knowledge.

Statuses:
- resolved: every material requested detail is explicitly present.
- partial: at least one concrete requested implementation detail is present,
  but material requested details remain missing.
- insufficient: only a heading, promise, high-level description, unrelated
  content, or no concrete requested detail is present.

For resolved or partial, provide one or more SHORT, EXACT quotes copied from the
source. Each quote must support a requested detail. A section title, citation,
URL, or statement that another table/source has the answer is not evidence.
Use resolved only when `missing_needs` is empty.

Output ONLY valid JSON:
{
  "judgments": [
    {
      "id": "gap_001",
      "status": "resolved|partial|insufficient",
      "evidence": [{"quote": "exact source text", "supports": "requested detail"}],
      "covered_needs": ["..."],
      "missing_needs": ["..."],
      "rationale": "..."
    }
  ]
}"""

EVIDENCE_EXTRACTION_SYSTEM = """\
You are a strict implementation-evidence extractor.

Decompose what_is_needed into independently checkable atomic needs. Mark an
atomic need supported only when one or more supplied spans explicitly establish
a requested concrete value, formula, executable behavior, configuration,
schema, command, or procedure. Do not count a heading, citation, URL, promise,
or high-level description as implementation evidence.

Respect source scope:
- target_paper can establish target method/setup choices;
- referenced_paper and referenced_implementation can establish how that cited
  method works, but cannot prove a target-paper-specific experimental choice;
- when what_is_needed explicitly asks for details "from the original
  implementation", referenced_implementation is in scope for those details.
Treat an explicit "X or Y" as alternative sufficient evidence, not two
simultaneously required needs. A concrete supported component remains supported
when other components are missing.
Absence is not evidence. Use no outside knowledge and cite only supplied span
IDs. Set coverage_complete=true only if all material requested needs are
supported. Output only valid JSON:
{"id":"gap_001","coverage":[{"need":"one atomic need","status":"supported|missing","span_ids":["allowed_id"],"explanation":"what is established and source scope"}],"coverage_complete":false,"rationale":"..."}"""

EVIDENCE_PLANNER_SYSTEM = """\
You plan source retrieval for one implementation gap. You receive the gap and
an exact inventory of official source paths. Infer concrete code symbols,
configuration keys, CLI flags, formulas, table labels, and phrases that could
establish each requested detail. Return 8-18 focused queries and at most 8
inventory paths. When the gap asks about a configuration, hyperparameter,
preprocessing step, or variant choice, inspect all three evidence roles when
available: (1) the definition, (2) an actual entrypoint/run command or config,
and (3) the evaluation/output consumer. Copy repository and path values
exactly. Do not claim that the gap is resolved and do not invent paths. Output
only valid JSON:
{"queries":["focused query"],"preferred_files":[{"repository":"exact repo","path":"exact path"}],"rationale":"..."}"""


class GapAgent(BaseAgent):
    def __init__(
        self,
        global_memory: GlobalMemory,
        output_dir: str = "data/output",
    ):
        super().__init__(name="gap_agent", global_memory=global_memory, output_dir=output_dir)
        self.output_dir = Path(output_dir)

    async def run(self) -> dict[str, Any]:
        partitions = self.global_memory.get("partitions", [])
        if not partitions:
            logger.warning("No partitions in global memory — skipping gap analysis")
            empty = self._build_supplementary_context([], {})
            self.global_memory.set("supplementary_context", empty)
            return empty

        self.log_reasoning("start", "Beginning gap analysis")

        # Step 1: Build paper text + references
        paper_text = self._get_relevant_text(partitions)
        references_text = self._get_references_text(partitions)
        full_input = paper_text
        if references_text:
            full_input += f"\n\n## References / Bibliography\n{references_text}"

        # Step 2: ONE LLM call — parse references + identify gaps
        parsed_refs, gaps = await self._identify_gaps(full_input)
        self.log_reasoning("identify", f"Found {len(gaps)} gaps, {len(parsed_refs)} references parsed")

        # Step 3: Link each gap to its parsed reference entry
        ref_by_id = {str(r.get("ref_id", "")): r for r in parsed_refs}
        for gap in gaps:
            ref_id = str(gap.get("ref_id", ""))
            if ref_id and ref_id in ref_by_id:
                gap["referenced_paper"] = ref_by_id[ref_id]
            elif "referenced_paper" not in gap:
                gap["referenced_paper"] = {}

        # MinerU output can end at the bibliography even when the body points
        # to an appendix. Fetch the verified target-paper HTML only for those
        # internal references that are genuinely absent from local partitions.
        official_target = await self._load_missing_official_target_source(partitions, gaps)
        configured_repository_spans, configured_repository_summary = (
            self._load_repository_evidence_sources()
        )

        # Step 4: Resolve dataset gaps from registry
        dataset_details = self._resolve_datasets_from_text(partitions)

        # Step 5: Collect candidates via appendix/S2/registry.
        resolved = []
        for gap in gaps:
            result = await self._resolve_gap(gap, partitions, official_target)
            resolved.append(result)
            tier = result.get("tier", 3)
            ref_title = result.get("referenced_paper", {}).get("title", "?")
            self.log_reasoning("resolve", f"Tier {tier}: {ref_title[:60]}")

        # Step 5.5: Validate appendix candidates in batches. Registry results
        # have a deterministic capability gate; S2 metadata remains source-only.
        resolved = await self._validate_appendix_candidates(resolved)

        # Step 5.6: Fetch verified referenced papers once, then discover scoped
        # reference implementations. The caller-supplied target codebase is
        # blocked before cloning; target_implementation is forbidden outright.
        # The generated manifest contains normal directories and exact commits.
        referenced_evidence = await self._load_referenced_evidence_sources(resolved)
        automatic_repository_spans, automatic_repository_summary = (
            await self._load_automatic_repository_evidence_sources(
                resolved,
                partitions,
                official_target,
                referenced_evidence[1],
                configured_repository_summary,
            )
        )
        repository_spans = configured_repository_spans + automatic_repository_spans
        repository_summary = {
            "configured": configured_repository_summary,
            "automatic": automatic_repository_summary,
            "span_count": len(repository_spans),
        }

        # Step 5.7: Search auditable spans from the target paper, each gap's
        # referenced paper, and explicitly scoped clean source repositories.
        # Every model claim is reduced back to known span IDs before it can
        # become actionable.
        resolved, evidence_summary = await self._validate_multisource_evidence(
            resolved,
            partitions,
            official_target,
            repository_spans,
            referenced_evidence=referenced_evidence,
        )

        # Step 6: Store supplementary_context
        supplementary = self._build_supplementary_context(resolved, dataset_details)
        supplementary["gap_source_summary"] = {
            "repositories": repository_summary,
            "evidence_resolution": evidence_summary,
            "official_target": (official_target or {}).get("metadata"),
            "repository_access_policy": self._repository_access_policy(),
        }
        self.global_memory.set("supplementary_context", supplementary)

        summary = supplementary["resolution_summary"]
        self.log_reasoning(
            "done",
            f"Resolved {summary['resolved']}/{summary['total_gaps']} gaps "
            f"(partial: {summary['partial']}, source-only: {summary['source_found']}; "
            f"datasets: {len(dataset_details)})"
        )
        return supplementary

    # ------------------------------------------------------------------
    # Gap identification (one LLM call — parses refs + finds gaps)
    # ------------------------------------------------------------------

    async def _identify_gaps(self, paper_text: str) -> tuple[list[dict], list[dict]]:
        """Returns (parsed_references, gaps)."""
        messages = [{"role": "user", "content": paper_text}]
        self.log_message("user", paper_text[:500] + "...")

        response = await call_minimax(
            messages=messages,
            system=GAP_IDENTIFICATION_SYSTEM,
            max_tokens=16384,
            stage="gap",
        )
        self.log_message("assistant", response)

        result = parse_json(response, fallback={"references": [], "gaps": []})
        if not isinstance(result, dict):
            return [], []

        refs = result.get("references", [])
        gaps = result.get("gaps", [])

        # Cap at 25 gaps, prioritize by importance
        importance_order = {"critical": 0, "high": 1, "low": 2}
        gaps.sort(key=lambda g: importance_order.get(g.get("importance", "low"), 2))
        return refs, gaps[:25]

    # ------------------------------------------------------------------
    # Resolution cascade
    # ------------------------------------------------------------------

    async def _resolve_gap(
        self,
        gap: dict,
        partitions: list,
        official_target: dict[str, Any] | None = None,
    ) -> dict:
        ref = gap.get("referenced_paper", {})

        # Tier 0: Dataset registry (for dataset_detail gaps)
        if gap.get("category") == "dataset_detail":
            result = resolve_dataset_gap(gap)
            if result:
                return validated_registry_result(gap, result)

        # Tier 0.5: Internal appendix reference (from within the paper itself)
        internal_ref = gap.get("internal_ref")
        if internal_ref:
            result = self._resolve_from_appendix(internal_ref, partitions)
            if result:
                return {
                    **gap,
                    "source_candidate": result,
                    "resolution_status": "pending_validation",
                    "tier": 0.5,
                    "confidence": "pending",
                }
            official_partitions = (official_target or {}).get("partitions") or []
            result = self._resolve_from_appendix(internal_ref, official_partitions)
            if result:
                metadata = (official_target or {}).get("metadata") or {}
                result.update({
                    "source_type": "target_paper_official_html",
                    "source_url": metadata.get("source_url"),
                    "source_sha256": metadata.get("sha256"),
                    "source_arxiv_id": metadata.get("arxiv_id"),
                    "cache_file": metadata.get("cache_file"),
                    "from_official_target_paper": True,
                })
                return {
                    **gap,
                    "source_candidate": result,
                    "resolution_status": "pending_validation",
                    "tier": 0.4,
                    "confidence": "pending",
                }

        # Skip gaps with no external reference to look up
        title = ref.get("title", "")
        arxiv_id = ref.get("arxiv_id", "")
        if not title and not arxiv_id:
            return {
                **gap, "resolution": None, "tier": 3, "confidence": "none",
                "resolution_status": "unresolved",
                "unresolved_note": "No external paper reference to look up",
            }

        # Tier 1: Semantic Scholar API with verification
        result = await self._resolve_from_semantic_scholar(ref)
        if result:
            return {
                **gap,
                "source_candidate": result,
                "resolution_status": "source_found",
                "tier": 1,
                "confidence": "medium",
                "unresolved_note": "Referenced source found, but requested implementation details were not extracted",
            }

        # Tier 3: Unresolvable
        return {
            **gap, "resolution": None, "tier": 3, "confidence": "none",
            "resolution_status": "unresolved",
            "unresolved_note": f"Could not find: {title or ref.get('authors') or 'unknown'}",
        }

    @staticmethod
    def _resolve_from_appendix(internal_ref: str, partitions: list) -> dict | None:
        """Aggregate a referenced appendix section and all descendants."""
        return aggregate_appendix_sections(internal_ref, partitions)

    async def _load_missing_official_target_source(
        self,
        partitions: list[dict],
        gaps: list[dict],
    ) -> dict[str, Any] | None:
        """Load official target HTML iff local internal references are absent."""
        missing_refs = sorted({
            str(gap.get("internal_ref") or "").strip()
            for gap in gaps
            if gap.get("internal_ref")
            and not self._resolve_from_appendix(str(gap["internal_ref"]), partitions)
        })
        if not missing_refs:
            return None

        title = self._target_paper_title(partitions)
        if not title:
            logger.warning("Cannot recover missing internal references: target title is absent")
            return None

        arxiv_id = extract_arxiv_id(self.global_memory.get("input_path", ""))
        if not arxiv_id:
            metadata = self.global_memory.get("paper_metadata", {})
            if isinstance(metadata, dict):
                arxiv_id = extract_arxiv_id(metadata.get("arxiv_id", ""))
        if not arxiv_id:
            search_result = await s2.search_by_title(self._clean_ocr_title(title))
            if (
                search_result
                and self._verify_title_match(title, str(search_result.get("title") or ""))
            ):
                arxiv_id = extract_arxiv_id(search_result.get("arxiv_id", ""))
        if not arxiv_id:
            logger.warning(
                "Cannot recover missing internal references %s: verified target arXiv ID not found",
                ", ".join(missing_refs),
            )
            return None

        fetched = await fetch_arxiv_html(
            arxiv_id,
            self.output_dir / "gap_sources" / "arxiv",
        )
        if not fetched:
            return None
        parsed = parse_arxiv_html(str(fetched.get("html") or ""))
        if not self._verify_title_match(title, str(parsed.get("title") or "")):
            logger.warning("Rejected arXiv %s: target-paper title verification failed", arxiv_id)
            return None

        recovered_refs = [
            internal_ref for internal_ref in missing_refs
            if self._resolve_from_appendix(internal_ref, parsed.get("partitions") or [])
        ]
        if not recovered_refs:
            logger.warning(
                "Rejected arXiv %s: none of the missing internal references were present",
                arxiv_id,
            )
            return None

        cache_path = Path(str(fetched["cache_path"]))
        try:
            cache_file = cache_path.relative_to(self.output_dir).as_posix()
        except ValueError:
            cache_file = cache_path.name
        metadata = {
            key: fetched.get(key)
            for key in ("arxiv_id", "source_url", "sha256", "size_bytes", "from_cache")
        }
        metadata.update({
            "cache_file": cache_file,
            "title": parsed.get("title"),
            "recovered_internal_refs": recovered_refs,
        })
        self.log_reasoning(
            "official_target_fallback",
            f"Recovered {len(recovered_refs)}/{len(missing_refs)} missing internal references "
            f"from verified arXiv {arxiv_id} ({fetched['sha256'][:12]})",
        )
        return {"partitions": parsed["partitions"], "metadata": metadata}

    @staticmethod
    def _target_paper_title(partitions: list[dict]) -> str:
        """Return the first non-generic document heading from parsed partitions."""
        generic = {"abstract", "introduction", "preamble", "references", "bibliography"}
        for section in partitions[:6]:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            normalized = re.sub(r"^\d+(?:\.\d+)*\s+", "", title).casefold()
            if len(title) >= 8 and normalized not in generic:
                return title
        return ""

    def _repository_access_policy(self) -> dict[str, Any]:
        """Return the canonical, caller-supplied repository contamination policy."""
        raw = self.global_memory.get("target_codebase_blacklist", [])
        values = raw if isinstance(raw, list) else [raw]
        try:
            blacklist = normalize_github_repository_blacklist(values)
        except ValueError as exc:
            return {
                "schema_version": 1,
                "repository_evidence_enabled": False,
                "reason": "target_codebase_blacklist_invalid",
                "detail": str(exc),
                "target_codebase_blacklist": [],
            }
        return {
            "schema_version": 1,
            "repository_evidence_enabled": bool(blacklist),
            "reason": "enabled" if blacklist else "target_codebase_blacklist_missing",
            "target_codebase_blacklist": blacklist,
            "target_implementations_allowed": False,
        }

    def _load_repository_evidence_sources(self) -> tuple[list[dict], dict[str, Any]]:
        """Load a colocated, explicitly scoped source manifest when present."""
        policy = self._repository_access_policy()
        if not policy["repository_evidence_enabled"]:
            return [], {
                "loaded": False,
                "reason": policy["reason"],
                "repository_access_policy": policy,
            }
        configured = self.global_memory.get("gap_source_manifest", "")
        if configured:
            manifest = Path(str(configured))
        else:
            input_path = Path(str(self.global_memory.get("input_path", "")))
            manifest = input_path.parent / "gap_sources.json" if str(input_path) else Path()
        if not configured and (not str(manifest) or not manifest.is_file()):
            return [], {"loaded": False, "reason": "not_configured"}
        spans, summary = load_gap_source_manifest(
            manifest,
            target_codebase_blacklist=policy["target_codebase_blacklist"],
        )
        if summary.get("accepted"):
            self.log_reasoning(
                "repository_sources",
                f"Loaded {len(summary['accepted'])} clean scoped repositories "
                f"with {len(spans)} auditable spans",
            )
        if summary.get("rejected"):
            logger.warning(
                "Rejected %d configured gap source repositories",
                len(summary["rejected"]),
            )
        return spans, summary

    async def _load_automatic_repository_evidence_sources(
        self,
        results: list[dict],
        partitions: list[dict],
        official_target: dict[str, Any] | None,
        referenced_summary: dict[str, Any],
        configured_summary: dict[str, Any],
    ) -> tuple[list[dict], dict[str, Any]]:
        """Discover paper-linked GitHub repositories and load their manifest."""
        if self.global_memory.get("disable_auto_gap_sources", False):
            return [], {"loaded": False, "reason": "disabled"}

        policy = self._repository_access_policy()
        if not policy["repository_evidence_enabled"]:
            return [], {
                "loaded": False,
                "reason": policy["reason"],
                "repository_access_policy": policy,
            }

        target_title = self._target_paper_title(partitions)
        if not target_title:
            return [], {"loaded": False, "reason": "target_title_missing"}

        target_metadata = (official_target or {}).get("metadata") or {}
        target_arxiv_id = extract_arxiv_id(target_metadata.get("arxiv_id", ""))
        if not target_arxiv_id:
            target_arxiv_id = extract_arxiv_id(self.global_memory.get("input_path", ""))
        if not target_arxiv_id:
            paper_metadata = self.global_memory.get("paper_metadata", {})
            if isinstance(paper_metadata, dict):
                target_arxiv_id = extract_arxiv_id(paper_metadata.get("arxiv_id", ""))
        if not target_arxiv_id:
            search_result = await s2.search_by_title(self._clean_ocr_title(target_title))
            if (
                search_result
                and self._verify_title_match(
                    target_title, str(search_result.get("title") or ""),
                )
            ):
                target_arxiv_id = extract_arxiv_id(search_result.get("arxiv_id", ""))

        html_documents: list[dict[str, Any]] = []
        target_cache_file = str(target_metadata.get("cache_file") or "")
        if target_cache_file:
            target_cache_path = Path(target_cache_file)
            if not target_cache_path.is_absolute():
                target_cache_path = self.output_dir / target_cache_path
            html_documents.append({
                "document_role": "target_paper",
                "paper_title": target_title,
                "arxiv_id": target_arxiv_id or "",
                "source_url": target_metadata.get("source_url"),
                "source_sha256": target_metadata.get("sha256"),
                "html_path": str(target_cache_path),
            })
        elif target_arxiv_id:
            # A readily available target arXiv ID lets us recover href values
            # that PDF/OCR partitions may have dropped. The title is still
            # checked before any links are trusted.
            fetched = await fetch_arxiv_html(
                target_arxiv_id,
                self.output_dir / "gap_sources" / "arxiv",
            )
            if fetched:
                parsed = parse_arxiv_html(str(fetched.get("html") or ""))
                if self._verify_title_match(target_title, str(parsed.get("title") or "")):
                    html_documents.append({
                        "document_role": "target_paper",
                        "paper_title": target_title,
                        "arxiv_id": target_arxiv_id,
                        "source_url": fetched.get("source_url"),
                        "source_sha256": fetched.get("sha256"),
                        "html_path": fetched.get("cache_path"),
                    })

        for source in referenced_summary.get("sources", []) or []:
            if not isinstance(source, dict) or source.get("status") != "loaded":
                continue
            arxiv_id = str(source.get("arxiv_id") or "")
            html_path = str(source.get("cache_path") or "")
            if not arxiv_id or not html_path:
                continue
            matching = []
            for gap in results:
                referenced = gap.get("referenced_paper") or {}
                candidate = gap.get("source_candidate") or {}
                gap_arxiv = extract_arxiv_id(
                    candidate.get("arxiv_id") or referenced.get("arxiv_id") or ""
                )
                if gap_arxiv == arxiv_id:
                    matching.append(gap)
            title = str(source.get("parsed_title") or "").strip()
            if not title and matching:
                title = str((matching[0].get("referenced_paper") or {}).get("title") or "")
            html_documents.append({
                "document_role": "referenced_paper",
                "paper_title": title,
                "arxiv_id": arxiv_id,
                "source_url": source.get("source_url"),
                "source_sha256": source.get("sha256"),
                "html_path": html_path,
            })

        excluded_urls = [
            str(item.get("url") or "")
            for item in configured_summary.get("accepted", []) or []
            if isinstance(item, dict) and item.get("url")
        ]
        try:
            discovery = discover_repository_manifest(
                output_dir=self.output_dir,
                target_partitions=partitions,
                target_title=target_title,
                target_arxiv_id=target_arxiv_id or "",
                gaps=results,
                target_codebase_blacklist=policy["target_codebase_blacklist"],
                html_documents=html_documents,
                excluded_repository_urls=excluded_urls,
            )
            spans, manifest_summary = load_gap_source_manifest(
                discovery["manifest_path"],
                target_codebase_blacklist=policy["target_codebase_blacklist"],
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Automatic paper repository discovery failed closed: %s", exc)
            return [], {
                "loaded": False,
                "reason": "discovery_failed",
                "detail": str(exc)[:300],
            }

        if manifest_summary.get("accepted"):
            self.log_reasoning(
                "automatic_repository_sources",
                f"Discovered {len(manifest_summary['accepted'])} paper-linked repositories "
                f"with {len(spans)} auditable spans",
            )
        return spans, {
            "loaded": True,
            "discovery": discovery,
            "manifest": manifest_summary,
        }

    @staticmethod
    def _target_evidence_spans(
        partitions: list[dict],
        official_target: dict[str, Any] | None,
    ) -> list[dict]:
        """Build one non-duplicated target-paper corpus with explicit scope."""
        if official_target and official_target.get("partitions"):
            source_partitions = official_target["partitions"]
            metadata = official_target.get("metadata") or {}
            prefix = "target_html_"
            source_type = "target_paper_official_html"
        else:
            source_partitions = partitions
            metadata = {}
            prefix = "target_local_"
            source_type = "target_paper"
        spans = []
        for span in build_paper_evidence_spans(source_partitions):
            section_index = int(span.get("section_index") or 0)
            spans.append({
                **span,
                "span_id": prefix + str(span["span_id"]),
                "source_type": source_type,
                "source_path": f"target-paper/section/{section_index:03d}",
                "source_url": metadata.get("source_url"),
                "source_sha256": metadata.get("sha256"),
                "repository": "target-paper",
                "repository_role": "target_paper",
            })
        return spans

    async def _load_referenced_evidence_sources(
        self,
        results: list[dict],
    ) -> tuple[dict[str, list[dict]], dict[str, Any]]:
        """Fetch title-verified official HTML for referenced papers with arXiv IDs."""
        definitions: dict[str, str] = {}
        for result in results:
            if is_gap_resolved(result):
                continue
            candidate = result.get("source_candidate") or {}
            referenced = result.get("referenced_paper") or {}
            arxiv_id = extract_arxiv_id(
                candidate.get("arxiv_id") or referenced.get("arxiv_id") or ""
            )
            title = str(referenced.get("title") or candidate.get("title") or "").strip()
            if arxiv_id and title:
                definitions.setdefault(arxiv_id, title)

        semaphore = asyncio.Semaphore(3)

        async def load_one(arxiv_id: str, expected_title: str) -> tuple[str, list[dict], dict]:
            async with semaphore:
                fetched = await fetch_arxiv_html(
                    arxiv_id,
                    self.output_dir / "gap_sources" / "arxiv",
                )
            if not fetched:
                return arxiv_id, [], {"arxiv_id": arxiv_id, "status": "fetch_failed"}
            parsed = parse_arxiv_html(str(fetched.get("html") or ""))
            parsed_title = str(parsed.get("title") or "")
            if not self._verify_title_match(expected_title, parsed_title):
                return arxiv_id, [], {
                    "arxiv_id": arxiv_id,
                    "status": "title_mismatch",
                    "expected_title": expected_title,
                    "parsed_title": parsed_title,
                }
            safe_id = arxiv_id.replace(".", "_")
            spans = []
            for span in build_paper_evidence_spans(parsed.get("partitions") or []):
                section_index = int(span.get("section_index") or 0)
                spans.append({
                    **span,
                    "span_id": f"ref_{safe_id}_" + str(span["span_id"]),
                    "source_type": "referenced_paper_official_html",
                    "source_path": f"arxiv:{arxiv_id}/section/{section_index:03d}",
                    "source_url": fetched.get("source_url"),
                    "source_sha256": fetched.get("sha256"),
                    "repository": f"arxiv:{arxiv_id}",
                    "repository_role": "referenced_paper",
                })
            return arxiv_id, spans, {
                "arxiv_id": arxiv_id,
                "status": "loaded",
                "source_url": fetched.get("source_url"),
                "sha256": fetched.get("sha256"),
                "from_cache": fetched.get("from_cache"),
                "cache_path": fetched.get("cache_path"),
                "parsed_title": parsed_title,
                "span_count": len(spans),
            }

        loaded = await asyncio.gather(*[
            load_one(arxiv_id, title) for arxiv_id, title in sorted(definitions.items())
        ]) if definitions else []
        spans_by_id = {arxiv_id: spans for arxiv_id, spans, _ in loaded if spans}
        return spans_by_id, {
            "requested": len(definitions),
            "loaded": len(spans_by_id),
            "sources": [metadata for _, _, metadata in loaded],
        }

    async def _call_evidence_model(
        self,
        semaphore: asyncio.Semaphore,
        *,
        system: str,
        payload: dict,
        max_tokens: int,
    ) -> tuple[str, dict] | tuple[None, None]:
        """Call MiniMax with bounded retries and fail closed on malformed output."""
        last_error: Exception | None = None
        for attempt in range(len(EVIDENCE_VALIDATION_RETRY_DELAYS) + 1):
            try:
                async with semaphore:
                    response = await asyncio.wait_for(
                        call_minimax(
                            messages=[{
                                "role": "user",
                                "content": json.dumps(
                                    payload, ensure_ascii=False, separators=(",", ":"),
                                ),
                            }],
                            system=system,
                            max_tokens=max_tokens,
                            temperature=0.1,
                            max_continuations=1,
                            stage="gap",
                        ),
                        timeout=EVIDENCE_REQUEST_TIMEOUT_SECONDS,
                    )
                parsed = parse_json(response, fallback=None)
                if isinstance(parsed, dict):
                    return response, parsed
                raise ValueError("MiniMax returned malformed evidence JSON")
            except Exception as exc:
                # ``Exception`` is intentional here: all provider, JSON, and
                # transport failures must leave a gap non-actionable.
                last_error = exc
                if attempt < len(EVIDENCE_VALIDATION_RETRY_DELAYS):
                    await asyncio.sleep(EVIDENCE_VALIDATION_RETRY_DELAYS[attempt])
        logger.warning("Gap evidence extraction failed closed: %s", last_error)
        return None, None

    @staticmethod
    def _source_inventory(spans: list[dict]) -> list[dict]:
        grouped: dict[tuple[str, str], set[str]] = {}
        for span in spans:
            path = str(span.get("source_path") or "")
            if not path:
                continue
            key = (
                str(span.get("repository") or "unknown"),
                str(span.get("repository_role") or "unknown"),
            )
            grouped.setdefault(key, set()).add(path)
        return [
            {"repository": repository, "role": role, "paths": sorted(paths)}
            for (repository, role), paths in sorted(grouped.items())
        ]

    @staticmethod
    def _compact_gap(gap: dict) -> dict:
        return {
            key: gap.get(key) for key in (
                "id", "category", "target_name", "description", "context_quote",
                "what_is_needed", "internal_ref", "ref_id", "requirement_ids",
            )
        }

    async def _extract_span_evidence(
        self,
        semaphore: asyncio.Semaphore,
        gap: dict,
        corpus: list[dict],
        *,
        planned_queries: list[str] | None = None,
        preferred_sources: list[tuple[str, str]] | None = None,
    ) -> tuple[dict | None, dict[str, Any]]:
        candidates = rank_evidence_spans_multi_query(
            gap,
            corpus,
            top_k=EVIDENCE_TOP_K,
            per_query=EVIDENCE_PER_QUERY,
            extra_queries=planned_queries,
            preferred_sources=preferred_sources,
            neighbor_radius=2 if planned_queries else 1,
        )
        record: dict[str, Any] = {
            "candidate_spans": candidates,
            "candidate_count": len(candidates),
            "planned_queries": planned_queries or [],
            "preferred_sources": [list(value) for value in (preferred_sources or [])],
        }
        if not candidates:
            record["error"] = "no_ranked_candidates"
            return None, record
        response, judgment = await self._call_evidence_model(
            semaphore,
            system=EVIDENCE_EXTRACTION_SYSTEM,
            payload={
                "gap": self._compact_gap(gap),
                "retrieval_queries": gap_retrieval_queries(gap),
                "planned_queries": planned_queries or [],
                "allowed_source_spans": candidates,
            },
            max_tokens=6144,
        )
        record.update({"response": response, "judgment": judgment})
        if not judgment or str(judgment.get("id")) != str(gap.get("id")):
            record["error"] = "missing_or_wrong_gap_id"
            return None, record
        result = apply_span_coverage_judgment(
            gap,
            judgment,
            candidates,
            tier=0.2 if not planned_queries else 0.1,
            source_type="verified_multisource_evidence",
        )
        record["applied_status"] = result.get("resolution_status")
        record["invalid_span_ids"] = (
            result.get("resolution_evidence") or {}
        ).get("invalid_span_ids", [])
        return result, record

    async def _plan_evidence_retrieval(
        self,
        semaphore: asyncio.Semaphore,
        gap: dict,
        corpus: list[dict],
    ) -> tuple[list[str], list[tuple[str, str]], dict[str, Any]]:
        inventory = self._source_inventory(corpus)
        response, plan = await self._call_evidence_model(
            semaphore,
            system=EVIDENCE_PLANNER_SYSTEM,
            payload={"gap": self._compact_gap(gap), "official_source_inventory": inventory},
            max_tokens=4096,
        )
        record: dict[str, Any] = {"response": response, "plan": plan, "inventory": inventory}
        if not plan:
            return [], [], record
        queries = [
            str(query) for query in (plan.get("queries") or [])
            if str(query).strip()
        ][:18]
        allowed = {
            (str(group["repository"]), str(path))
            for group in inventory for path in group["paths"]
        }
        preferred: list[tuple[str, str]] = []
        invalid: list[list[str]] = []
        for item in plan.get("preferred_files") or []:
            if not isinstance(item, dict):
                continue
            pair = (str(item.get("repository") or ""), str(item.get("path") or ""))
            if pair in allowed:
                preferred.append(pair)
            else:
                invalid.append(list(pair))
        record["validated_preferred_sources"] = [list(value) for value in preferred[:8]]
        record["invalid_preferred_sources"] = invalid
        return queries, preferred[:8], record

    def _write_evidence_record(self, gap_id: object, record: dict[str, Any]) -> None:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(gap_id or "unknown"))
        directory = self.output_dir / "gap_sources" / "judgments"
        if directory.is_symlink():
            logger.warning("Refusing to write gap evidence through symlink: %s", directory)
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{safe_id}.json"
            if path.is_symlink():
                raise OSError("judgment path is a symlink")
            path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not persist evidence record for %s: %s", gap_id, exc)

    @staticmethod
    def _evidence_quality(gap: dict | None) -> tuple[int, int, int, int, int]:
        """Order two span-gated candidates without trusting prose rationale."""
        if not isinstance(gap, dict):
            return (0, 0, 0, 0, 0)
        status_rank = (
            2 if is_gap_resolved(gap)
            else 1 if is_gap_actionable(gap)
            else 0
        )
        evidence = gap.get("resolution_evidence") or {}
        covered = {
            str(value).strip().casefold()
            for value in evidence.get("covered_needs") or []
            if str(value).strip()
        }
        evidence_ids = {
            str(item.get("span_id"))
            for item in evidence.get("evidence") or []
            if isinstance(item, dict) and item.get("span_id")
        }
        missing = {
            str(value).strip().casefold()
            for value in evidence.get("missing_needs") or []
            if str(value).strip()
        }
        material_relevance = material_coverage_relevance(gap, list(covered))
        return (
            status_rank,
            material_relevance,
            -len(missing),
            len(covered),
            len(evidence_ids),
        )

    @classmethod
    def _prefer_evidence_candidate(cls, current: dict, candidate: dict | None) -> dict:
        """Keep a candidate only when its mechanically derived coverage is better."""
        if cls._evidence_quality(candidate) > cls._evidence_quality(current):
            return candidate  # type: ignore[return-value]
        return current

    async def _validate_multisource_evidence(
        self,
        results: list[dict],
        partitions: list[dict],
        official_target: dict[str, Any] | None,
        repository_spans: list[dict],
        *,
        referenced_evidence: tuple[dict[str, list[dict]], dict[str, Any]] | None = None,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Resolve remaining gaps with span-gated target/reference/source evidence."""
        target_spans = self._target_evidence_spans(partitions, official_target)
        if referenced_evidence is None:
            referenced_by_id, referenced_summary = (
                await self._load_referenced_evidence_sources(results)
            )
        else:
            referenced_by_id, referenced_summary = referenced_evidence
        semaphore = asyncio.Semaphore(EVIDENCE_VALIDATION_CONCURRENCY)

        async def validate_one(gap: dict) -> tuple[dict, dict[str, Any]]:
            if is_gap_resolved(gap):
                return gap, {"attempted": False, "reason": "already_resolved"}
            candidate = gap.get("source_candidate") or {}
            referenced = gap.get("referenced_paper") or {}
            arxiv_id = extract_arxiv_id(
                candidate.get("arxiv_id") or referenced.get("arxiv_id") or ""
            )
            scoped_repository_spans = repository_spans_for_gap(repository_spans, gap)
            corpus = list(target_spans)
            if arxiv_id:
                corpus.extend(referenced_by_id.get(arxiv_id, []))
            corpus.extend(scoped_repository_spans)
            # Defensive ID de-duplication: an ID may authorize only one exact span.
            unique_corpus = {
                str(span["span_id"]): span for span in corpus if span.get("span_id")
            }
            corpus = list(unique_corpus.values())
            first, first_record = await self._extract_span_evidence(semaphore, gap, corpus)
            record: dict[str, Any] = {
                "schema_version": 1,
                "gap": self._compact_gap(gap),
                "source_counts": {
                    "target_paper": len(target_spans),
                    "referenced_paper": len(referenced_by_id.get(arxiv_id, [])) if arxiv_id else 0,
                    "scoped_repositories": len(scoped_repository_spans),
                    "total_unique": len(corpus),
                },
                "deterministic_retrieval": first_record,
            }
            chosen = self._prefer_evidence_candidate(gap, first)
            first_quality = self._evidence_quality(first)
            # A partial first pass is useful, but it must not stop retrieval:
            # definitions often arrive before the run command/config and the
            # evaluation consumer that establish the actual choice.  Plan one
            # bounded second pass whenever repository evidence exists and the
            # gap is not fully covered.
            if scoped_repository_spans and not is_gap_resolved(chosen):
                queries, preferred, plan_record = await self._plan_evidence_retrieval(
                    semaphore, gap, corpus,
                )
                record["retrieval_plan"] = plan_record
                if queries or preferred:
                    planned, planned_record = await self._extract_span_evidence(
                        semaphore,
                        gap,
                        corpus,
                        planned_queries=queries,
                        preferred_sources=preferred,
                    )
                    record["planned_retrieval"] = planned_record
                    chosen = self._prefer_evidence_candidate(chosen, planned)
            record["candidate_quality"] = {
                "input": list(self._evidence_quality(gap)),
                "deterministic": list(first_quality),
                "final": list(self._evidence_quality(chosen)),
            }
            record["final_status"] = chosen.get("resolution_status")
            record["actionable"] = is_gap_actionable(chosen)
            self._write_evidence_record(gap.get("id"), record)
            return chosen, {
                "attempted": True,
                "planned": "retrieval_plan" in record,
                "upgraded": chosen is not gap,
            }

        rows = await asyncio.gather(*[validate_one(result) for result in results])
        validated = [row[0] for row in rows]
        attempts = [row[1] for row in rows]
        upgraded = sum(bool(item.get("upgraded")) for item in attempts)
        return validated, {
            "target_span_count": len(target_spans),
            "repository_span_count": len(repository_spans),
            "referenced_papers": referenced_summary,
            "attempted": sum(bool(item.get("attempted")) for item in attempts),
            "planned_retrievals": sum(bool(item.get("planned")) for item in attempts),
            "upgraded_to_actionable": upgraded,
        }

    async def _validate_appendix_candidates(self, results: list[dict]) -> list[dict]:
        """Validate appendix candidates with exact, machine-checked evidence."""
        candidate_indexes = [
            index for index, result in enumerate(results)
            if result.get("resolution_status") == "pending_validation"
            and (result.get("source_candidate") or {}).get("internal_text")
        ]
        if not candidate_indexes:
            return results

        validated = list(results)
        for start in range(0, len(candidate_indexes), APPENDIX_VALIDATION_BATCH_SIZE):
            batch_indexes = candidate_indexes[start:start + APPENDIX_VALIDATION_BATCH_SIZE]
            sources: dict[str, dict] = {}
            gaps = []
            for index in batch_indexes:
                result = results[index]
                candidate = result["source_candidate"]
                source_id = str(candidate.get("content_sha256") or f"source_{index}")
                sources.setdefault(source_id, {
                    "source_id": source_id,
                    "title": candidate.get("title"),
                    "source_sections": candidate.get("source_sections", []),
                    "text": candidate.get("internal_text", ""),
                    "text_truncated": candidate.get("text_truncated", False),
                })
                gaps.append({
                    "id": result.get("id"),
                    "source_id": source_id,
                    "target_name": result.get("target_name"),
                    "description": result.get("description"),
                    "what_is_needed": result.get("what_is_needed"),
                })

            messages = [{
                "role": "user",
                "content": json.dumps(
                    {"sources": list(sources.values()), "gaps": gaps},
                    ensure_ascii=False,
                ),
            }]
            self.log_message("user", f"[appendix_validation] {messages[0]['content'][:500]}...")
            response = ""
            for attempt in range(len(APPENDIX_VALIDATION_RETRY_DELAYS) + 1):
                try:
                    response = await call_minimax(
                        messages=messages,
                        system=APPENDIX_VALIDATION_SYSTEM,
                        max_tokens=8192,
                        stage="gap",
                    )
                    break
                except Exception as exc:
                    if attempt >= len(APPENDIX_VALIDATION_RETRY_DELAYS):
                        logger.warning(
                            "Appendix validation failed closed after %d attempts: %s",
                            attempt + 1,
                            type(exc).__name__,
                        )
                        break
                    delay = APPENDIX_VALIDATION_RETRY_DELAYS[attempt]
                    logger.warning(
                        "Appendix validation attempt %d failed (%s); retrying in %ds",
                        attempt + 1,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
            self.log_message("assistant", f"[appendix_validation] {response}")
            parsed = parse_json(response, fallback={"judgments": []}) if response else {"judgments": []}
            judgments = parsed.get("judgments", []) if isinstance(parsed, dict) else []
            judgment_by_id = {
                str(judgment.get("id")): judgment
                for judgment in judgments
                if isinstance(judgment, dict) and judgment.get("id")
            }
            for index in batch_indexes:
                result = results[index]
                judgment = judgment_by_id.get(str(result.get("id")))
                validated[index] = apply_appendix_judgment(result, judgment)
                final_status = validated[index].get("resolution_status")
                self.log_reasoning(
                    "appendix_validation",
                    f"{result.get('id')}: {final_status}",
                )
        return validated

    @staticmethod
    def _clean_ocr_title(title: str) -> str:
        """Fix common PDF/OCR ligature corruption in titles."""
        fixes = {
            "±": "ffi", "²": "fi", "³": "fl", "¡": "ff",
            "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl",
            "  ": " ",
        }
        for bad, good in fixes.items():
            title = title.replace(bad, good)
        # Fix broken word boundaries from ligature splits (e.g. "ne-tuning" → "fine-tuning")
        title = title.replace("ne-tuning", "fine-tuning")
        title = title.replace("nely-tuned", "finely-tuned")
        return title.strip()

    async def _resolve_from_semantic_scholar(self, ref: dict) -> dict | None:
        """Query S2 API with verification: arxiv ID first, then fuzzy title search."""
        query_title = self._clean_ocr_title(ref.get("title", ""))

        # Try ArXiv ID first (exact, no verification needed)
        arxiv_id = ref.get("arxiv_id", "")
        if arxiv_id:
            clean = arxiv_id.lower().replace("arxiv:", "").strip()
            result = await s2.fetch_paper(f"ARXIV:{clean}")
            if result:
                return result

        if not query_title or len(query_title) < 5:
            return None

        # Try full title search
        result = await s2.search_by_title(query_title)
        if result and self._verify_title_match(query_title, result.get("title", "")):
            return result

        # Try shortened title (first part before colon or dash) for papers like
        # "CoFi: Coarse-to-Fine..." where the short name is more searchable
        short = query_title.split(":")[0].split(" - ")[0].split(" — ")[0].strip()
        if short != query_title and len(short) >= 4:
            result = await s2.search_by_title(short)
            if result and self._verify_title_match(query_title, result.get("title", "")):
                return result

        return None

    def _verify_title_match(self, query: str, result_title: str) -> bool:
        """Verify that an S2 result actually matches our query (reject false positives)."""
        if not query or not result_title:
            return False
        q = query.lower().strip()
        r = result_title.lower().strip()

        ratio = SequenceMatcher(None, q, r).ratio()
        logger.debug(f"Title verify: ratio={ratio:.2f} q='{q[:50]}' r='{r[:50]}'")

        if ratio >= TITLE_VERIFY_THRESHOLD:
            return True

        # Check if enough significant words from query appear in result.
        # Skip short/common words to avoid false matches on "a", "the", "for".
        q_words = {w for w in q.split() if len(w) >= 4}
        r_words = set(r.split())
        if q_words:
            overlap = len(q_words & r_words) / len(q_words)
            if overlap >= 0.4:
                return True

        return False

    # ------------------------------------------------------------------
    # Dataset extraction from paper text
    # ------------------------------------------------------------------

    def _resolve_datasets_from_text(self, partitions: list) -> dict[str, dict]:
        """Scan body and appendices, but not bibliography-only mentions."""
        return scan_text_for_datasets(self._get_relevant_text(partitions))

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_supplementary_context(
        self, resolved: list[dict], dataset_details: dict
    ) -> dict:
        resolved_gaps = [g for g in resolved if is_gap_resolved(g)]
        partial_gaps = [
            g for g in resolved
            if g.get("resolution_status") == "partial" and is_gap_actionable(g)
        ]
        actionable_gaps = [g for g in resolved if is_gap_actionable(g)]
        source_gaps = [g for g in resolved if g.get("resolution_status") == "source_found"]
        missing_source_gaps = [g for g in resolved if g.get("resolution_status") == "unresolved"]
        unresolved_gaps = [g for g in resolved if not is_gap_resolved(g)]

        by_tier: dict[str, int] = {}
        by_category: dict[str, dict] = {}
        for g in resolved:
            tier_key = str(g.get("tier", 3))
            by_tier[tier_key] = by_tier.get(tier_key, 0) + 1
            cat = g.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"total": 0, "resolved": 0}
            by_category[cat]["total"] += 1
            if is_gap_resolved(g):
                by_category[cat]["resolved"] += 1

        return {
            "resolution_summary": {
                "total_gaps": len(resolved),
                "resolved": len(resolved_gaps),
                "partial": len(partial_gaps),
                "source_found": len(source_gaps),
                "unresolved": len(unresolved_gaps),
                "missing_source": len(missing_source_gaps),
                "by_tier": by_tier,
                "by_category": by_category,
            },
            "all_gaps": resolved,
            "actionable_gaps": actionable_gaps,
            "resolved_gaps": resolved_gaps,
            "partial_gaps": partial_gaps,
            "source_gaps": source_gaps,
            "unresolved_gaps": unresolved_gaps,
            "missing_source_gaps": missing_source_gaps,
            "dataset_details": dataset_details,
        }

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_references_section(section: dict) -> bool:
        """True if this section is the bibliography. Prefers `section_role` tag,
        falls back to title-match for partitions from runs before the tag existed."""
        role = section.get("section_role")
        if role is not None:
            return role == "references"
        title = section.get("title", "").lower()
        return any(kw in title for kw in ("reference", "bibliography", "cited"))

    @classmethod
    def _get_relevant_text(cls, partitions: list) -> str:
        """All paper content EXCEPT the references section (that is sent separately).

        Keeps body + appendix + preamble. The gap agent needs to see the appendix
        so it can link gaps to internal references and so the LLM catches dataset
        mentions that only appear in experimental setup sections."""
        parts = []
        for section in partitions:
            if cls._is_references_section(section):
                continue
            parts.append(f"## {section['title']}")
            for item in section["items"]:
                text = item_to_text(item)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    @classmethod
    def _get_references_text(cls, partitions: list) -> str:
        """Extract ONLY the references/bibliography section."""
        for section in partitions:
            if cls._is_references_section(section):
                return "\n".join(
                    t for t in (item_to_text(item) for item in section["items"]) if t
                )
        return ""
