import unittest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.utils.dataset_registry import find_dataset, find_datasets_in_text
from src.utils.gap_matching import (
    attach_target_files_to_context,
    gap_relevant_to_target,
    resolve_dataset_gap,
    route_gap_to_architecture,
)
from src.utils.gap_evidence import (
    add_contextual_evidence_spans,
    apply_span_judgment,
    apply_span_coverage_judgment,
    build_paper_evidence_spans,
    build_repository_evidence_spans,
    gap_retrieval_queries,
    material_coverage_relevance,
    rank_evidence_spans,
    rank_evidence_spans_multi_query,
    select_candidate_spans,
)
from src.utils.gap_resolution import (
    aggregate_appendix_sections,
    apply_appendix_judgment,
    evidence_quote_is_present,
    is_gap_actionable,
    is_gap_resolved,
)
from src.utils.gap_sources import repository_applies_to_gap
from src.utils.paper_content import format_gaps_for_context
from src.utils.paper_source import extract_arxiv_id, parse_arxiv_html, validate_arxiv_html


class DatasetRegistryMatchingTests(unittest.TestCase):
    def test_exact_lookup_accepts_common_aliases(self):
        self.assertEqual(find_dataset("CIFAR-10")["full_name"], "CIFAR-10")
        self.assertEqual(
            find_dataset("SQuAD v2")["full_name"],
            "SQuAD v2.0 (with unanswerable questions)",
        )

    def test_exact_lookup_rejects_generic_and_unknown_names(self):
        for name in ("dataset", "training", "HotpotQA", "EmbedLLM", "UltraFeedback"):
            with self.subTest(name=name):
                self.assertIsNone(find_dataset(name))

    def test_prose_scan_rejects_known_false_matches(self):
        false_match_texts = (
            "Load HotpotQA and reproduce its train and validation splits.",
            "Use the tau-bench retail user simulator and product database.",
            "Sample chess opening positions from the evaluation suite.",
            "Compute representations with EmbedLLM for every model.",
            "Generate preference pairs from the UltraFeedback data.",
        )
        for text in false_match_texts:
            with self.subTest(text=text):
                self.assertEqual(find_datasets_in_text(text), [])

    def test_prose_scan_accepts_bounded_multiword_aliases(self):
        matches = find_datasets_in_text(
            "Train on CIFAR-10, evaluate SQuAD v2, then report GSM8K accuracy."
        )
        self.assertEqual(
            {match["dataset_key"] for match in matches},
            {"cifar10", "squad_v2", "gsm8k"},
        )
        self.assertTrue(all(match["match_confidence"] == "high" for match in matches))
        self.assertNotIn("squad", {match["dataset_key"] for match in matches})

    def test_prose_scan_enforces_boundaries_and_ambiguous_acronym_case(self):
        self.assertEqual(find_datasets_in_text("architectural arcs are compared"), [])
        matches = find_datasets_in_text("Report ARC-Challenge accuracy.")
        self.assertEqual([match["dataset_key"] for match in matches], ["arc"])

    def test_prose_scan_rejects_unregistered_compound_variants(self):
        for name in ("MMLU-Pro", "ImageNet-R", "CIFAR-10-C"):
            with self.subTest(name=name):
                self.assertEqual(find_datasets_in_text(f"Evaluate on {name}."), [])


class DatasetGapResolutionTests(unittest.TestCase):
    def test_unknown_explicit_target_is_not_resolved_from_generic_prose(self):
        result = resolve_dataset_gap({
            "target_name": "HotpotQA",
            "description": "Dataset preprocessing and train split are missing.",
            "what_is_needed": "Find the dataset details.",
        })
        self.assertIsNone(result)

    def test_unregistered_compound_target_is_not_resolved_as_base_dataset(self):
        self.assertIsNone(resolve_dataset_gap({
            "target_name": "MMLU-PRO preference pair construction",
            "description": "The pair construction procedure is missing.",
        }))

    def test_one_explicit_registered_target_resolves_with_evidence(self):
        result = resolve_dataset_gap({
            "target_name": "SQuAD v2",
            "description": "The preprocessing details are missing.",
        })
        self.assertEqual(result["dataset_key"], "squad_v2")
        self.assertEqual(result["match_type"], "bounded_alias")
        self.assertEqual(result["match_confidence"], "high")
        self.assertEqual(result["resolution_scope"], "canonical_registry_metadata")

    def test_multiple_registry_names_remain_unresolved(self):
        result = resolve_dataset_gap({
            "description": "The paper compares SQuAD and GSM8K preprocessing.",
            "what_is_needed": "Determine which dataset split is intended.",
        })
        self.assertIsNone(result)

    def test_registry_does_not_claim_paper_specific_protocol_details(self):
        unsupported = (
            ("GSM8K", "Recover the exact annotation prompt and judge protocol."),
            ("SQuAD", "Determine the specific version and exact text field."),
            ("HellaSwag", "Find the random seed and filtering criteria."),
        )
        for target_name, what_is_needed in unsupported:
            with self.subTest(target_name=target_name):
                self.assertIsNone(resolve_dataset_gap({
                    "target_name": target_name,
                    "description": "Paper-specific details are missing.",
                    "what_is_needed": what_is_needed,
                }))


class GapRoutingTests(unittest.TestCase):
    def test_explicit_target_file_has_precedence(self):
        gap = {
            "category": "dataset_detail",
            "target_files": ["data/hotpotqa.py"],
        }
        self.assertTrue(gap_relevant_to_target(
            gap,
            "data/hotpotqa.py HotpotQA loader",
            target_files=("data/hotpotqa.py",),
        ))
        self.assertFalse(gap_relevant_to_target(
            gap,
            "data/chess.py Chess loader",
            target_files=("data/chess.py",),
        ))

    def test_requirement_id_overlap_has_precedence(self):
        gap = {"category": "algorithm_gap", "requirement_ids": ["Eq. 4"]}
        self.assertTrue(gap_relevant_to_target(
            gap,
            "methods/objective.py",
            requirement_ids=("Equation 4",),
        ))
        self.assertFalse(gap_relevant_to_target(
            gap,
            "methods/objective.py",
            requirement_ids=("Eq. 5",),
        ))

    def test_target_name_routes_named_algorithm(self):
        gap = {"category": "algorithm_gap", "target_name": "Natural Gradient"}
        self.assertTrue(gap_relevant_to_target(
            gap,
            "methods/natural_gradient.py Natural-gradient update",
        ))

    def test_category_fallback_is_disabled_without_explicit_opt_in(self):
        self.assertFalse(gap_relevant_to_target(
            {"category": "algorithm_gap"},
            "methods/solver.py Main optimization method",
        ))
        hyperparameter_gap = {"category": "missing_hyperparameter"}
        self.assertFalse(gap_relevant_to_target(hyperparameter_gap, "config/train.py"))
        self.assertFalse(gap_relevant_to_target(hyperparameter_gap, "plots/figures.py"))
        self.assertTrue(gap_relevant_to_target(
            {"category": "algorithm_gap", "allow_category_fallback": True},
            "methods/solver.py Main optimization method",
        ))

    def test_empty_post_planning_route_is_authoritative(self):
        gap = {
            "category": "dataset_detail",
            "target_name": "HotpotQA",
            "target_files": [],
        }
        self.assertFalse(gap_relevant_to_target(
            gap,
            "data/hotpotqa.py HotpotQA dataset loader",
            target_files=("data/hotpotqa.py",),
        ))

    def test_requirement_mismatch_does_not_block_target_name(self):
        gap = {
            "category": "referenced_architecture",
            "target_name": "TopK Transcoder",
            "requirement_ids": ["Section 3.2.1"],
        }
        self.assertTrue(gap_relevant_to_target(
            gap,
            "models/transcoder.py Implements the TopK-Transcoder architecture",
            requirement_ids=("Algorithm: CRV Pipeline Stage 1",),
        ))


class GapSourceScopeTests(unittest.TestCase):
    def test_target_repository_is_never_available_to_gaps(self):
        self.assertFalse(repository_applies_to_gap(
            {"role": "target_implementation"},
            {"id": "gap_001"},
        ))

    def test_referenced_repository_requires_explicit_scope_match(self):
        definition = {
            "role": "referenced_implementation",
            "ref_ids": ["12"],
            "paper_titles": ["Exact Baseline Paper"],
            "target_names": ["Exact Baseline"],
        }
        self.assertFalse(repository_applies_to_gap(definition, {
            "ref_id": "13",
            "referenced_paper": {"title": "Different Paper"},
            "target_name": "Different Baseline",
        }))
        self.assertTrue(repository_applies_to_gap(definition, {"ref_id": "12"}))
        self.assertTrue(repository_applies_to_gap(definition, {
            "referenced_paper": {"title": "Exact Baseline Paper"},
        }))
        self.assertTrue(repository_applies_to_gap(definition, {
            "target_name": "Exact Baseline",
        }))


class RepositoryAccessPolicyTests(unittest.TestCase):
    def test_gap_cache_requires_exact_blacklist_attestation(self):
        from src.main import _gap_cache_matches_repository_policy

        blacklist = ["https://github.com/example/official-target"]
        context = {"gap_source_summary": {"repository_access_policy": {
            "schema_version": 1,
            "target_codebase_blacklist": blacklist,
            "target_implementations_allowed": False,
        }}}
        self.assertTrue(_gap_cache_matches_repository_policy(context, blacklist))
        self.assertFalse(_gap_cache_matches_repository_policy(context, []))
        self.assertFalse(_gap_cache_matches_repository_policy({}, blacklist))

    def test_gap_agent_policy_canonicalizes_caller_input(self):
        from src.agents.gap_agent import GapAgent
        from src.memory.global_memory import GlobalMemory

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = GlobalMemory(path=str(Path(temp_dir) / "memory"))
            memory.set("target_codebase_blacklist", [
                "https://github.com/Example/Official.GIT/tree/main",
                "https://github.com/example/official",
            ])
            agent = GapAgent(memory, output_dir=temp_dir)
            policy = agent._repository_access_policy()
        self.assertTrue(policy["repository_evidence_enabled"])
        self.assertEqual(
            policy["target_codebase_blacklist"],
            ["https://github.com/example/official"],
        )
        self.assertFalse(policy["target_implementations_allowed"])


class ArchitectureGapRoutingTests(unittest.TestCase):
    def setUp(self):
        self.architecture = {
            "files": [
                {
                    "path": "models/transcoder.py",
                    "purpose": "Implements the TopK-Transcoder architecture",
                    "implements": ["Algorithm: CRV Pipeline Stage 1"],
                    "orchestrator": False,
                },
                {
                    "path": "attribution/greedy_path_finder.py",
                    "purpose": "Greedy attribution graph path finding and graph parameters",
                    "implements": ["Algorithm: CRV Pipeline Stage 2"],
                    "orchestrator": False,
                },
                {
                    "path": "environments/os_tuning_env.py",
                    "purpose": "Run sysbench cpu with thread pinning",
                    "implements": [],
                    "orchestrator": False,
                },
                {
                    "path": "losses/auxdpo.py",
                    "purpose": "AuxDPO loss",
                    "implements": ["Eq. 8 (penalty term)"],
                    "orchestrator": False,
                },
                {
                    "path": "data/unrelated.py",
                    "purpose": "Unrelated dataset loader",
                    "implements": [],
                    "orchestrator": False,
                },
                {
                    "path": "main.py",
                    "purpose": "Orchestrate all stages",
                    "implements": [],
                    "orchestrator": True,
                },
            ]
        }

    def test_named_targets_reach_primary_implementation_files(self):
        cases = (
            (
                {"target_name": "Transcoder (TopK-Transcoder)", "category": "referenced_architecture"},
                "models/transcoder.py",
            ),
            (
                {"target_name": "Attribution graph computation parameters", "category": "missing_hyperparameter"},
                "attribution/greedy_path_finder.py",
            ),
            (
                {"target_name": "sysbench cpu configuration", "category": "dataset_detail"},
                "environments/os_tuning_env.py",
            ),
        )
        for gap, expected in cases:
            with self.subTest(expected=expected):
                paths, evidence = route_gap_to_architecture(gap, self.architecture)
                self.assertIn(expected, paths)
                self.assertNotIn("main.py", paths)
                self.assertTrue(evidence)

    def test_requirement_ids_match_descriptive_suffixes(self):
        paths, evidence = route_gap_to_architecture(
            {
                "target_name": "nullspace constraint",
                "category": "algorithm_gap",
                "requirement_ids": ["Eq. (8)"],
            },
            self.architecture,
        )
        self.assertIn("losses/auxdpo.py", paths)
        aux_evidence = next(item for item in evidence if item["file"] == "losses/auxdpo.py")
        self.assertEqual(aux_evidence["requirement_ids"], ["eq8"])

    def test_context_attachment_updates_every_collection(self):
        gap = {
            "id": "gap_001",
            "target_name": "sysbench cpu configuration",
            "category": "dataset_detail",
            "resolution_status": "partial",
            "resolution": {"internal_text": "verified"},
            "resolution_evidence": {"validated": True, "coverage": "partial"},
        }
        context = {
            "all_gaps": [gap],
            "actionable_gaps": [gap],
            "partial_gaps": [gap],
            "unresolved_gaps": [gap],
        }
        routed = attach_target_files_to_context(context, self.architecture)
        for collection in ("all_gaps", "actionable_gaps", "partial_gaps", "unresolved_gaps"):
            self.assertIn("environments/os_tuning_env.py", routed[collection][0]["target_files"])
        self.assertEqual(routed["routing_summary"]["category_fallback_routes"], 0)


class AppendixResolutionTests(unittest.TestCase):
    def test_evidence_matching_normalizes_latex_unicode_and_html(self):
        source = (
            "The model’s TopK setting is $k = 1 2 8$. "
            "<table><tr><td>Learning Rate</td><td>1e-4</td></tr></table>"
        )
        self.assertTrue(evidence_quote_is_present("The model's TopK setting is k=128", source))
        self.assertTrue(evidence_quote_is_present("Learning Rate 1e-4", source))

    @staticmethod
    def _paragraph(text):
        return {
            "type": "paragraph",
            "content": {"paragraph_content": [{"type": "text", "content": text}]},
        }

    def test_parent_reference_aggregates_descendants_and_table_html(self):
        partitions = [
            {"section_role": "appendix", "title": "A DETAILS", "items": [self._paragraph("Overview only.")]},
            {"section_role": "appendix", "title": "A.1 DATA", "items": [
                self._paragraph("Use random seed 42."),
                {"type": "table", "content": {
                    "table_caption": [{"type": "text", "content": "Hyperparameters"}],
                    "html": "<table><tr><td>batch_size</td><td>16</td></tr></table>",
                }},
            ]},
            {"section_role": "appendix", "title": "A.2 VALIDATION", "items": [self._paragraph("Two annotators reviewed labels.")]},
            {"section_role": "appendix", "title": "B OTHER", "items": [self._paragraph("Must not be included.")]},
        ]
        result = aggregate_appendix_sections("Appendix A", partitions)
        self.assertEqual(result["source_sections"], ["A DETAILS", "A.1 DATA", "A.2 VALIDATION"])
        self.assertIn("Use random seed 42.", result["internal_text"])
        self.assertIn("<td>batch_size</td>", result["internal_text"])
        self.assertNotIn("Must not be included", result["internal_text"])

    def test_child_reference_does_not_include_sibling(self):
        partitions = [
            {"section_role": "appendix", "title": "B.2 HOTPOT", "items": [self._paragraph("Root.")]},
            {"section_role": "appendix", "title": "B.2.1 SETUP", "items": [self._paragraph("Search API format.")]},
            {"section_role": "appendix", "title": "B.3 OTHER", "items": [self._paragraph("Sibling.")]},
        ]
        result = aggregate_appendix_sections("Section B.2", partitions)
        self.assertIn("Search API format.", result["internal_text"])
        self.assertNotIn("Sibling.", result["internal_text"])

    def test_evidence_gate_fails_closed_on_fabricated_quote(self):
        candidate = {
            "id": "gap_001",
            "source_candidate": {"internal_text": "We use batch size 16 and learning rate 1e-4."},
            "resolution_status": "pending_validation",
        }
        result = apply_appendix_judgment(candidate, {
            "status": "resolved",
            "evidence": [{"quote": "We use batch size 512.", "supports": "batch size"}],
            "covered_needs": ["batch size"],
            "missing_needs": [],
        })
        self.assertEqual(result["resolution_status"], "source_found")
        self.assertNotIn("resolution", result)
        self.assertFalse(is_gap_actionable(result))

    def test_partial_evidence_is_actionable_but_not_resolved(self):
        candidate = {
            "id": "gap_001",
            "source_candidate": {"internal_text": "We use batch size 16 and learning rate 1e-4."},
            "resolution_status": "pending_validation",
        }
        result = apply_appendix_judgment(candidate, {
            "status": "partial",
            "evidence": [{"quote": "We use batch size 16", "supports": "batch size"}],
            "covered_needs": ["batch size"],
            "missing_needs": ["random seed"],
        })
        self.assertEqual(result["resolution_status"], "partial")
        self.assertTrue(is_gap_actionable(result))
        self.assertFalse(is_gap_resolved(result))

    def test_mixed_valid_and_invalid_evidence_downgrades_to_partial(self):
        candidate = {
            "id": "gap_001",
            "source_candidate": {"internal_text": "We use AdamW at learning rate 1e-4."},
            "resolution_status": "pending_validation",
        }
        result = apply_appendix_judgment(candidate, {
            "status": "resolved",
            "evidence": [
                {"quote": "We use AdamW at learning rate 1e-4.", "supports": "optimizer"},
                {"quote": "The batch size is 512.", "supports": "batch size"},
            ],
            "covered_needs": ["optimizer", "batch size"],
            "missing_needs": [],
        })
        self.assertEqual(result["resolution_status"], "partial")
        self.assertTrue(is_gap_actionable(result))
        self.assertFalse(is_gap_resolved(result))
        self.assertEqual(result["resolution_evidence"]["covered_needs"], ["optimizer"])
        self.assertIn(
            "unverified evidence: batch size",
            result["resolution_evidence"]["missing_needs"],
        )

    def test_legacy_resolution_without_evidence_is_not_actionable(self):
        self.assertFalse(is_gap_actionable({"resolution": {"abstract": "metadata only"}}))
        self.assertEqual(format_gaps_for_context([
            {"resolution": {"abstract": "metadata only"}, "description": "legacy"}
        ]), "")


class OfficialPaperSourceTests(unittest.TestCase):
    HTML = """<!doctype html><html><body><article class="ltx_document">
<h1 class="ltx_title_document">A Verified Target Paper</h1>
<section><h2>1 Method</h2><p>Body evidence.</p></section>
<section><h2 class="ltx_title_bibliography">References</h2>
<p>A citation that must not become appendix evidence.</p></section>
<section class="ltx_appendix"><h2 class="ltx_title_appendix">Appendix B Details</h2>
<section><h3>B.2 Training Settings</h3>
<p>We set <math alttext="\\lambda=0.3"><semantics><mi>lambda</mi>
<annotation>duplicate math text</annotation></semantics></math> for AuxDPO.</p>
<table><tr><th>Key</th><th>Value</th></tr>
<tr><td>DPOConfig batch size</td><td>4</td></tr></table>
</section></section></article></body></html>"""

    def test_parser_keeps_appendix_after_references_table_and_math(self):
        parsed = parse_arxiv_html(self.HTML)
        self.assertEqual(parsed["title"], "A Verified Target Paper")
        references = [
            section for section in parsed["partitions"]
            if section["section_role"] == "references"
        ]
        self.assertEqual(len(references), 1)
        result = aggregate_appendix_sections("Appendix B.2", parsed["partitions"])
        self.assertIsNotNone(result)
        self.assertIn("DPOConfig batch size | 4", result["internal_text"])
        self.assertIn("$\\lambda=0.3$", result["internal_text"])
        self.assertNotIn("duplicate math text", result["internal_text"])
        self.assertNotIn("citation that must not", result["internal_text"])

    def test_arxiv_identifier_is_bounded_and_version_is_removed(self):
        self.assertEqual(extract_arxiv_id("paper-2510.20413v2/content_list.json"), "2510.20413")
        self.assertIsNone(extract_arxiv_id("run_251020413_output"))

    def test_parser_accepts_arxiv_title_rendered_as_first_article_paragraph(self):
        parsed = parse_arxiv_html("""
        <html><body><article>
          <div class="ltx_para ltx_align_center"><p>
            <span class="ltx_text ltx_font_bold">A General Theoretical Paradigm to
            Understand Learning from Human Preferences</span>
          </p></div>
          <section><h2 class="ltx_title ltx_title_section">1 Introduction</h2>
          <p>Concrete paper content.</p></section>
        </article></body></html>
        """)
        self.assertEqual(
            parsed["title"],
            "A General Theoretical Paradigm to Understand Learning from Human Preferences",
        )
        self.assertEqual(parsed["partitions"][0]["title"], "1 Introduction")

    def test_format_integrity_rejects_truncated_html(self):
        with self.assertRaises(ValueError):
            validate_arxiv_html(self.HTML.encode("utf-8"))


class TargetPaperEvidenceTests(unittest.TestCase):
    def test_material_coverage_relevance_prefers_requested_api_schema(self):
        gap = {
            "what_is_needed": (
                "Complete list of retail APIs with parameters, return types, "
                "and example utterances that trigger each API"
            ),
        }
        peripheral = [
            "Confirmation of retail domain",
            "Total count of retail APIs",
            "How number of calls is determined dynamically",
        ]
        material = [
            "Complete list of retail APIs",
            "API signatures with parameters and return types",
        ]
        self.assertGreater(
            material_coverage_relevance(gap, material),
            material_coverage_relevance(gap, peripheral),
        )

    def test_context_spans_stitch_split_sentence_and_equation_explanation(self):
        spans = [
            {
                "span_id": "paper_s014_i001_c04",
                "section_title": "5.3 MODEL ROUTING",
                "item_type": "paragraph",
                "text": "These DNAs are directly compared",
                "source_type": "target_paper",
            },
            {
                "span_id": "paper_s014_i002_c00",
                "section_title": "5.3 MODEL ROUTING",
                "item_type": "figure",
                "text": "Figure 4",
                "source_type": "target_paper",
            },
            {
                "span_id": "paper_s014_i003_c00",
                "section_title": "5.3 MODEL ROUTING",
                "item_type": "paragraph",
                "text": "with the query embeddings to make routing decisions.",
                "source_type": "target_paper",
            },
            {
                "span_id": "paper_s007_i007_c00",
                "section_title": "4.1 NATURAL GRADIENT",
                "item_type": "equation",
                "text": "$$theta*=theta0+F^-1 g. \tag{5}$$",
                "source_type": "target_paper",
            },
            {
                "span_id": "paper_s007_i008_c00",
                "section_title": "4.1 NATURAL GRADIENT",
                "item_type": "paragraph",
                "text": "This has the form of a natural policy gradient update.",
                "source_type": "target_paper",
            },
        ]
        enriched = add_contextual_evidence_spans(spans)
        contexts = [span for span in enriched if span.get("component_span_ids")]
        self.assertEqual(len(contexts), 2)
        self.assertIn(
            "directly compared\nwith the query embeddings",
            contexts[0]["text"],
        )
        self.assertIn("natural policy gradient update", contexts[1]["text"])
        self.assertEqual(
            contexts[0]["component_span_ids"],
            ["paper_s014_i001_c04", "paper_s014_i003_c00"],
        )

    def test_repository_spans_keep_paths_and_exclude_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "results").mkdir()
            (root / "src" / "config.py").write_text(
                "LEARNING_RATE = 1e-4\nBATCH_SIZE = 16\n",
                encoding="utf-8",
            )
            (root / "results" / "large.json").write_text(
                '{"claimed": "not source code"}',
                encoding="utf-8",
            )
            spans = build_repository_evidence_spans(root)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["source_path"], "src/config.py")
        self.assertIn("BATCH_SIZE = 16", spans[0]["text"])

    def test_repository_notebook_reads_sources_but_never_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "method.ipynb").write_text(json.dumps({
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": ["Training settings\n", "lambda = 1.0"],
                        "outputs": [{"text": ["fabricated output"]}],
                    },
                    {
                        "cell_type": "code",
                        "source": ["batch_size = 4\n", "seed = 42"],
                        "outputs": [{"text": ["claimed accuracy 100%"]}],
                    },
                ],
            }), encoding="utf-8")
            spans = build_repository_evidence_spans(root, id_prefix="official-a")
        combined = "\n".join(span["text"] for span in spans)
        self.assertIn("lambda = 1.0", combined)
        self.assertIn("batch_size = 4", combined)
        self.assertNotIn("fabricated output", combined)
        self.assertNotIn("claimed accuracy", combined)
        self.assertTrue(all(span["span_id"].startswith("official-a_") for span in spans))
        self.assertEqual({span["cell_index"] for span in spans}, {0, 1})

    def test_repository_prefix_prevents_cross_repository_id_collisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.py").write_text("batch_size = 4\n", encoding="utf-8")
            first = build_repository_evidence_spans(root, id_prefix="first")
            second = build_repository_evidence_spans(root, id_prefix="second")
        self.assertNotEqual(first[0]["span_id"], second[0]["span_id"])

    def test_retrieval_ranks_named_and_internal_sections(self):
        spans = [
            {"span_id": "intro", "section_title": "1 Introduction", "text": "General motivation."},
            {
                "span_id": "settings",
                "section_title": "B.2 Training Settings",
                "text": "AuxDPO uses lambda 0.3 with batch size 16.",
            },
            {
                "span_id": "other",
                "section_title": "B.3 Results",
                "text": "Accuracy is reported for three models.",
            },
        ]
        gap = {
            "id": "gap_001",
            "target_name": "AuxDPO lambda",
            "what_is_needed": "lambda value and batch size",
            "internal_ref": "Appendix B.2",
        }
        ranked = rank_evidence_spans(gap, spans, top_k=2)
        self.assertEqual(ranked[0]["span_id"], "settings")
        candidates, ids = select_candidate_spans([gap], spans, per_gap=1)
        self.assertEqual(ids, {"gap_001": ["settings"]})
        self.assertEqual([span["span_id"] for span in candidates], ["settings"])

    def test_multi_query_reserves_evidence_for_each_atomic_need(self):
        spans = [
            {
                "span_id": "formula",
                "section_title": "algorithm.py",
                "source_path": "algorithm.py",
                "text": "def nei_distance(p, q): return -log(nei_similarity(p, q))",
            },
            {
                "span_id": "tokenizer",
                "section_title": "tokenizer.py",
                "source_path": "tokenizer.py",
                "text": "allele = completion.lstrip()[:4]  # first four characters",
            },
        ] + [
            {
                "span_id": f"noise_{index}",
                "section_title": "README.md",
                "source_path": "README.md",
                "text": "PhyloLM distance benchmark model implementation details",
            }
            for index in range(20)
        ]
        gap = {
            "target_name": "PhyloLM",
            "what_is_needed": (
                "(1) exact Nei distance formula; "
                "(2) cross-tokenizer approximation using first four characters"
            ),
        }
        self.assertGreaterEqual(len(gap_retrieval_queries(gap)), 4)
        ranked = rank_evidence_spans_multi_query(
            gap, spans, top_k=8, per_query=4,
        )
        self.assertTrue({"formula", "tokenizer"} <= {span["span_id"] for span in ranked})

    def test_multi_query_uses_exact_planned_paths_and_neighbor_context(self):
        spans = [
            {
                "span_id": "config_0",
                "source_path": "config.yml",
                "section_title": "config.yml",
                "text": "prompts:\n  standard_game: reason extensively",
            },
            {
                "span_id": "config_1",
                "source_path": "config.yml",
                "section_title": "config.yml",
                "text": "guess: reason succinctly and return three moves",
            },
            {
                "span_id": "call_0",
                "source_path": "actor.py",
                "section_title": "actor.py",
                "text": "reasoning = {'effort': 'high'}",
            },
        ] + [
            {
                "span_id": f"noise_{index}",
                "source_path": "README.md",
                "section_title": "README.md",
                "text": "Actor Speculator system prompt overview",
            }
            for index in range(20)
        ]
        ranked = rank_evidence_spans_multi_query(
            {
                "target_name": "Actor / Speculator",
                "what_is_needed": "complete prompts and reasoning effort",
            },
            spans,
            top_k=8,
            per_query=3,
            extra_queries=["standard_game guess reasoning effort"],
            preferred_paths=["config.yml"],
            neighbor_radius=1,
        )
        selected = {span["span_id"] for span in ranked}
        self.assertTrue({"config_0", "config_1"} <= selected)

    def test_implementation_queries_retrieve_entrypoint_and_consumer(self):
        spans = [
            {
                "span_id": "definition",
                "source_path": "score.py",
                "section_title": "score.py",
                "text": "def compute_CoE_R(): pass\ndef compute_CoE_C(): pass",
            },
            {
                "span_id": "entrypoint",
                "source_path": "slurm_main.sh",
                "section_title": "slurm_main.sh",
                "text": (
                    "/project/Chain-of-Embedding/main.py --save_coe_score "
                    "--token_aggregation first"
                ),
            },
            {
                "span_id": "consumer",
                "source_path": "Evaluation/eval.py",
                "section_title": "Evaluation/eval.py",
                "text": (
                    'coer_list = [coe["R"] for coe in coe_list]\n'
                    'coec_list = [coe["C"] for coe in coe_list]\n'
                    "coer_auroc = self_eval(coer_list)\n"
                    "coec_auroc = self_eval(coec_list)"
                ),
            },
        ] + [
            {
                "span_id": f"noise_{index}",
                "source_path": "README.md",
                "section_title": "README.md",
                "text": "Chain-of-Embedding baseline overview",
            }
            for index in range(12)
        ]
        gap = {
            "category": "baseline_method",
            "target_name": "Chain-of-Embedding (CoE-R, CoE-C)",
            "description": "Implementation details are missing.",
            "what_is_needed": "variant, hyperparameters, and preprocessing steps",
        }
        queries = gap_retrieval_queries(gap)
        self.assertTrue(any("entrypoint" in query for query in queries))
        self.assertTrue(any("consumer" in query for query in queries))
        ranked = rank_evidence_spans_multi_query(
            gap, spans, top_k=10, per_query=5,
        )
        selected = {span["span_id"] for span in ranked}
        self.assertTrue({"definition", "entrypoint", "consumer"} <= selected)

    def test_planned_repository_and_path_pair_does_not_broadcast_by_filename(self):
        spans = [
            {
                "span_id": "wrong",
                "repository": "other-repo",
                "source_path": "README.md",
                "section_title": "README.md",
                "text": "generic baseline details",
            },
            {
                "span_id": "right",
                "repository": "expected-repo",
                "source_path": "README.md",
                "section_title": "README.md",
                "text": "exact obscure setting zeta_config = 17",
            },
        ]
        ranked = rank_evidence_spans_multi_query(
            {"target_name": "unmatched", "what_is_needed": "unknown setting"},
            spans,
            top_k=1,
            per_query=1,
            preferred_sources=[("expected-repo", "README.md")],
        )
        self.assertEqual([span["span_id"] for span in ranked], ["right"])

    def test_spans_exclude_references_and_preserve_table_rows(self):
        partitions = [
            {
                "title": "4 Experiments",
                "section_role": "body",
                "items": [{
                    "type": "table",
                    "content": {
                        "table_caption": [{"type": "text", "content": "Settings"}],
                        "html": "<table><tr><td>batch size</td><td>16</td></tr></table>",
                    },
                }],
            },
            {
                "title": "References",
                "section_role": "references",
                "items": [{
                    "type": "paragraph",
                    "content": {"paragraph_content": [{"type": "text", "content": "Citation only"}]},
                }],
            },
        ]
        spans = build_paper_evidence_spans(partitions)
        self.assertEqual(len(spans), 2)
        self.assertIn("batch size | 16", spans[1]["text"])
        self.assertNotIn("Citation only", " ".join(span["text"] for span in spans))
        self.assertEqual(spans[0]["span_id"], "s000_i000_c00")

    def test_span_id_gate_uses_source_text_not_model_quote(self):
        spans = [{
            "span_id": "s001_i002_c00",
            "section_title": "4.2 Method",
            "text": "We use AdamW with learning rate 1e-4 and batch size 16.",
        }]
        result = apply_span_judgment({"id": "gap_001"}, {
            "status": "resolved",
            "evidence": [{"span_id": "s001_i002_c00", "supports": "training settings"}],
            "covered_needs": ["optimizer", "learning rate", "batch size"],
            "missing_needs": [],
        }, spans)
        self.assertTrue(is_gap_resolved(result))
        self.assertEqual(
            result["resolution_evidence"]["evidence"][0]["quote"],
            spans[0]["text"],
        )

    def test_unknown_span_id_fails_closed(self):
        result = apply_span_judgment({"id": "gap_001"}, {
            "status": "resolved",
            "evidence": [{"span_id": "invented", "supports": "settings"}],
            "covered_needs": ["settings"],
            "missing_needs": [],
        }, [])
        self.assertEqual(result["resolution_status"], "source_found")
        self.assertFalse(is_gap_actionable(result))

    def test_coverage_rows_derive_partial_instead_of_trusting_summary_label(self):
        spans = [{
            "span_id": "settings",
            "section_title": "4 Experiments",
            "text": "The dataset is split 80/20 for training and evaluation.",
        }]
        result = apply_span_coverage_judgment({"id": "gap_001"}, {
            "coverage": [
                {"need": "split ratio", "status": "supported", "span_ids": ["settings"]},
                {"need": "random seed", "status": "missing", "span_ids": []},
            ],
            "coverage_complete": False,
        }, spans)
        self.assertEqual(result["resolution_status"], "partial")
        self.assertTrue(is_gap_actionable(result))
        self.assertEqual(result["resolution_evidence"]["missing_needs"], ["random seed"])


class GapAgentCascadeTests(unittest.IsolatedAsyncioTestCase):
    async def test_automatic_repository_discovery_is_loaded_with_configured_urls_excluded(self):
        from src.agents.gap_agent import GapAgent
        from src.memory.global_memory import GlobalMemory

        partitions = [{
            "title": "A Verified Target Paper",
            "section_role": "body",
            "items": [],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = GlobalMemory(path=str(Path(temp_dir) / "memory"))
            memory.set(
                "target_codebase_blacklist",
                ["https://github.com/example/official-target"],
            )
            agent = GapAgent(memory, output_dir=temp_dir)
            agent.log_reasoning = lambda *args, **kwargs: None
            discovered = {
                "manifest_path": str(Path(temp_dir) / "gap_sources/auto_gap_sources.json"),
                "accepted": [{"identity": "example/auto"}],
            }
            loaded_spans = [{
                "span_id": "auto_f0000_c0000",
                "repository": "example--auto",
                "repository_role": "referenced_implementation",
                "text": "batch_size = 16",
            }]
            with (
                patch(
                    "src.agents.gap_agent.s2.search_by_title",
                    AsyncMock(return_value=None),
                ),
                patch(
                    "src.agents.gap_agent.discover_repository_manifest",
                    return_value=discovered,
                ) as discover,
                patch(
                    "src.agents.gap_agent.load_gap_source_manifest",
                    return_value=(loaded_spans, {"accepted": [{"name": "example--auto"}]}),
                ),
            ):
                spans, summary = await agent._load_automatic_repository_evidence_sources(
                    [],
                    partitions,
                    None,
                    {"sources": []},
                    {"accepted": [{"url": "https://github.com/example/configured"}]},
                )

        self.assertEqual(spans, loaded_spans)
        self.assertTrue(summary["loaded"])
        kwargs = discover.call_args.kwargs
        self.assertEqual(kwargs["target_title"], "A Verified Target Paper")
        self.assertEqual(
            kwargs["excluded_repository_urls"],
            ["https://github.com/example/configured"],
        )
        self.assertEqual(
            kwargs["target_codebase_blacklist"],
            ["https://github.com/example/official-target"],
        )

    async def test_automatic_repository_discovery_reads_target_html_for_blacklist_matching(self):
        from src.agents.gap_agent import GapAgent
        from src.memory.global_memory import GlobalMemory

        partitions = [{
            "title": "A Verified Target Paper",
            "section_role": "body",
            "items": [],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = GlobalMemory(path=str(Path(temp_dir) / "memory"))
            memory.set(
                "target_codebase_blacklist",
                ["https://github.com/example/official-target"],
            )
            agent = GapAgent(memory, output_dir=temp_dir)
            agent.log_reasoning = lambda *args, **kwargs: None
            fetched = {
                "arxiv_id": "2510.20413",
                "source_url": "https://arxiv.org/html/2510.20413",
                "sha256": "a" * 64,
                "cache_path": str(Path(temp_dir) / "gap_sources/arxiv/2510.20413/paper.html"),
                "html": OfficialPaperSourceTests.HTML,
            }
            discovered = {
                "manifest_path": str(Path(temp_dir) / "gap_sources/auto_gap_sources.json"),
                "accepted": [],
            }
            with (
                patch(
                    "src.agents.gap_agent.s2.search_by_title",
                    AsyncMock(return_value={
                        "title": "A Verified Target Paper",
                        "arxiv_id": "2510.20413",
                    }),
                ),
                patch(
                    "src.agents.gap_agent.fetch_arxiv_html",
                    AsyncMock(return_value=fetched),
                ),
                patch(
                    "src.agents.gap_agent.discover_repository_manifest",
                    return_value=discovered,
                ) as discover,
                patch(
                    "src.agents.gap_agent.load_gap_source_manifest",
                    return_value=([], {"accepted": []}),
                ),
            ):
                await agent._load_automatic_repository_evidence_sources(
                    [], partitions, None, {"sources": []}, {"loaded": False},
                )

        kwargs = discover.call_args.kwargs
        self.assertEqual(kwargs["target_arxiv_id"], "2510.20413")
        self.assertEqual(len(kwargs["html_documents"]), 1)
        self.assertEqual(kwargs["html_documents"][0]["document_role"], "target_paper")
        self.assertEqual(kwargs["html_documents"][0]["source_sha256"], "a" * 64)

    async def test_missing_blacklist_disables_repository_discovery_before_network(self):
        from src.agents.gap_agent import GapAgent
        from src.memory.global_memory import GlobalMemory

        partitions = [{
            "title": "A Verified Target Paper",
            "section_role": "body",
            "items": [],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = GlobalMemory(path=str(Path(temp_dir) / "memory"))
            agent = GapAgent(memory, output_dir=temp_dir)
            with (
                patch("src.agents.gap_agent.fetch_arxiv_html", new=AsyncMock()) as fetch,
                patch("src.agents.gap_agent.discover_repository_manifest") as discover,
            ):
                spans, summary = await agent._load_automatic_repository_evidence_sources(
                    [], partitions, None, {"sources": []}, {"accepted": []},
                )
        self.assertFalse(spans)
        self.assertFalse(summary["loaded"])
        self.assertEqual(summary["reason"], "target_codebase_blacklist_missing")
        fetch.assert_not_awaited()
        discover.assert_not_called()

    async def test_semantic_scholar_hit_is_source_found_not_resolved(self):
        from src.agents.gap_agent import GapAgent

        agent = GapAgent.__new__(GapAgent)
        gap = {
            "id": "gap_001",
            "category": "baseline_method",
            "referenced_paper": {"title": "Exact source paper"},
        }
        with patch.object(
            agent,
            "_resolve_from_semantic_scholar",
            AsyncMock(return_value={"title": "Exact source paper", "abstract": "High-level only"}),
        ):
            result = await agent._resolve_gap(gap, [])
        self.assertEqual(result["resolution_status"], "source_found")
        self.assertNotIn("resolution", result)
        self.assertEqual(result["source_candidate"]["title"], "Exact source paper")

    async def test_appendix_batch_validation_requires_present_quote(self):
        from src.agents.gap_agent import GapAgent

        agent = GapAgent.__new__(GapAgent)
        agent.log_message = lambda *args, **kwargs: None
        agent.log_reasoning = lambda *args, **kwargs: None
        candidate = {
            "id": "gap_001",
            "target_name": "optimizer",
            "what_is_needed": "optimizer and learning rate",
            "source_candidate": {
                "title": "A.1 Training",
                "source_sections": ["A.1 Training"],
                "content_sha256": "source-one",
                "internal_text": "We train with AdamW at learning rate 1e-4.",
            },
            "resolution_status": "pending_validation",
            "tier": 0.5,
        }
        response = """{"judgments":[{"id":"gap_001","status":"resolved","evidence":[{"quote":"We train with AdamW at learning rate 1e-4.","supports":"optimizer and learning rate"}],"covered_needs":["optimizer","learning rate"],"missing_needs":[],"rationale":"complete"}]}"""
        with patch("src.agents.gap_agent.call_minimax", AsyncMock(return_value=response)):
            results = await agent._validate_appendix_candidates([candidate])
        self.assertTrue(is_gap_resolved(results[0]))
        self.assertEqual(results[0]["resolution_evidence"]["coverage"], "complete")

    async def test_appendix_validator_failure_fails_closed(self):
        from src.agents.gap_agent import GapAgent

        agent = GapAgent.__new__(GapAgent)
        agent.log_message = lambda *args, **kwargs: None
        agent.log_reasoning = lambda *args, **kwargs: None
        candidate = {
            "id": "gap_001",
            "what_is_needed": "exact optimizer settings",
            "source_candidate": {
                "title": "A.1 Training",
                "source_sections": ["A.1 Training"],
                "content_sha256": "source-one",
                "internal_text": "We train with AdamW at learning rate 1e-4.",
            },
            "resolution_status": "pending_validation",
            "tier": 0.5,
        }
        with (
            patch("src.agents.gap_agent.APPENDIX_VALIDATION_RETRY_DELAYS", ()),
            patch(
                "src.agents.gap_agent.call_minimax",
                AsyncMock(side_effect=RuntimeError("temporary provider error")),
            ),
        ):
            results = await agent._validate_appendix_candidates([candidate])
        self.assertEqual(results[0]["resolution_status"], "source_found")
        self.assertNotIn("resolution", results[0])
        self.assertFalse(is_gap_actionable(results[0]))

    async def test_missing_local_appendix_uses_title_verified_official_html(self):
        from src.agents.gap_agent import GapAgent
        from src.memory.global_memory import GlobalMemory

        partitions = [{
            "title": "A VERIFIED TARGET PAPER",
            "section_role": "body",
            "items": [],
        }]
        gaps = [{
            "id": "gap_001",
            "category": "missing_hyperparameter",
            "internal_ref": "Appendix B.2",
            "what_is_needed": "AuxDPO lambda and batch size",
            "referenced_paper": {},
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = GlobalMemory(path=str(Path(temp_dir) / "memory"))
            agent = GapAgent(memory, output_dir=temp_dir)
            fetched = {
                "arxiv_id": "2510.20413",
                "source_url": "https://arxiv.org/html/2510.20413",
                "sha256": "a" * 64,
                "size_bytes": 12345,
                "from_cache": True,
                "cache_path": str(Path(temp_dir) / "gap_sources/arxiv/2510.20413/paper.html"),
                "html": OfficialPaperSourceTests.HTML,
            }
            with (
                patch(
                    "src.agents.gap_agent.s2.search_by_title",
                    AsyncMock(return_value={
                        "title": "A Verified Target Paper",
                        "arxiv_id": "2510.20413",
                    }),
                ),
                patch("src.agents.gap_agent.fetch_arxiv_html", AsyncMock(return_value=fetched)),
            ):
                official = await agent._load_missing_official_target_source(partitions, gaps)
            self.assertIsNotNone(official)
            result = await agent._resolve_gap(gaps[0], partitions, official)

        self.assertEqual(result["resolution_status"], "pending_validation")
        self.assertEqual(result["tier"], 0.4)
        candidate = result["source_candidate"]
        self.assertEqual(candidate["source_type"], "target_paper_official_html")
        self.assertEqual(candidate["source_sha256"], "a" * 64)
        self.assertEqual(
            candidate["cache_file"], "gap_sources/arxiv/2510.20413/paper.html",
        )
        self.assertIn("DPOConfig batch size", candidate["internal_text"])

    async def test_multisource_extractor_accepts_only_known_span_and_keeps_provenance(self):
        from src.agents.gap_agent import GapAgent

        agent = GapAgent.__new__(GapAgent)
        gap = {
            "id": "gap_001",
            "target_name": "AuxDPO",
            "what_is_needed": "batch size and random seed",
        }
        spans = [{
            "span_id": "repo_f0001_c0001",
            "section_title": "train.py",
            "source_path": "train.py",
            "source_type": "official_repository",
            "repository": "auxdpo",
            "repository_role": "referenced_implementation",
            "repository_commit": "abc123",
            "text": "BATCH_SIZE = 4",
        }]
        response = json.dumps({
            "id": "gap_001",
            "coverage": [
                {
                    "need": "batch size",
                    "status": "supported",
                    "span_ids": ["repo_f0001_c0001"],
                    "explanation": "explicit constant",
                },
                {
                    "need": "random seed",
                    "status": "missing",
                    "span_ids": [],
                    "explanation": "not present",
                },
            ],
            "coverage_complete": False,
            "rationale": "one of two needs is supported",
        })
        with patch("src.agents.gap_agent.call_minimax", AsyncMock(return_value=response)):
            result, record = await agent._extract_span_evidence(
                asyncio.Semaphore(1), gap, spans,
            )
        self.assertTrue(is_gap_actionable(result))
        self.assertEqual(result["resolution_status"], "partial")
        evidence = result["resolution_evidence"]["evidence"][0]
        self.assertEqual(evidence["repository"], "auxdpo")
        self.assertEqual(evidence["repository_commit"], "abc123")
        self.assertEqual(evidence["quote"], "BATCH_SIZE = 4")
        self.assertEqual(record["invalid_span_ids"], [])

    async def test_partial_first_pass_still_runs_planned_retrieval_and_keeps_better_coverage(self):
        from src.agents.gap_agent import GapAgent

        def partial(covered, missing, span_ids):
            return {
                "id": "gap_008",
                "resolution_status": "partial",
                "resolution_evidence": {
                    "validated": True,
                    "coverage": "partial",
                    "covered_needs": covered,
                    "missing_needs": missing,
                    "invalid_span_ids": [],
                    "evidence": [
                        {"span_id": span_id, "quote": span_id, "supports": covered[0]}
                        for span_id in span_ids
                    ],
                },
            }

        input_gap = partial(["formula"], ["run setting", "variant use"], ["definition"])
        first = partial(["formula"], ["run setting", "variant use"], ["definition"])
        planned = partial(
            ["formula", "run setting", "both variants evaluated"],
            ["target-paper-specific choice"],
            ["definition", "entrypoint", "consumer"],
        )
        agent = GapAgent.__new__(GapAgent)
        agent.output_dir = Path("unused")
        agent._target_evidence_spans = lambda partitions, official: []
        agent._load_referenced_evidence_sources = AsyncMock(return_value=({}, {}))
        agent._extract_span_evidence = AsyncMock(side_effect=[
            (first, {"pass": "deterministic"}),
            (planned, {"pass": "planned"}),
        ])
        agent._plan_evidence_retrieval = AsyncMock(return_value=(
            ["actual run command", "evaluation consumer"],
            [("chain-of-embedding", "slurm_main.sh")],
            {"plan": "bounded"},
        ))
        agent._write_evidence_record = lambda *args, **kwargs: None

        with patch(
            "src.agents.gap_agent.repository_spans_for_gap",
            return_value=[{"span_id": "entrypoint"}],
        ):
            results, summary = await agent._validate_multisource_evidence(
                [input_gap], [], None, [{"span_id": "entrypoint"}],
            )

        self.assertEqual(agent._extract_span_evidence.await_count, 2)
        agent._plan_evidence_retrieval.assert_awaited_once()
        self.assertEqual(
            results[0]["resolution_evidence"]["covered_needs"],
            planned["resolution_evidence"]["covered_needs"],
        )
        self.assertEqual(summary["planned_retrievals"], 1)


if __name__ == "__main__":
    unittest.main()
