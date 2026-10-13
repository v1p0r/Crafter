import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.gap_source_discovery import (
    associate_repository_candidates,
    canonical_github_repository_url,
    discover_repository_manifest,
    extract_github_repository_urls,
    normalize_github_repository_blacklist,
    repository_candidates_from_arxiv_html,
    repository_candidates_from_partitions,
)
from src.utils.gap_sources import load_gap_source_manifest


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _make_repository(repository: Path, url: str, readme: str) -> dict[str, str]:
    repository.mkdir(parents=True)
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "Gap Source Tests")
    (repository / "README.md").write_text(readme, encoding="utf-8")
    (repository / "method.py").write_text("BATCH_SIZE = 16\n", encoding="utf-8")
    _git(repository, "add", "README.md", "method.py")
    _git(repository, "commit", "-m", "fixture")
    _git(repository, "remote", "add", "origin", url)
    return {
        "commit": _git(repository, "rev-parse", "HEAD"),
        "tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "origin": url,
    }


class GithubUrlExtractionTests(unittest.TestCase):
    def test_canonicalizes_ocr_spaces_and_nested_paths(self):
        self.assertEqual(
            extract_github_repository_urls(
                "Code: https://github.com/ Xtra-Computing / LLM-DNA/tree/main."
            ),
            ["https://github.com/Xtra-Computing/LLM-DNA"],
        )
        self.assertEqual(
            canonical_github_repository_url(
                "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md"
            ),
            "https://github.com/meta-llama/llama3",
        )
        self.assertEqual(canonical_github_repository_url("https://github.com/deepmind"), "")

    def test_blacklist_normalizes_identity_and_rejects_non_repository_urls(self):
        self.assertEqual(
            normalize_github_repository_blacklist([
                "https://github.com/Example/Target.git/tree/main",
                "https://github.com/example/target",
            ]),
            ["https://github.com/example/target"],
        )
        with self.assertRaisesRegex(ValueError, "invalid_target_codebase_blacklist_url"):
            normalize_github_repository_blacklist(["https://example.com/not-github"])

    def test_html_reads_article_href_but_ignores_arxiv_footer(self):
        html = """<html><body><article>
        <h1>A Useful Method</h1><p>Our code is <a
        href="https://github.com/example/useful-method/tree/main">available here</a>.</p>
        </article><footer><a href="https://github.com/arXiv/html_feedback/issues">feedback</a>
        </footer></body></html>"""
        candidates = repository_candidates_from_arxiv_html(
            html,
            source={"document_role": "target_paper", "paper_title": "A Useful Method"},
        )
        self.assertEqual([row["identity"] for row in candidates], ["example/useful-method"])
        self.assertIn("Our code", candidates[0]["context"])


class RepositoryScopeDiscoveryTests(unittest.TestCase):
    def test_partitions_bind_target_and_reference_links_without_broadcasting(self):
        partitions = [
            {
                "title": "1 Introduction",
                "section_role": "body",
                "items": [{
                    "type": "paragraph",
                    "content": {"paragraph_content": [{
                        "type": "text",
                        "content": (
                            "Our code is publicly available at "
                            "https://github.com/example/target-method."
                        ),
                    }]},
                }],
            },
            {
                "title": "References",
                "section_role": "references",
                # MinerU commonly stores the entire bibliography as one list
                # item containing many rows. Each row must retain its own
                # context or an unrelated GitHub URL can inherit another
                # citation's title/arXiv ID.
                "items": [{
                    "type": "list",
                    "content": {"list_items": [
                        {"item_content": [{
                            "type": "text",
                            "content": (
                                "Smith. Exact Baseline Paper. "
                                "https://github.com/example/exact-baseline"
                            ),
                        }]},
                        {"item_content": [{
                            "type": "text",
                            "content": (
                                "Unrelated model card. "
                                "https://github.com/example/unrelated"
                            ),
                        }]},
                    ]},
                }],
            },
        ]
        gaps = [{
            "id": "gap_001",
            "ref_id": "12",
            "target_name": "Exact Baseline",
            "referenced_paper": {"title": "Exact Baseline Paper"},
        }]
        candidates = repository_candidates_from_partitions(
            partitions,
            source={"document_role": "target_paper", "paper_title": "Target Method"},
        )
        associated, rejected = associate_repository_candidates(
            candidates,
            gaps=gaps,
            target_title="Target Method",
        )
        by_identity = {row["identity"]: row for row in associated}
        self.assertEqual(
            by_identity["example/target-method"]["association"]["role"],
            "target_implementation",
        )
        reference = by_identity["example/exact-baseline"]["association"]
        self.assertEqual(reference["role"], "referenced_implementation")
        self.assertEqual(reference["ref_ids"], ["12"])
        self.assertNotIn("example/unrelated", by_identity)
        self.assertIn("no_paper_or_gap_scope_match", {row["reason"] for row in rejected})

    def test_page_footer_without_scheme_is_still_extracted(self):
        partitions = [{
            "title": "Abstract",
            "section_role": "body",
            "items": [{
                "type": "page_footer",
                "content": {"page_footer_content": [{
                    "type": "text", "content": "§ github.com/Xtra-Computing/LLM-DNA",
                }]},
            }],
        }]
        candidates = repository_candidates_from_partitions(
            partitions,
            source={"document_role": "target_paper", "paper_title": "LLM DNA"},
        )
        self.assertEqual([row["identity"] for row in candidates], ["xtra-computing/llm-dna"])


class RepositoryManifestMaterializationTests(unittest.TestCase):
    def test_discovery_blacklists_target_and_loads_scoped_reference(self):
        partitions = [{
            "title": "1 Introduction",
            "section_role": "body",
            "items": [{
                "type": "paragraph",
                "content": {"paragraph_content": [{
                    "type": "text",
                    "content": (
                        "Our code is publicly available at "
                        "https://github.com/example/verified-method."
                    ),
                }]},
            }],
        }]
        gaps = [{
            "id": "gap_001",
            "ref_id": "12",
            "target_name": "Exact Baseline",
            "referenced_paper": {"title": "Exact Baseline Paper"},
        }]
        reference_html = """<article><h1>Exact Baseline Paper</h1><p>
        Official implementation: <a href="https://github.com/example/exact-baseline">
        code</a>.</p></article>"""

        def fake_clone(url: str, destination: Path, *, timeout: int):
            del timeout
            self.assertEqual(url, "https://github.com/example/exact-baseline")
            return _make_repository(
                destination,
                url,
                "# Exact Baseline\nCode for the paper Exact Baseline Paper.\n",
            ), False

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.utils.gap_source_discovery._clone_or_reuse_repository",
            side_effect=fake_clone,
        ):
            summary = discover_repository_manifest(
                output_dir=temp_dir,
                target_partitions=partitions,
                target_title="Verified Method",
                gaps=gaps,
                target_codebase_blacklist=[
                    "https://github.com/EXAMPLE/verified-method/tree/main"
                ],
                html_documents=[{
                    "document_role": "referenced_paper",
                    "paper_title": "Exact Baseline Paper",
                    "html": reference_html,
                }],
            )
            manifest_path = Path(summary["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            definition = manifest["repositories"][0]
            self.assertFalse(Path(definition["path"]).is_absolute())
            self.assertEqual(definition["role"], "referenced_implementation")
            self.assertEqual(definition["commit"], summary["accepted"][0]["commit"])
            self.assertIn(
                "target_codebase_blacklisted",
                {row["reason"] for row in summary["rejected"]},
            )
            spans, loaded = load_gap_source_manifest(
                manifest_path,
                target_codebase_blacklist=["https://github.com/example/verified-method"],
            )

        self.assertEqual(len(loaded["accepted"]), 1)
        self.assertFalse(loaded["rejected"])
        self.assertIn("BATCH_SIZE = 16", "\n".join(span["text"] for span in spans))
        self.assertTrue(all(span["repository_commit"] == definition["commit"] for span in spans))

    def test_blacklisted_repo_is_blocked_before_clone_even_when_scoped_as_reference(self):
        target_url = "https://github.com/example/target-method"
        partitions = [{
            "title": "Abstract",
            "section_role": "body",
            "items": [{
                "type": "paragraph",
                "content": {"paragraph_content": [{
                    "type": "text", "content": f"Our code is available at {target_url}.",
                }]},
            }],
        }]
        gaps = [{
            "id": "gap_001",
            "ref_id": "12",
            "referenced_paper": {"title": "Exact Baseline Paper"},
        }]
        html = (
            "<article><h1>Exact Baseline Paper</h1><p>Exact Baseline Paper code: "
            f"<a href=\"{target_url}\">repository</a>.</p></article>"
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.utils.gap_source_discovery._clone_or_reuse_repository",
        ) as clone:
            summary = discover_repository_manifest(
                output_dir=temp_dir,
                target_partitions=partitions,
                target_title="Target Method",
                gaps=gaps,
                target_codebase_blacklist=[target_url + "/tree/main"],
                html_documents=[{
                    "document_role": "referenced_paper",
                    "paper_title": "Exact Baseline Paper",
                    "html": html,
                }],
            )
        clone.assert_not_called()
        self.assertFalse(summary["accepted"])
        self.assertTrue(summary["rejected"])
        self.assertEqual(
            {row["reason"] for row in summary["rejected"]},
            {"target_codebase_blacklisted"},
        )

    def test_discovery_requires_explicit_blacklist_before_clone(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.utils.gap_source_discovery._clone_or_reuse_repository",
        ) as clone:
            with self.assertRaisesRegex(ValueError, "target_codebase_blacklist_required"):
                discover_repository_manifest(
                    output_dir=temp_dir,
                    target_partitions=[],
                    target_title="Target Method",
                    gaps=[],
                    target_codebase_blacklist=[],
                )
        clone.assert_not_called()

    def test_manifest_rejects_checkout_moved_to_another_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "repositories" / "example--repo"
            metadata = _make_repository(
                repository,
                "https://github.com/example/repo",
                "# Example Repository\n",
            )
            manifest = root / "gap_sources.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "repositories": [{
                    "name": "example--repo",
                    "role": "referenced_implementation",
                    "path": "repositories/example--repo",
                    "url": "https://github.com/example/repo",
                    "commit": "0" * len(metadata["commit"]),
                    "ref_ids": ["12"],
                }],
            }), encoding="utf-8")
            spans, summary = load_gap_source_manifest(
                manifest,
                target_codebase_blacklist=["https://github.com/example/target"],
            )
        self.assertFalse(spans)
        self.assertEqual(summary["rejected"][0]["reason"], "commit_mismatch")

    def test_manifest_blocks_target_role_and_blacklist_before_filesystem_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "gap_sources.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "repositories": [
                    {
                        "name": "explicit-blacklist",
                        "role": "referenced_implementation",
                        "path": "does-not-exist-a",
                        "url": "https://github.com/example/target",
                    },
                    {
                        "name": "forbidden-role",
                        "role": "target_implementation",
                        "path": "does-not-exist-b",
                        "url": "https://github.com/example/other-target",
                    },
                ],
            }), encoding="utf-8")
            spans, summary = load_gap_source_manifest(
                manifest,
                target_codebase_blacklist=["https://github.com/EXAMPLE/target.git"],
            )
        self.assertFalse(spans)
        self.assertEqual(
            [row["reason"] for row in summary["rejected"]],
            ["target_codebase_blacklisted", "target_implementation_forbidden"],
        )

    def test_manifest_checks_actual_origin_against_blacklist_before_source_spans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "target-checkout"
            _make_repository(
                repository,
                "https://github.com/example/official-target",
                "# Target implementation\nSECRET_REFERENCE_ANSWER = 42\n",
            )
            manifest = root / "gap_sources.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "repositories": [{
                    "name": "misdeclared-reference",
                    "role": "referenced_implementation",
                    "path": "target-checkout",
                    "url": "https://github.com/example/claimed-reference",
                    "ref_ids": ["12"],
                }],
            }), encoding="utf-8")
            with patch(
                "src.utils.gap_sources.build_repository_evidence_spans",
            ) as build_spans:
                spans, summary = load_gap_source_manifest(
                    manifest,
                    target_codebase_blacklist=[
                        "https://github.com/example/official-target"
                    ],
                )
        build_spans.assert_not_called()
        self.assertFalse(spans)
        self.assertEqual(
            summary["rejected"][0]["reason"],
            "target_codebase_blacklisted_origin",
        )


if __name__ == "__main__":
    unittest.main()
