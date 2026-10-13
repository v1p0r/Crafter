import ast
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.codegen_agent import CodegenAgent
from src.memory.global_memory import GlobalMemory
from src.utils.llm_client import call_minimax, call_minimax_with_tools
from src.utils.requirements import (
    REQUIRED_MSG_ALLOWLIST,
    append_requirement,
    ensure_requirements,
    module_to_pip_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("debug_agent")

# ── Tool definitions (Anthropic format) ───────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a source file from the codebase. Returns numbered lines. "
            "Use this to inspect files referenced in errors or to check definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within the codebase, e.g. 'model/unet.py'",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_codebase",
        "description": (
            "Search for a string across all files in the codebase. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find where a function/class is defined or used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for (e.g. class name, function name, variable)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "submit_fix",
        "description": (
            "Submit your fix. Use 'edits' for small targeted line-range changes, "
            "or 'file_rewrites' when you need to rewrite most of a file (e.g. a whole method). "
            "If your fix fails to apply you will get an error back — correct it and call again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_cause": {
                    "type": "string",
                    "description": (
                        "What actually produces the bad state — not the symptom. "
                        "E.g. 'trainer passes a dict but Simformer.__call__ expects kwargs', "
                        "not 'TypeError in __call__'."
                    ),
                },
                "evidence": {
                    "type": "string",
                    "description": (
                        "What you read/verified that confirms the root cause "
                        "(file+line of the definition, the mismatched signature, etc.)"
                    ),
                },
                "analysis": {
                    "type": "string",
                    "description": "Brief explanation of your fix strategy",
                },
                "edits": {
                    "type": "array",
                    "description": "Small targeted line-range edits. Use for changes under ~15 lines.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative file path, e.g. 'model/adaptor.py'",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "First line to replace (1-indexed, inclusive)",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Last line to replace (1-indexed, inclusive)",
                            },
                            "new_code": {
                                "type": "string",
                                "description": "Replacement code (indentation will be auto-corrected)",
                            },
                        },
                        "required": ["path", "start_line", "end_line", "new_code"],
                    },
                },
                "file_rewrites": {
                    "type": "array",
                    "description": (
                        "Complete file rewrites. Use when the fix requires changing most of a file "
                        "or when line edits would be too complex. Provide the COMPLETE file content."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative file path, e.g. 'model/adaptor.py'",
                            },
                            "content": {
                                "type": "string",
                                "description": "Complete file content (all imports, classes, functions)",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
                "files_to_create": {
                    "type": "array",
                    "description": "Paths of missing files that need to be generated (empty if none)",
                    "items": {"type": "string"},
                },
                "notes_for_next_attempt": {
                    "type": "string",
                    "description": (
                        "Optional: durable facts about this codebase worth remembering for later "
                        "debugging attempts (interfaces, expected tensor shapes, gotchas)."
                    ),
                },
            },
            "required": ["root_cause", "analysis"],
        },
    },
]

ANTI_DEGRADATION_RULES = (
    "- NEVER 'fix' an error by deleting functionality, stubbing a computation to a constant, "
    "wrapping the failure in a bare try/except-pass, or commenting out the failing call. "
    "If a component cannot work in this environment, gate it behind a flag/fallback — "
    "keep the real implementation intact\n"
)

PROGRESS_MARKERS = re.compile(r"epoch|step|loss|it/s|iter\b|\d+%", re.IGNORECASE)

IMPORT_LEVEL_ERRORS = ("ModuleNotFoundError", "ImportError", "SyntaxError", "IndentationError")

RESOURCE_ERROR_TRIGGER = re.compile(
    r"FileNotFoundError|No such file|not found|does not exist", re.IGNORECASE)
RESOURCE_ERROR_HINT = re.compile(
    r"dataset|imagenet|data (?:path|dir|directory)|download|checkpoint|pretrained"
    r"|\.ckpt|\.pth|weights|api[_ ]?key|credential", re.IGNORECASE)


class DebugAgent(BaseAgent):
    def __init__(
        self,
        global_memory: GlobalMemory,
        output_dir: str = "data/output",
        codebase_dir: str = "data/output/codebase",
        entry_point: str | None = None,
        run_timeout: int = 120,
        strict_timeout: bool = False,
    ):
        super().__init__(name="debug_agent", global_memory=global_memory, output_dir=output_dir)
        self.codebase_dir = Path(codebase_dir)
        self.output_dir = output_dir
        self.entry_point_override = entry_point
        self.run_timeout = run_timeout
        self.strict_timeout = strict_timeout

        self.journal: list[dict] = []       # one entry per fix/regen attempt
        self.codebase_notes: list[str] = []  # durable facts the model records
        self.pip_notes: list[str] = []       # unresolvable-dependency notes
        self._install_attempted: set[str] = set()
        self._iter_changes: dict[int, dict] = {}  # iteration -> {backed_up, created}
        self._snapshots_dir = Path(output_dir) / "debug_snapshots"

    # ── Main debug loop ────────────────────────────────────────────────

    async def run(self, max_iterations: int = 20) -> dict[str, Any]:
        logger.info("=== Starting debug loop ===")
        self.log_reasoning("start", "Beginning debug loop")

        ensure_requirements(self.codebase_dir)

        if self._detect_entry_point() is None:
            msg = (f"No entry point found in {self.codebase_dir} "
                   f"(looked for architecture entry_point, reproduce.sh, main.py, run.py, "
                   f"train.py, run_experiments.py)")
            logger.error(msg)
            return {"success": False, "iterations": 0, "error": msg}

        try:
            await self._preflight()
        except Exception as e:
            logger.warning(f"Preflight failed (continuing to run-crash loop): {e}")

        best_score: tuple | None = None
        best_iteration = 0
        prev_score: tuple | None = None
        error_streak_sig: str | None = None
        error_streak = 0
        regen_attempted: set[str] = set()
        last_fix_iteration: int | None = None
        result: dict[str, Any] = {}

        for iteration in range(1, max_iterations + 1):
            logger.info(f"--- Debug iteration {iteration} ---")
            self.log_reasoning(f"iteration_{iteration}", "Running code")

            run_result = self._run_code()
            score = self._score_run(run_result)
            logger.info(f"Code run result: success={run_result['success']} score={score}")

            # Close out the previous attempt's journal entry
            error_now = "" if run_result["success"] else run_result.get("error", "")
            self._finalize_last_entry(run_result["success"], error_now)

            # Regression check: did the previous iteration's fix clearly make things worse?
            if (not run_result["success"] and last_fix_iteration is not None
                    and prev_score is not None and self._is_clear_regression(prev_score, score)):
                logger.warning(
                    f"Fix from iteration {last_fix_iteration} caused a clear regression "
                    f"({prev_score} -> {score}) — reverting it"
                )
                self._restore_iteration(last_fix_iteration)
                if self.journal:
                    self.journal[-1]["outcome"] = "reverted"
                self.codebase_notes.append(
                    f"Attempt at iteration {last_fix_iteration} was REVERTED: it made the run "
                    f"fail earlier ({self._phase_name(prev_score[0])} -> {self._phase_name(score[0])})."
                )
                last_fix_iteration = None
                prev_score = None  # state changed; don't compare next run against this one
                self._write_journal()
                continue

            prev_score = score
            last_fix_iteration = None

            if run_result["success"]:
                logger.info(f"SUCCESS after {iteration} iterations")
                self.log_reasoning("done", f"Code runs successfully after {iteration} iterations")
                self._write_journal()
                result = {"success": True, "iterations": iteration}
                for k in ("verified", "silent", "note"):
                    if k in run_result:
                        result[k] = run_result[k]
                return result

            # Track the best (furthest-progressing) state so we never end worse
            if best_score is None or score > best_score:
                best_score = score
                best_iteration = iteration
                self._save_best_state()

            error = run_result["error"]
            logger.error(f"ERROR:\n{error}")
            self.log_reasoning(f"error_{iteration}", error)

            # Auto-install missing packages (no LLM needed)
            self._install_missing_packages(error)

            # Track how long the same (normalized) error has persisted
            sig = self._normalize_error(error)
            if sig == error_streak_sig:
                error_streak += 1
            else:
                error_streak_sig = sig
                error_streak = 1

            crash_file, _ = self._get_crash_location(error)
            resource_error = self._is_resource_error(error)

            try:
                # Same error 3+ runs in a row: escalate to regeneration (unless the
                # error is about a missing external resource — regen can't fix that)
                if (error_streak >= 3 and crash_file and not resource_error
                        and crash_file not in regen_attempted):
                    logger.info(
                        f"Error persisted {error_streak} runs — regenerating {crash_file} from scratch"
                    )
                    self.log_reasoning(f"regen_{crash_file}", "Persistent error, regenerating file")
                    regen_attempted.add(crash_file)

                    files_to_regen = [crash_file]
                    for tf in self._get_all_traceback_files(error):
                        if tf != crash_file and tf not in regen_attempted:
                            current = (self.global_memory.get("generated_code", {}).get(crash_file, "")
                                       or self._read_file_raw(crash_file))
                            dep_module = tf.replace("/", ".").replace(".py", "")
                            if current and dep_module in current:
                                files_to_regen.append(tf)
                                regen_attempted.add(tf)

                    for fp in reversed(files_to_regen):
                        await self._regenerate_file(fp, error, iteration)

                    self.journal.append({
                        "iteration": iteration,
                        "error_before": error[-300:],
                        "root_cause": "(escalation)",
                        "fix": f"Regenerated from scratch: {', '.join(files_to_regen)}",
                        "files": files_to_regen,
                        "outcome": "pending",
                        "applied": True,
                    })
                    last_fix_iteration = iteration
                    error_streak = 0
                    error_streak_sig = None
                    self._write_journal()
                    continue

                # Interactive investigate + fix
                fix_result = await self._investigate_and_fix(error, iteration)
            except Exception as e:
                logger.error(f"Iteration {iteration} failed with harness error: {e}")
                self.journal.append({
                    "iteration": iteration,
                    "error_before": error[-300:],
                    "root_cause": "(harness error)",
                    "fix": f"Investigation crashed: {str(e)[:200]}",
                    "files": [],
                    "outcome": "no-fix",
                    "applied": False,
                })
                self._write_journal()
                continue

            if not fix_result:
                logger.warning("No fix produced — retrying next iteration")
                self.journal.append({
                    "iteration": iteration,
                    "error_before": error[-300:],
                    "root_cause": "",
                    "fix": "Investigation ended without a fix",
                    "files": [],
                    "outcome": "no-fix",
                    "applied": False,
                })
                self._write_journal()
                continue

            applied = fix_result.pop("_applied", False)
            files_changed = [e["path"] for e in fix_result.get("edits", []) if isinstance(e, dict)]
            files_changed += [r["path"] for r in fix_result.get("file_rewrites", [])
                              if isinstance(r, dict)]
            note = (fix_result.get("notes_for_next_attempt") or "").strip()
            if note:
                self.codebase_notes.append(note[:400])

            self.journal.append({
                "iteration": iteration,
                "error_before": error[-300:],
                "root_cause": (fix_result.get("root_cause") or "")[:300],
                "fix": (fix_result.get("analysis") or "")[:300],
                "files": files_changed,
                "outcome": "pending" if applied else "rejected",
                "applied": applied,
            })
            if applied:
                last_fix_iteration = iteration

            # Create missing files (needs pipeline memory — skipped in bare mode)
            for fp in fix_result.get("files_to_create", []):
                if not isinstance(fp, str):
                    continue
                if not self.global_memory.get("module_specs") and not self.global_memory.get("architecture"):
                    logger.info(f"Skipping files_to_create ({fp}): no pipeline memory (bare codebase mode)")
                    continue
                fp = self._normalize_path(fp)
                if fp and self._is_codebase_path(fp):
                    logger.info(f"Creating missing file: {fp}")
                    agent = CodegenAgent(
                        file_path=fp,
                        global_memory=self.global_memory,
                        output_dir=self.output_dir,
                        codebase_dir=str(self.codebase_dir),
                        is_orchestrator=self._is_orchestrator_file(fp),
                    )
                    await agent.run()

            self._write_journal()

        logger.error(f"Max iterations ({max_iterations}) reached")
        result = {"success": False, "iterations": max_iterations,
                  "error": f"Max iterations ({max_iterations}) reached"}

        # End-of-run guarantee: never leave the codebase in a worse state than the
        # best one we saw. Skip the restore if the final iteration applied a fix we
        # never got to score — reverting an untested fix could discard the solution.
        if (best_score is not None and prev_score is not None
                and last_fix_iteration is None and best_score > prev_score):
            logger.warning(
                f"Final state (score {prev_score}) is worse than iteration {best_iteration} "
                f"(score {best_score}) — restoring best state"
            )
            self._restore_best_state()
            result["best_state_restored"] = best_iteration
        self._write_journal()
        return result

    # ── Preflight: static checks before the run-crash loop ─────────────

    async def _preflight(self) -> None:
        """Catch all syntax errors and broken local imports in one batched pass
        instead of discovering them one crash at a time."""
        for round_no in (1, 2):
            issues = self._collect_static_issues()
            if not issues:
                if round_no == 1:
                    logger.info("Preflight: no static issues found")
                return
            logger.info(f"Preflight round {round_no}: {len(issues)} static issue(s)")
            report = "## Static check failures (fix ALL of them in one submit_fix call)\n"
            for fp, problem in issues[:20]:
                report += f"- {fp}: {problem}\n"
            shown = set()
            for fp, _ in issues[:8]:
                if fp in shown:
                    continue
                shown.add(fp)
                code = self._read_file(fp)
                if code:
                    report += f"\n## File: {fp}\n{code}\n"
            fix = await self._investigate_and_fix(report, iteration=0, stage_label="preflight")
            if fix:
                self.journal.append({
                    "iteration": 0,
                    "error_before": f"preflight: {len(issues)} static issue(s)",
                    "root_cause": (fix.get("root_cause") or "")[:300],
                    "fix": (fix.get("analysis") or "")[:300],
                    "files": [e.get("path") for e in fix.get("edits", []) if isinstance(e, dict)]
                             + [r.get("path") for r in fix.get("file_rewrites", []) if isinstance(r, dict)],
                    "outcome": "preflight",
                    "applied": fix.pop("_applied", False),
                })
            else:
                return  # nothing fixable this round; don't loop again

    def _collect_static_issues(self) -> list[tuple[str, str]]:
        """Syntax errors + local `from x import y` where y doesn't exist in x.
        Pure static analysis — nothing is executed."""
        sources: dict[str, str] = {}
        for py_file in self.codebase_dir.rglob("*.py"):
            rel = str(py_file.relative_to(self.codebase_dir))
            try:
                sources[rel] = py_file.read_text()
            except Exception:
                continue

        issues: list[tuple[str, str]] = []
        trees: dict[str, ast.AST] = {}
        for rel, text in sources.items():
            try:
                trees[rel] = ast.parse(text)
            except SyntaxError as e:
                issues.append((rel, f"SyntaxError line {e.lineno}: {e.msg} "
                                    f"({(e.text or '').strip()[:80]})"))

        exported: dict[str, set[str]] = {
            rel: self._public_names_from_tree(tree) for rel, tree in trees.items()
        }
        for rel, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
                    continue
                target = node.module.replace(".", "/") + ".py"
                if target not in sources:
                    continue  # not a direct local module (package or third-party)
                if target not in exported:
                    continue  # target has a syntax error — already reported
                for alias in node.names:
                    if alias.name != "*" and alias.name not in exported[target]:
                        issues.append((rel, f"`from {node.module} import {alias.name}` — "
                                            f"'{alias.name}' is not defined in {target}"))
        return issues

    def _public_names_from_tree(self, tree: ast.AST) -> set[str]:
        """Top-level names a module exports (defs, classes, assignments, imports)."""
        names: set[str] = set()
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        names.update(e.id for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.If, ast.Try)):
                # names defined under `if TYPE_CHECKING:`/try-import blocks
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for alias in sub.names:
                            if alias.name != "*":
                                names.add(alias.asname or alias.name.split(".")[0])
                    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(sub.name)
                    elif isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                names.add(t.id)
        return names

    # ── Interactive tool-use investigation ─────────────────────────────

    async def _investigate_and_fix(
        self, error: str, iteration: int, stage_label: str = "debug"
    ) -> dict | None:
        """Give the LLM the crash context + tools, let it investigate, apply the
        fix in-session (with feedback on failures), and return the fix dict.

        Returned dict carries `_applied` (bool). Returns None if no fix emerged.
        """
        crash_file, _ = self._get_crash_location(error) if iteration > 0 else (None, None)
        traceback_files = self._get_all_traceback_files(error) if iteration > 0 else []

        initial_context = f"## Error traceback\n{error}\n" if iteration > 0 else error + "\n"

        shown_files = set()
        for tf in traceback_files:
            code = self._read_file(tf)
            if code and tf not in shown_files:
                label = "Crash file" if tf == crash_file else "Traceback file"
                initial_context += f"\n## {label}: {tf}\n{code}\n"
                shown_files.add(tf)

        if crash_file and crash_file not in shown_files:
            crash_code = self._read_file(crash_file)
            if crash_code:
                initial_context += f"\n## Crash file: {crash_file}\n{crash_code}\n"
            else:
                initial_context += f"\n(Crash file {crash_file} not found or empty)\n"

        # Error with no codebase traceback (e.g. a printed message + exit):
        # show the entry point so the model has something to work from
        if iteration > 0 and not shown_files and not crash_file:
            ep = self._detect_entry_point()
            ep_code = self._read_file(ep) if ep else ""
            if ep_code:
                initial_context += f"\n## Entry point: {ep}\n{ep_code}\n"

        # What is this module SUPPOSED to do (when pipeline memory is available)
        if crash_file:
            spec = self._module_spec_for(crash_file)
            if spec:
                initial_context += f"\n## Intended behavior of {crash_file} (from spec)\n{spec}\n"

        arch = self.global_memory.get("architecture", {})
        if isinstance(arch, dict) and arch.get("files"):
            initial_context += "\n## Codebase structure\n"
            for f in arch["files"]:
                if isinstance(f, dict):
                    initial_context += f"- {f.get('path')}: {f.get('purpose', '')}\n"

        journal_ctx = self._journal_context()
        if journal_ctx:
            initial_context += journal_ctx

        if self.pip_notes:
            initial_context += "\n## Unresolvable dependencies\n"
            for n in self.pip_notes[-5:]:
                initial_context += f"- {n}\n"

        if iteration > 0 and self._is_resource_error(error):
            initial_context += (
                "\n## Environment note: missing external resource\n"
                "The error indicates a missing external resource (dataset, checkpoint, or similar). "
                "This environment does NOT have it and cannot download large datasets. "
                "Do NOT keep fixing paths or loader logic — modify the code so a smoke run can "
                "proceed without the resource: add a small synthetic/dummy-data fallback with "
                "correct shapes (keep the real-data path behind a flag), or skip the component.\n"
            )

        rewrite_guidance = ""
        recent = [j for j in self.journal if j.get("iteration", 0) > 0][-2:]
        if len(recent) == 2 and all(j.get("applied") and j.get("outcome") == "unchanged"
                                    for j in recent):
            rewrite_guidance = (
                "\n## CRITICAL: Your previous fixes were applied but the error did NOT change.\n"
                "You MUST use `file_rewrites` to rewrite the entire file with the bug properly "
                "fixed. Do NOT use `edits` — small edits have not worked. Read the file "
                "carefully, understand the full control flow, and rewrite it correctly.\n"
            )
        elif self.journal:
            rewrite_guidance = (
                "\n## Note: If your fix requires changing more than ~15 lines or restructuring "
                "a method, use `file_rewrites` in submit_fix instead of `edits`.\n"
            )

        system = (
            "You are an expert Python/ML debugger. You have been given an error and the "
            "codebase files involved. Find the ROOT CAUSE and fix it.\n\n"
            "## How to work\n"
            "1. Read the FULL traceback — the bug may not be in the last file. Trace the call "
            "chain to the ROOT CAUSE: the place that produces the bad state, not the place "
            "that crashes on it\n"
            "2. If the root cause is not provably visible in the context above, you MUST verify "
            "your hypothesis first with read_file or search_codebase (check the caller, the "
            "callee's signature, the shape at the producer)\n"
            "3. Then call submit_fix with `root_cause`, `evidence`, and your fix\n"
            "4. If the fix fails to apply you will get an error back — correct it and resubmit\n\n"
            "## submit_fix options\n"
            "- `edits`: small targeted line-range changes (under ~15 lines). Line numbers refer "
            "to the ORIGINAL file\n"
            "- `file_rewrites`: complete file content — for structural bugs or method rewrites\n"
            "- `edits` and `file_rewrites` MUST be JSON arrays of objects, never strings\n\n"
            "## Rules\n"
            "- Fix the file where the bug ORIGINATES, not just the file that crashed\n"
            + ANTI_DEGRADATION_RULES +
            "- You have at most 5 investigation calls (read_file/search_codebase) — use them "
            "wisely\n"
            "- Only fix codebase files — NEVER try to fix torch, numpy, or other packages\n"
            "- For shape mismatches: check spatial dimensions at each step\n"
            "- For shape mismatches with nn.Linear on [B,C,H,W] tensors: apply channel-wise "
            "(permute to [B,H,W,C], reshape to [B*H*W,C], apply Linear, reshape back). "
            "NEVER flatten to [B,C*H*W]\n"
            "- If a previous applied fix didn't change the error, try a fundamentally "
            "different approach\n"
            "- Use `notes_for_next_attempt` to record durable facts (interfaces, shapes, "
            "gotchas) that later debugging attempts should know\n"
            + rewrite_guidance
        )

        messages: list[dict] = [{"role": "user", "content": initial_context}]
        self.log_message("user", initial_context)

        max_tool_calls = 5     # investigation budget (read/search only)
        max_fix_retries = 2    # resubmits after a fix fails to apply
        max_rounds = 12        # hard cap on API round-trips
        investigate_calls = 0
        fix_retries = 0
        truncation_retries = 0
        nudges = 0

        for _ in range(max_rounds):
            text, tool_calls, stop_reason = await call_minimax_with_tools(
                messages=messages,
                tools=TOOLS,
                system=system,
                max_tokens=16384,
                stage=stage_label,
                cache=True,
            )

            logger.info(f"LLM response: text={len(text)} chars, "
                        f"tool_calls={len(tool_calls)}, stop={stop_reason}")
            if text:
                self.log_message("assistant", text)

            if not tool_calls:
                if stop_reason == "max_tokens":
                    # Model burned the whole budget (usually on thinking) without
                    # producing a tool call — retry instead of aborting the iteration
                    truncation_retries += 1
                    if truncation_retries > 2:
                        logger.warning("Repeated max_tokens truncation — giving up this iteration")
                        break
                    logger.warning("Response truncated before any tool call — retrying")
                    continue
                if stop_reason == "end_turn":
                    if nudges >= 2:
                        logger.warning("Model keeps ending turn without tools — giving up")
                        break
                    nudges += 1
                    messages.append({"role": "assistant", "content": text or "(no response)"})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must use tools to investigate and fix the bug. "
                            "Use read_file to inspect a file, search_codebase to find symbols, "
                            "or submit_fix when you're ready to fix. Do not just explain — act."
                        ),
                    })
                    continue
                logger.warning(f"Model stopped ({stop_reason}) without calling submit_fix")
                break

            assistant_content: list[dict] = []
            if text:
                assistant_content.append({"type": "text", "text": text})
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict] = []
            fix_result: dict | None = None

            for tc in tool_calls:
                name = tc["name"]
                inp = tc["input"]
                # MiniMax's anthropic-compat shim sometimes delivers input as a
                # JSON string instead of an object — decode rather than drop it
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except (json.JSONDecodeError, ValueError):
                        inp = {}
                if not isinstance(inp, dict):
                    inp = {}
                logger.info(f"Tool call: {name}({json.dumps(inp, default=str)[:200]})")

                if name == "read_file":
                    investigate_calls += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": self._read_file(inp.get("path", "")) or "(file not found or empty)",
                    })

                elif name == "search_codebase":
                    investigate_calls += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": self._search_code(inp.get("query", "")),
                    })

                elif name == "submit_fix":
                    if fix_result is not None:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": "A fix was already received in this turn.",
                        })
                        continue
                    coercion_problems = self._coerce_fix_args(inp)
                    if coercion_problems:
                        applied, problems = False, coercion_problems
                    else:
                        applied, problems = self._apply_fix(inp, iteration)
                    logger.info(f"Fix submitted (applied={applied}): "
                                f"{(inp.get('root_cause') or inp.get('analysis') or '')[:200]}")
                    self.log_message("assistant", f"FIX: {json.dumps(inp, default=str)[:2000]}")

                    if applied:
                        fix_result = inp
                        fix_result["_applied"] = True
                        msg = "Fix applied."
                        if problems:
                            msg += " Some parts were skipped:\n- " + "\n- ".join(problems)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": msg,
                        })
                    elif fix_retries < max_fix_retries:
                        fix_retries += 1
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "is_error": True,
                            "content": (
                                "Your fix was NOT applied:\n- " + "\n- ".join(problems) +
                                "\nCorrect these problems and call submit_fix again. If line "
                                "edits keep failing, use `file_rewrites` with the COMPLETE "
                                "corrected file content."
                            ),
                        })
                        logger.warning(f"Fix rejected ({fix_retries}/{max_fix_retries}): {problems}")
                    else:
                        fix_result = inp
                        fix_result["_applied"] = False
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": "Fix received but could not be applied.",
                        })

                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": f"Unknown tool: {name}",
                    })

            content: list[dict] = tool_results
            if fix_result is None and investigate_calls >= max_tool_calls:
                content = tool_results + [{
                    "type": "text",
                    "text": "You have used your investigation budget. "
                            "Call submit_fix NOW with your best fix.",
                }]
            messages.append({"role": "user", "content": content})

            if fix_result is not None:
                return fix_result

        logger.warning("Investigation ended without a fix")
        return None

    def _coerce_fix_args(self, inp: dict) -> list[str]:
        """Normalize submit_fix args in place. Models sometimes emit list fields as
        JSON-encoded strings or leaked XML — iterating a string char-by-char must
        never happen. Returns a list of problems (empty = OK)."""
        problems: list[str] = []
        for field in ("edits", "file_rewrites", "files_to_create"):
            value = inp.get(field)
            if value is None or isinstance(value, list):
                continue
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, list):
                    inp[field] = parsed
                    continue
                inp[field] = []
                problems.append(
                    f"`{field}` must be a JSON array of objects, but you sent a string "
                    f"starting {value[:60]!r}. Resubmit with a proper array."
                )
            else:
                inp[field] = []
                problems.append(
                    f"`{field}` must be a JSON array, got {type(value).__name__}."
                )
        return problems

    # ── Regenerate file from scratch ────────────────────────────────────

    async def _regenerate_file(
        self, crash_file: str, error: str, iteration: int,
        extra_guidance: str = "",
    ) -> None:
        """Regenerate a file from scratch when the same error persists.

        Uses full available context: module spec, error traceback, dependent
        files, debug journal. Smoke-tests the result and checks that no public
        names were dropped; retries once with that feedback."""
        logger.info(f"Regenerating {crash_file} from scratch")

        module_spec_text = self._module_spec_for(crash_file, max_chars=100_000)
        generated_code = self.global_memory.get("generated_code", {})

        current_code = generated_code.get(crash_file, "") or self._read_file_raw(crash_file)

        dependent_code = ""
        if current_code:
            for line in current_code.splitlines():
                line = line.strip()
                if line.startswith(("from ", "import ")):
                    parts = line.replace("from ", "").replace("import ", "").split()[0]
                    dep_path = parts.replace(".", "/") + ".py"
                    dep_code = generated_code.get(dep_path, "") or self._read_file_raw(dep_path)
                    if dep_code:
                        dependent_code += f"\n--- {dep_path} ---\n{dep_code}\n"

        system = (
            "You are an expert Python/ML engineer. A generated file has a structural bug that "
            "could not be fixed with small edits. You must REWRITE THE ENTIRE FILE from scratch.\n\n"
            "## Rules\n"
            "- Write COMPLETE, WORKING code — no TODOs, no placeholders\n"
            "- Keep EVERY public function and class the current file exposes — other files "
            "import them; dropping one breaks the codebase\n"
            + ANTI_DEGRADATION_RULES +
            "- The file must be compatible with the dependent files shown below\n"
            "- Match all function signatures and class interfaces exactly\n"
            "- Fix the bug described in the error traceback\n"
            "- Include all imports\n"
            "- Use type hints\n"
            "- For tensor shape mismatches: trace shapes step by step through the forward pass\n"
            "- For import errors: check the dependent files for correct class/function names\n"
            "\n## Output Format\n"
            f"Write only the file: {crash_file}\n"
            "```python\n"
            "...\n"
            "```"
        )

        context = f"## File to rewrite: {crash_file}\n\n"
        if module_spec_text:
            context += f"## Module spec:\n{module_spec_text}\n\n"
        context += f"## Error traceback (the bug to fix):\n{error}\n\n"
        if extra_guidance:
            context += f"## Additional guidance:\n{extra_guidance}\n\n"

        journal_ctx = self._journal_context()
        if journal_ctx:
            context += journal_ctx + "\n"

        if current_code:
            context += f"## Current (BUGGY) code:\n```python\n{current_code}\n```\n\n"

        architecture_md = self.global_memory.get("architecture_md", "")
        if architecture_md:
            context += f"## Architecture:\n{architecture_md}\n\n"

        if dependent_code:
            context += f"## Dependent files (match these interfaces):\n{dependent_code}\n\n"

        context += (
            "REWRITE the entire file from scratch, fixing the bug. "
            "Make sure all function signatures match what other files expect."
        )

        messages = [{"role": "user", "content": context}]
        self.log_message("user", f"[REGEN] {context[:2000]}...")

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="debug"
        )
        self.log_message("assistant", f"[REGEN] {response[:2000]}...")

        code = self._clean_code(response)

        if not code or not self._is_valid_python(code):
            logger.warning(f"Regenerated code for {crash_file} is not valid Python")
            return

        # Interface guard: a rewrite that silently drops public names degrades the codebase
        dropped = sorted(self._public_names(current_code) - self._public_names(code))

        self._snapshot_file(iteration, crash_file)
        out_path = self.codebase_dir / crash_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code)
        generated_code[crash_file] = code
        self.global_memory.set("generated_code", generated_code)
        logger.info(f"Regenerated {crash_file}: {len(code)} chars")

        smoke_error = self._smoke_test(crash_file)
        feedback = ""
        if smoke_error:
            logger.warning(f"Smoke test failed for {crash_file}: {smoke_error[:200]}")
            feedback += (f"The previous regeneration FAILED the smoke test with:\n"
                         f"{smoke_error}\nFix this error in the rewrite.\n")
        if dropped:
            logger.warning(f"Regen of {crash_file} dropped public names: {dropped}")
            feedback += (f"Your rewrite DROPPED these public names that other files may import: "
                         f"{', '.join(dropped)}. Keep every public function/class.\n")

        if feedback and not extra_guidance:  # only retry once
            await self._regenerate_file(crash_file, error, iteration, extra_guidance=feedback)
        elif not smoke_error:
            logger.info(f"Smoke test passed for {crash_file}")

        self.log_reasoning(f"regen_done_{crash_file}", f"Regenerated {len(code)} chars")

    def _public_names(self, code: str) -> set[str]:
        if not code:
            return set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {node.name for node in getattr(tree, "body", [])
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and not node.name.startswith("_")}

    def _module_spec_for(self, file_path: str, max_chars: int = 1500) -> str:
        module_specs = self.global_memory.get("module_specs", {})
        if not isinstance(module_specs, dict):
            return ""
        for spec in module_specs.values():
            if isinstance(spec, dict) and spec.get("spec_text") and file_path in spec["spec_text"]:
                return spec["spec_text"][:max_chars]
        return ""

    def _smoke_test(self, file_path: str) -> str | None:
        """Run a quick import test on a regenerated file. Returns error string or None."""
        if not file_path.endswith(".py"):
            return None
        module = file_path.replace("/", ".").replace(".py", "")
        test_code = f"import sys; sys.path.insert(0, '.'); import {module}; print('OK')"
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.codebase_dir),
            )
            if result.returncode == 0:
                return None
            return result.stderr or result.stdout or f"Exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            return str(e)

    # ── Apply fix ──────────────────────────────────────────────────────

    def _apply_fix(self, fix_result: dict, iteration: int) -> tuple[bool, list[str]]:
        """Apply edits and/or file rewrites from a submit_fix call.

        Snapshots every file before changing it. Returns (any_applied, problems)
        where problems explain exactly why parts (or all) of the fix were skipped —
        they are fed back to the model so it can resubmit."""
        edits = fix_result.get("edits") or []
        file_rewrites = fix_result.get("file_rewrites") or []
        problems: list[str] = []

        if not edits and not file_rewrites:
            return False, ["No `edits` or `file_rewrites` provided — submit_fix must "
                           "include at least one."]

        generated_code = self.global_memory.get("generated_code", {})
        any_applied = False

        for rewrite in file_rewrites:
            if not isinstance(rewrite, dict):
                problems.append(f"file_rewrites item is not an object: {str(rewrite)[:80]!r}")
                continue
            fp = self._normalize_path(rewrite.get("path", ""))
            if not fp or not self._is_codebase_path(fp):
                problems.append(f"Rewrite path outside the codebase: {rewrite.get('path')!r}")
                continue
            cleaned = self._clean_code(rewrite.get("content", ""))
            syntax = self._syntax_error(cleaned, fp)
            if syntax:
                problems.append(f"Rewrite of {fp} is not valid Python: {syntax}")
                continue
            self._snapshot_file(iteration, fp)
            out_path = self.codebase_dir / fp
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned)
            generated_code[fp] = cleaned
            self.global_memory.set("generated_code", generated_code)
            logger.info(f"Full rewrite: {fp} ({len(cleaned)} chars)")
            self.log_reasoning(f"rewrite_{fp}", f"Full rewrite ({len(cleaned)} chars)")
            any_applied = True

        edits_by_file: dict[str, list[dict]] = {}
        for edit in edits:
            if not isinstance(edit, dict):
                problems.append(f"edits item is not an object: {str(edit)[:80]!r}")
                continue
            path = self._normalize_path(edit.get("path", ""))
            if not path or not self._is_codebase_path(path):
                problems.append(f"Edit path outside the codebase: {edit.get('path')!r}")
                continue
            if not edit.get("start_line") or not edit.get("end_line"):
                problems.append(f"Edit for {path} is missing start_line/end_line")
                continue
            edits_by_file.setdefault(path, []).append(edit)

        for fp, file_edits in edits_by_file.items():
            current_code = generated_code.get(fp, "") or self._read_file_raw(fp)
            if not current_code:
                problems.append(f"Cannot edit {fp} — file not found")
                continue

            total_lines = len(current_code.splitlines())
            edit_tuples = [(e["start_line"], e["end_line"], e["new_code"]) for e in file_edits]
            new_code = self._apply_line_edits(current_code, edit_tuples)

            if new_code and new_code != current_code:
                syntax = self._syntax_error(new_code, fp)
                if syntax:
                    problems.append(f"Edits to {fp} produce invalid Python: {syntax}. "
                                    f"Line numbers refer to the ORIGINAL file "
                                    f"({total_lines} lines).")
                    logger.warning(f"Edits to {fp} produced invalid Python — skipping")
                    continue
                self._snapshot_file(iteration, fp)
                out_path = self.codebase_dir / fp
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(new_code)
                generated_code[fp] = new_code
                self.global_memory.set("generated_code", generated_code)
                logger.info(f"Applied {len(edit_tuples)} edit(s) to {fp}")
                self.log_reasoning(f"edited_{fp}", f"Applied {len(edit_tuples)} edit(s)")
                any_applied = True
            else:
                problems.append(f"Edits to {fp} changed nothing — line ranges may be out of "
                                f"bounds (file has {total_lines} lines)")

        return any_applied, problems

    def _syntax_error(self, code: str, name: str) -> str | None:
        """Return a human-readable syntax problem, or None if the code compiles."""
        if not code or not code.strip():
            return "empty content"
        stripped = code.strip()
        if stripped.startswith("<") or "<invoke" in stripped[:200]:
            return "content looks like markup, not Python"
        try:
            compile(code, name, "exec")
            return None
        except SyntaxError as e:
            return f"line {e.lineno}: {e.msg} ({(e.text or '').strip()[:80]})"

    # ── Snapshots & regression protection ──────────────────────────────

    def _snapshot_file(self, iteration: int, rel_path: str) -> None:
        """Back up a file before modifying it (once per iteration per file)."""
        rec = self._iter_changes.setdefault(iteration, {"backed_up": [], "created": []})
        if rel_path in rec["backed_up"] or rel_path in rec["created"]:
            return
        src = self.codebase_dir / rel_path
        if src.exists():
            dst = self._snapshots_dir / f"iter_{iteration}" / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rec["backed_up"].append(rel_path)
        else:
            rec["created"].append(rel_path)

    def _restore_iteration(self, iteration: int) -> None:
        """Undo every file change made in an iteration."""
        rec = self._iter_changes.get(iteration)
        if not rec:
            return
        generated_code = self.global_memory.get("generated_code", {})
        for rel_path in rec["backed_up"]:
            src = self._snapshots_dir / f"iter_{iteration}" / rel_path
            if src.exists():
                dst = self.codebase_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                generated_code[rel_path] = src.read_text()
                logger.info(f"Restored {rel_path} from iteration {iteration} snapshot")
        for rel_path in rec["created"]:
            target = self.codebase_dir / rel_path
            if target.exists():
                target.unlink()
                generated_code.pop(rel_path, None)
                logger.info(f"Removed {rel_path} (created by reverted iteration {iteration})")
        self.global_memory.set("generated_code", generated_code)

    def _save_best_state(self) -> None:
        """Keep a full copy of the best-progressing codebase state seen so far."""
        best_dir = self._snapshots_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        for src in self.codebase_dir.rglob("*"):
            if src.is_dir() or "__pycache__" in src.parts:
                continue
            if src.suffix not in (".py", ".txt", ".sh", ".yaml", ".yml", ".json"):
                continue
            rel = src.relative_to(self.codebase_dir)
            dst = best_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def _restore_best_state(self) -> None:
        best_dir = self._snapshots_dir / "best"
        if not best_dir.exists():
            return
        generated_code = self.global_memory.get("generated_code", {})
        for src in best_dir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(best_dir)
            dst = self.codebase_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if src.suffix == ".py":
                generated_code[str(rel)] = src.read_text()
        self.global_memory.set("generated_code", generated_code)
        logger.info("Restored best-known codebase state")

    def _score_run(self, run_result: dict) -> tuple[int, int]:
        """(phase, stdout_bytes): higher = further progress. Phase 2 = success,
        1 = runtime failure, 0 = import/syntax-level failure."""
        if run_result.get("success"):
            return (2, len(run_result.get("stdout") or ""))
        error = run_result.get("error", "")
        phase = 1
        for line in reversed(error.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(name) for name in IMPORT_LEVEL_ERRORS):
                phase = 0
            break
        return (phase, len(run_result.get("stdout") or ""))

    def _is_clear_regression(self, prev: tuple, cur: tuple) -> bool:
        """Only unambiguous signals: runtime failure became import-level failure,
        or output disappeared entirely at the same phase. A merely *different*
        error is normal progress, never a regression."""
        if cur[0] >= 2:
            return False
        if prev[0] > cur[0]:
            return True
        if prev[0] == cur[0] and prev[1] > 0 and cur[1] == 0:
            return True
        return False

    def _phase_name(self, phase: int) -> str:
        return {0: "import-time failure", 1: "runtime failure", 2: "success"}.get(phase, "?")

    # ── Debug journal ──────────────────────────────────────────────────

    def _finalize_last_entry(self, success: bool, error_now: str) -> None:
        """Set the outcome of the previous attempt based on what happened next."""
        if not self.journal or self.journal[-1].get("outcome") != "pending":
            return
        entry = self.journal[-1]
        entry["error_after"] = "" if success else error_now[-300:]
        if success:
            entry["outcome"] = "fixed"
        elif self._normalize_error(error_now) == self._normalize_error(entry.get("error_before", "")):
            entry["outcome"] = "unchanged"
        else:
            entry["outcome"] = "error-changed"

    def _journal_context(self) -> str:
        """Render the journal for prompts: recent attempts + all durable notes."""
        if not self.journal and not self.codebase_notes:
            return ""
        out = "\n## Debug journal (what has been tried so far)\n"
        for j in self.journal[-15:]:
            outcome = j.get("outcome", "?")
            status = {
                "fixed": "FIXED",
                "unchanged": "applied, error UNCHANGED — do not repeat this approach",
                "error-changed": "applied, error changed (progress)",
                "reverted": "REVERTED — it made the run fail earlier",
                "rejected": "NOT applied (fix could not be applied)",
                "no-fix": "no fix produced",
                "pending": "pending",
                "preflight": "preflight batch fix",
            }.get(outcome, outcome)
            files = ", ".join(j.get("files", [])) or "-"
            out += (f"- [iter {j.get('iteration')}] {status} | files: {files}\n"
                    f"    root cause: {j.get('root_cause', '')[:150]}\n"
                    f"    fix: {j.get('fix', '')[:150]}\n")
            if j.get("error_after"):
                out += f"    error after: ...{j['error_after'][-120:]}\n"
        if self.codebase_notes:
            out += "\n## Codebase notes (accumulated knowledge — trust these)\n"
            for n in self.codebase_notes[-20:]:
                out += f"- {n}\n"
        return out

    def _write_journal(self) -> None:
        try:
            path = Path(self.output_dir) / "debug_journal.md"
            lines = ["# Debug journal\n"]
            for j in self.journal:
                lines.append(f"## Iteration {j.get('iteration')} — {j.get('outcome')}")
                lines.append(f"- files: {', '.join(j.get('files', [])) or '-'}")
                if j.get("root_cause"):
                    lines.append(f"- root cause: {j['root_cause']}")
                if j.get("fix"):
                    lines.append(f"- fix: {j['fix']}")
                if j.get("error_before"):
                    lines.append(f"- error before: ...{j['error_before'][-200:]}")
                if j.get("error_after"):
                    lines.append(f"- error after: ...{j['error_after'][-200:]}")
                lines.append("")
            if self.codebase_notes:
                lines.append("## Codebase notes")
                lines.extend(f"- {n}" for n in self.codebase_notes)
            path.write_text("\n".join(lines) + "\n")
        except Exception as e:
            logger.warning(f"Could not write debug journal: {e}")

    # ── Tool implementations ──────────────────────────────────────────

    def _read_file(self, path: str) -> str:
        """Read a file and return numbered lines."""
        fp = self._normalize_path(path)
        if not fp:
            return ""
        generated_code = self.global_memory.get("generated_code", {})
        code = generated_code.get(fp, "") or self._read_file_raw(fp)
        if not code:
            return ""
        return self._number_lines(code)

    def _search_code(self, query: str) -> str:
        """Search for a term across memory AND disk (memory wins on duplicates)."""
        if not query:
            return "No query provided."
        generated_code = self.global_memory.get("generated_code", {})
        matches: list[str] = []
        searched: set[str] = set()
        for fp, code in generated_code.items():
            searched.add(fp)
            for lineno, line in enumerate(code.splitlines(), 1):
                if query in line:
                    matches.append(f"{fp}:{lineno}: {line.strip()}")
        for py_file in self.codebase_dir.rglob("*.py"):
            try:
                rel = str(py_file.relative_to(self.codebase_dir))
                if rel in searched:
                    continue
                for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                    if query in line:
                        matches.append(f"{rel}:{lineno}: {line.strip()}")
            except Exception:
                pass
        return "\n".join(matches[:30]) if matches else "No matches found."

    def _get_crash_location(self, error: str) -> tuple[str | None, int | None]:
        """Extract the primary crash file and line from a traceback.

        Returns the LAST codebase file in the traceback (closest to the error),
        or (None, None) when the error has no codebase traceback — callers must
        not attribute such errors to any file."""
        locations = []
        for line in error.split("\n"):
            m = re.search(r'File "([^"]+)", line (\d+)', line)
            if not m:
                continue
            path, lineno = m.group(1), int(m.group(2))
            if Path(path).is_absolute():
                try:
                    rel = str(Path(path).relative_to(self.codebase_dir.resolve()))
                    locations.append((rel, lineno))
                except ValueError:
                    pass
        if locations:
            return locations[-1]
        return (None, None)

    def _get_all_traceback_files(self, error: str) -> list[str]:
        """Extract ALL codebase files from a traceback (unique, in order)."""
        seen = set()
        result = []
        for line in error.split("\n"):
            m = re.search(r'File "([^"]+)", line (\d+)', line)
            if not m:
                continue
            path = m.group(1)
            if Path(path).is_absolute():
                try:
                    rel = str(Path(path).relative_to(self.codebase_dir.resolve()))
                    if rel not in seen:
                        seen.add(rel)
                        result.append(rel)
                except ValueError:
                    pass
        return result

    def _read_file_raw(self, path: str) -> str:
        """Read a file from disk without numbering."""
        fp = self._normalize_path(path)
        if not fp:
            return ""
        disk_path = self.codebase_dir / fp
        if disk_path.exists():
            try:
                return disk_path.read_text()
            except Exception:
                return ""
        return ""

    # ── Error classification ──────────────────────────────────────────

    def _normalize_error(self, error: str) -> str:
        """Error signature stable across shifting line numbers and addresses."""
        if not error:
            return ""
        s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", error)
        s = re.sub(r"\d+", "#", s)
        return s[-240:]

    def _is_resource_error(self, error: str) -> bool:
        """Missing dataset/checkpoint/credential — unfixable by editing paths."""
        if not RESOURCE_ERROR_TRIGGER.search(error):
            return False
        return bool(RESOURCE_ERROR_HINT.search(error))

    # ── Helpers ────────────────────────────────────────────────────────

    def _number_lines(self, code: str) -> str:
        lines = code.splitlines()
        width = len(str(len(lines)))
        return "\n".join(f"{i+1:>{width}} | {line}" for i, line in enumerate(lines))

    def _apply_line_edits(self, code: str, edits: list[tuple[int, int, str]]) -> str | None:
        """Apply line-range edits in reverse order so line numbers stay valid.

        First tries the raw new_code. If the result doesn't compile, falls back
        to auto-dedent + re-indent to match the original indentation.
        """
        lines = code.splitlines(keepends=True)
        total = len(lines)

        for start, end, new_code in sorted(edits, key=lambda e: e[0], reverse=True):
            if start < 1 or end > total or start > end:
                logger.warning(f"Edit out of range: {start}-{end} (file has {total} lines)")
                continue

            if not new_code.strip():
                lines[start - 1:end] = []
                logger.info(f"Deleted lines {start}-{end}")
                continue

            # Try 1: use the raw new_code as-is
            raw_lines = self._code_to_lines(new_code)
            candidate = lines[:start - 1] + raw_lines + lines[end:]
            candidate_str = "".join(candidate)

            if self._is_valid_python(candidate_str):
                lines = candidate
                logger.info(f"Replaced lines {start}-{end} with {len(raw_lines)} lines (raw)")
                continue

            # Try 2: auto-dedent + re-indent to match original indentation
            original_line = lines[start - 1]
            target_indent = original_line[: len(original_line) - len(original_line.lstrip())]
            reindented = self._reindent(new_code, target_indent)
            candidate = lines[:start - 1] + reindented + lines[end:]
            candidate_str = "".join(candidate)

            if self._is_valid_python(candidate_str):
                lines = candidate
                logger.info(f"Replaced lines {start}-{end} with {len(reindented)} lines (re-indented)")
                continue

            # Both failed — apply raw anyway, let the caller's compile check catch it
            lines[start - 1:end] = raw_lines
            logger.warning(f"Lines {start}-{end}: both raw and re-indented failed to compile, applying raw")

        return "".join(lines)

    def _code_to_lines(self, code: str) -> list[str]:
        """Convert code string to list of lines with newline endings."""
        raw = code.splitlines(keepends=True)
        if raw and not raw[-1].endswith("\n"):
            raw[-1] += "\n"
        return raw

    def _reindent(self, code: str, target_indent: str) -> list[str]:
        """Dedent code and re-indent to target_indent level."""
        raw_lines = code.splitlines()
        min_indent = None
        for rl in raw_lines:
            if rl.strip():
                indent_len = len(rl) - len(rl.lstrip())
                if min_indent is None or indent_len < min_indent:
                    min_indent = indent_len
        min_indent = min_indent or 0

        result = []
        for rl in raw_lines:
            if rl.strip():
                dedented = rl[min_indent:] if len(rl) >= min_indent else rl.lstrip()
                result.append(target_indent + dedented + "\n")
            else:
                result.append("\n")
        return result

    def _install_missing_packages(self, error: str) -> None:
        """Install missing third-party packages, once each, with verification.

        Handles both `ModuleNotFoundError` and conservative "<X> is required"
        messages. Uses the import-name -> PyPI-name map so e.g. `hydra` installs
        `hydra-core` instead of the wrong package, and records failures so the
        LLM stops waiting for pip to solve them."""
        modules = {m.split(".")[0] for m in
                   re.findall(r"No module named ['\"]([^'\"]+)['\"]", error)}
        for m in re.findall(r"([A-Za-z_][\w-]*) is required", error):
            if m.lower() in REQUIRED_MSG_ALLOWLIST:
                modules.add(m.lower())

        for module in modules:
            if module in self._install_attempted:
                continue
            self._install_attempted.add(module)
            pkg = module_to_pip_name(module)
            logger.info(f"Auto-installing: {pkg} (for module '{module}')")
            try:
                r = subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                                   capture_output=True, text=True, timeout=300)
                if r.returncode != 0:
                    logger.warning(f"pip install {pkg} failed: {r.stderr[-300:]}")
                    self.pip_notes.append(
                        f"pip install {pkg} FAILED — module '{module}' must be handled in "
                        f"code (vendor a fallback or remove the dependency)")
                    continue
                check = subprocess.run(
                    [sys.executable, "-c", f"import {module}"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(Path(self.output_dir)),
                )
                if check.returncode == 0:
                    logger.info(f"Installed and verified: {pkg}")
                    append_requirement(self.codebase_dir, pkg)
                else:
                    logger.warning(
                        f"pip install {pkg} succeeded but 'import {module}' still fails")
                    self.pip_notes.append(
                        f"pip installed '{pkg}' but module '{module}' still does not import — "
                        f"the PyPI package name likely differs; handle it in code")
            except Exception as e:
                logger.warning(f"Failed to install {pkg}: {e}")

    # ── Running the code ──────────────────────────────────────────────

    def _detect_entry_point(self) -> str | None:
        """Entry point resolution: explicit override > architecture > reproduce.sh
        > conventional names. Returns a path relative to the codebase dir."""
        if self.entry_point_override:
            return self.entry_point_override

        arch = self.global_memory.get("architecture", {})
        if isinstance(arch, dict) and arch.get("entry_point"):
            ep = Path(arch["entry_point"])
            try:
                ep = ep.relative_to(self.codebase_dir)
            except ValueError:
                pass
            if (self.codebase_dir / ep).exists():
                return str(ep)

        reproduce = self.codebase_dir / "reproduce.sh"
        if reproduce.exists():
            try:
                for line in reproduce.read_text().splitlines():
                    m = re.match(r"^\s*(?:exec\s+)?python3?\s+([\w./-]+\.py)\b", line)
                    if m and (self.codebase_dir / m.group(1)).exists():
                        return m.group(1)
            except Exception:
                pass

        for cand in ("main.py", "run.py", "train.py", "run_experiments.py", "run_experiment.py"):
            if (self.codebase_dir / cand).exists():
                return cand
        return None

    def _run_code(self) -> dict[str, Any]:
        entry = self._detect_entry_point()
        if entry is None:
            return {"success": False, "error": "Entry point not found", "stdout": ""}
        entry_path = self.codebase_dir.resolve() / entry
        logger.info(f"Running: {sys.executable} {entry_path} (timeout={self.run_timeout}s)")
        if not entry_path.exists():
            return {"success": False, "error": f"Entry point not found: {entry_path}", "stdout": ""}

        try:
            result = subprocess.run(
                [sys.executable, str(entry_path)],
                capture_output=True, text=True, timeout=self.run_timeout,
                cwd=str(self.codebase_dir),
            )
            if result.returncode == 0:
                return {"success": True, "stdout": result.stdout, "verified": True}
            error_text = (result.stderr or result.stdout
                          or f"Exit code {result.returncode} (no output)")
            return {"success": False, "error": error_text, "stdout": result.stdout or ""}
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = e.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            combined = stdout + stderr
            if not combined.strip():
                if self.strict_timeout:
                    logger.warning(f"{self.run_timeout}s timeout with NO output — treating as hang")
                    return {
                        "success": False,
                        "error": (f"Process ran {self.run_timeout}s and produced no output at "
                                  f"all — likely hung (deadlock, waiting for input, or a "
                                  f"silent download). Add progress logging/flushing or fix "
                                  f"the hang."),
                        "stdout": "",
                    }
                logger.info(f"{self.run_timeout}s timeout with no output — success (unverified)")
                return {"success": True, "stdout": "", "verified": False, "silent": True,
                        "note": f"timed out at {self.run_timeout}s with no output"}
            has_progress = bool(PROGRESS_MARKERS.search(combined[-4000:]))
            logger.info(f"{self.run_timeout}s timeout — code is running "
                        f"(progress markers {'found' if has_progress else 'not found'})")
            return {"success": True, "stdout": stdout, "verified": has_progress,
                    "note": (f"timed out at {self.run_timeout}s; progress markers "
                             f"{'found' if has_progress else 'not found'}")}
        except Exception as e:
            return {"success": False, "error": str(e), "stdout": ""}

    def _normalize_path(self, p: str) -> str | None:
        path = Path(p)
        for base in (self.codebase_dir.resolve(), self.codebase_dir):
            try:
                return str(path.relative_to(base))
            except ValueError:
                pass
        if not path.is_absolute():
            for base in (self.codebase_dir.resolve(), self.codebase_dir):
                try:
                    return str(Path("/" + p).relative_to(base))
                except ValueError:
                    pass
        if not path.is_absolute() and self._is_codebase_path(p):
            return p
        return p

    def _is_codebase_path(self, p: str) -> bool:
        path = Path(p)
        if path.is_absolute():
            return False
        first = path.parts[0] if path.parts else ""
        if first in ("home", "usr", "lib", "etc", "opt", "var", "tmp", "root", "proc"):
            return False
        try:
            (self.codebase_dir.resolve() / p).resolve().relative_to(self.codebase_dir.resolve())
            return True
        except ValueError:
            return False

    def _is_orchestrator_file(self, file_path: str) -> bool:
        """Check if a file is an orchestrator based on architecture metadata."""
        architecture = self.global_memory.get("architecture", {})
        entry_point = architecture.get("entry_point", "main.py")
        if file_path in (entry_point, "reproduce.sh"):
            return True
        for f in architecture.get("files", []):
            if isinstance(f, dict) and f.get("path") == file_path:
                return f.get("orchestrator", False)
        return False

    def _is_valid_python(self, code: str) -> bool:
        if not code or not code.strip():
            return False
        stripped = code.strip()
        if stripped.startswith("<") or "<invoke" in stripped[:200]:
            return False
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _clean_code(self, response: str) -> str:
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("import ", "from ", "#!", "# ", "class ", "def ")):
                return "\n".join(lines[i:]).strip()
        return response.strip()
