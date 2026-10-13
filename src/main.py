import argparse
import asyncio
import json
import os
import random
import stat
from datetime import datetime, timezone
from pathlib import Path

from src.agents.codegen_agent import CodegenAgent
from src.agents.debug_agent import DebugAgent
from src.agents.gap_agent import GapAgent
from src.agents.partition_agent import PartitionAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.unittest_agent import UnitTestAgent
from src.agents.vlm_agent import VLMAgent
from src.memory.global_memory import GlobalMemory
from src.utils.gap_matching import attach_target_files_to_context
from src.utils.gap_source_discovery import normalize_github_repository_blacklist
from src.utils.llm_client import get_usage, reset_usage, write_usage_summary
from src.utils.requirements import ensure_requirements


REPOSITORY_ACCESS_POLICY_VERSION = 1


def _print_usage(stage: str):
    u = get_usage()
    usage = u.per_stage.get(stage, {})
    print(
        f"  [{stage}] tokens: "
        f"in={usage.get('input_tokens', 0):,} "
        f"(uncached={usage.get('uncached_input_tokens', 0):,}, "
        f"cache_write={usage.get('cache_creation_input_tokens', 0):,}, "
        f"cache_read={usage.get('cache_read_input_tokens', 0):,}) "
        f"out={usage.get('output_tokens', 0):,} "
        f"(visible={usage.get('visible_output_tokens', 0):,}, "
        f"reasoning={usage.get('reasoning_output_tokens', 0):,}) | "
        f"responses={usage.get('calls', 0)} | "
        f"cost=${usage.get('estimated_cost_usd', 0.0):.6f}"
    )


def _gap_cache_matches_repository_policy(
    supplementary_context: object,
    target_codebase_blacklist: list[str],
) -> bool:
    """Reject cached gap output without an exact contamination-policy attestation."""
    if not isinstance(supplementary_context, dict):
        return False
    source_summary = supplementary_context.get("gap_source_summary") or {}
    policy = source_summary.get("repository_access_policy") or {}
    return (
        policy.get("schema_version") == REPOSITORY_ACCESS_POLICY_VERSION
        and policy.get("target_codebase_blacklist") == target_codebase_blacklist
        and policy.get("target_implementations_allowed") is False
    )


def _has_cached_downstream_work(global_memory: GlobalMemory, codebase_dir: str) -> bool:
    """Return whether stale gap evidence may already have influenced later stages."""
    if global_memory.get("module_specs") or global_memory.get("architecture"):
        return True
    codebase = Path(codebase_dir)
    return codebase.is_dir() and any(path.is_file() for path in codebase.rglob("*"))


async def run_pipeline(
    input_path: str,
    images_dir: str,
    output_dir: str,
    enable_vlm: bool = True,
    skip_debug: bool = False,
    only_debug: bool = False,
    skip_gaps: bool = False,
    enable_unittest: bool = False,
    debug_iterations: int = 20,
    codebase_dir_override: str | None = None,
    entry_point: str | None = None,
    run_timeout: int = 120,
    strict_timeout: bool = False,
    target_codebase_blacklist: list[str] | None = None,
):
    canonical_blacklist = normalize_github_repository_blacklist(
        target_codebase_blacklist or []
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    codebase_dir = codebase_dir_override or str(output / "codebase")

    global_memory = GlobalMemory(path=str(output / "global_memory"))
    # Preserve the source hint for verified official-paper recovery.  A modern
    # arXiv ID in the input directory is safer and cheaper than title search;
    # the GapAgent still verifies the recovered document title before use.
    global_memory.set("input_path", input_path)
    global_memory.set("target_codebase_blacklist", canonical_blacklist)
    if not only_debug and not skip_gaps and not canonical_blacklist:
        print(
            "Repository evidence disabled: no --target-codebase-blacklist was supplied"
        )

    if only_debug:
        print(f"[5/6] Running debug loop only (--only-debug, max_iterations={debug_iterations})...")
        debug_agent = DebugAgent(
            global_memory=global_memory,
            output_dir=output_dir,
            codebase_dir=codebase_dir,
            entry_point=entry_point,
            run_timeout=run_timeout,
            strict_timeout=strict_timeout,
        )
        result = await debug_agent.run(max_iterations=debug_iterations)
        _print_usage("debug")
        print(f"Debug complete: {result}")

        if enable_unittest:
            print("[6/6] Generating unit tests...")
            unittest_agent = UnitTestAgent(
                global_memory=global_memory,
                output_dir=output_dir,
                codebase_dir=codebase_dir,
            )
            ut_result = await unittest_agent.run()
            _print_usage("unittest")
            print(f"Unit tests complete: {ut_result}")
        return

    # Stage 1: Partition (skip if cached)
    if global_memory.get("partitions"):
        print("[1/6] Skipping partition (cached)")
    else:
        print("[1/6] Partitioning paper...")
        partition_agent = PartitionAgent(
            global_memory=global_memory,
            input_path=input_path,
            output_dir=output_dir,
        )
        await partition_agent.run()
        _print_usage("partition")

    # Stage 2: VLM figure analysis
    if not enable_vlm:
        print("[2/6] Skipping VLM figure analysis (--no-vlm)")
        global_memory.set("figure_analysis", [])
    elif global_memory.get("figure_analysis") is not None:
        print("[2/6] Skipping VLM (cached)")
    else:
        print("[2/6] Analyzing figures...")
        vlm_agent = VLMAgent(
            global_memory=global_memory,
            images_dir=images_dir,
            output_dir=output_dir,
        )
        await vlm_agent.run()
        _print_usage("vlm")

    # Stage 2.5: Gap analysis
    if skip_gaps:
        print("[2.5/6] Skipping gap analysis (--skip-gaps)")
        global_memory.set("supplementary_context", {})
    elif _gap_cache_matches_repository_policy(
        global_memory.get("supplementary_context"), canonical_blacklist,
    ):
        print("[2.5/6] Skipping gap analysis (cached)")
    else:
        if global_memory.get("supplementary_context") is not None:
            if _has_cached_downstream_work(global_memory, codebase_dir):
                raise RuntimeError(
                    "stale_repository_access_policy_with_downstream_artifacts; "
                    "use a fresh --output directory to prevent reuse of code or plans "
                    "that may have seen the target codebase"
                )
            print("[2.5/6] Cached gap analysis has a stale repository policy; rerunning")
        print("[2.5/6] Analyzing information gaps...")
        gap_agent = GapAgent(
            global_memory=global_memory,
            output_dir=output_dir,
        )
        await gap_agent.run()
        _print_usage("gap")

    # Stage 3: Planning (skip if cached)
    if global_memory.get("module_specs"):
        print("[3/6] Skipping planning (cached)")
    else:
        print("[3/6] Planning...")
        planning_agent = PlanningAgent(
            global_memory=global_memory,
            output_dir=output_dir,
        )
        await planning_agent.run()
        _print_usage("planning")

    # Also run on resumed/cached planning states so legacy supplementary
    # context cannot bypass post-architecture routing.
    cached_architecture = global_memory.get("architecture", {})
    cached_supplementary = global_memory.get("supplementary_context", {})
    if cached_architecture.get("files") and cached_supplementary:
        global_memory.set(
            "supplementary_context",
            attach_target_files_to_context(cached_supplementary, cached_architecture),
        )

    # Stage 4: Code generation — two-pass (modules first, orchestrators second)
    architecture = global_memory.get("architecture", {})
    files = architecture.get("files", [])
    entry_point = architecture.get("entry_point", "main.py")

    # Build orchestrator lookup
    orchestrator_lookup = {}
    for f in files:
        if not isinstance(f, dict):
            continue
        path = f.get("path", "")
        is_orch = f.get("orchestrator", False)
        # Enforce: entry_point and reproduce.sh are always orchestrators
        if path in (entry_point, "reproduce.sh"):
            is_orch = True
        orchestrator_lookup[path] = is_orch

    # Separate into module files and orchestrator files
    ordered_files = _topological_sort(files)
    module_files = [f for f in ordered_files if not orchestrator_lookup.get(f, False)]
    orchestrator_files = [f for f in ordered_files if orchestrator_lookup.get(f, False)]

    # Ensure reproduce.sh is last among orchestrators
    if "reproduce.sh" in orchestrator_files:
        orchestrator_files.remove("reproduce.sh")
        orchestrator_files.append("reproduce.sh")

    # Skip Stage 4 if codebase already has every planned file — otherwise rerun all.
    all_planned = set(module_files + orchestrator_files)
    all_present = bool(all_planned) and all((Path(codebase_dir) / f).exists() for f in all_planned)
    if all_present:
        print(f"[4/6] Skipping codegen ({len(all_planned)} files already exist)")
    else:
        print("[4/6] Generating code...")
        # Pass 1: Module files (implementation code)
        print(f"  Pass 1: {len(module_files)} module files...")
        for file_path in module_files:
            print(f"    Generating {file_path}...")
            codegen_agent = CodegenAgent(
                file_path=file_path,
                global_memory=global_memory,
                output_dir=output_dir,
                codebase_dir=codebase_dir,
                is_orchestrator=False,
            )
            await codegen_agent.run()

        # Pass 2: Orchestrator files (import and wire modules)
        print(f"  Pass 2: {len(orchestrator_files)} orchestrator files...")
        for file_path in orchestrator_files:
            print(f"    Generating {file_path}...")
            codegen_agent = CodegenAgent(
                file_path=file_path,
                global_memory=global_memory,
                output_dir=output_dir,
                codebase_dir=codebase_dir,
                is_orchestrator=True,
            )
            await codegen_agent.run()
        _print_usage("codegen")

    # Stage 4.5: Ensure reproduce.sh, requirements.txt, and package __init__s exist
    _ensure_reproduce_sh(Path(codebase_dir), global_memory)
    ensure_requirements(codebase_dir)
    _ensure_package_inits(Path(codebase_dir))

    # Stage 5: Debug loop
    if skip_debug:
        print("[5/6] Skipping debug loop (--skip-debug)")
    else:
        print(f"[5/6] Debug loop (max_iterations={debug_iterations})...")
        debug_agent = DebugAgent(
            global_memory=global_memory,
            output_dir=output_dir,
            codebase_dir=codebase_dir,
            entry_point=entry_point,
            run_timeout=run_timeout,
            strict_timeout=strict_timeout,
        )
        result = await debug_agent.run(max_iterations=debug_iterations)
        _print_usage("debug")
        print(f"Debug complete: {result}")

    # Stage 6: Unit test generation (opt-in)
    if enable_unittest:
        print("[6/6] Generating unit tests...")
        unittest_agent = UnitTestAgent(
            global_memory=global_memory,
            output_dir=output_dir,
            codebase_dir=codebase_dir,
        )
        ut_result = await unittest_agent.run()
        _print_usage("unittest")
        print(f"Unit tests complete: {ut_result}")

    # Final usage summary
    u = get_usage()
    print(f"\n{'='*60}")
    print(
        f"  CURRENT PROCESS TOTAL: tokens in={u.input_tokens:,} "
        f"out={u.output_tokens:,} | responses={u.calls} | cost=${u.cost():.6f}"
    )
    print(f"  Per stage: {json.dumps(u.per_stage, indent=4)}")
    print(f"{'='*60}")


def _ensure_reproduce_sh(codebase_dir: Path, global_memory) -> None:
    """Ensure reproduce.sh exists in the codebase and is executable."""
    codebase_dir.mkdir(parents=True, exist_ok=True)
    reproduce_path = codebase_dir / "reproduce.sh"

    if not reproduce_path.exists():
        # Generate a fallback reproduce.sh from the architecture
        architecture = global_memory.get("architecture", {})
        entry_point = architecture.get("entry_point", "main.py")

        print("  reproduce.sh not found — generating fallback...")
        script = (
            "#!/bin/bash\n"
            "set -e\n"
            "exec > >(tee -a reproduce.log) 2>&1\n\n"
            "echo '=== Starting reproduction ==='\n"
            "echo \"Date: $(date)\"\n"
            "echo \"Python: $(python --version)\"\n\n"
            "# Install dependencies\n"
            "python -m pip install -r requirements.txt\n\n"
            "# Run the full pipeline\n"
            f"python {entry_point}\n\n"
            "echo '=== Reproduction complete ==='\n"
        )
        reproduce_path.write_text(script)
        generated_code = global_memory.get("generated_code", {})
        generated_code["reproduce.sh"] = script
        global_memory.set("generated_code", generated_code)

    # Make executable
    st = reproduce_path.stat()
    reproduce_path.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  reproduce.sh ready at {reproduce_path}")


def _ensure_package_inits(codebase_dir: Path) -> None:
    """Touch an empty __init__.py in every subdirectory containing .py files so
    imports work under `unittest discover` and other strict package contexts."""
    if not codebase_dir.is_dir():
        return
    for d in {p.parent for p in codebase_dir.rglob("*.py") if p.parent != codebase_dir}:
        init_file = d / "__init__.py"
        if not init_file.exists():
            init_file.touch()


def _topological_sort(files: list[dict]) -> list[str]:
    if not files:
        return []

    # Build adjacency from dependency info
    graph: dict[str, list[str]] = {}
    all_files: list[str] = []

    for f in files:
        path = f.get("path", "") if isinstance(f, dict) else f
        deps = f.get("dependencies", []) if isinstance(f, dict) else []
        graph[path] = deps
        all_files.append(path)

    # Kahn's algorithm
    in_degree = {f: 0 for f in all_files}
    for f, deps in graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[f] += 1

    queue = [f for f in all_files if in_degree[f] == 0]
    ordered = []

    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for f, deps in graph.items():
            if node in deps:
                in_degree[f] -= 1
                if in_degree[f] == 0 and f not in ordered:
                    queue.append(f)

    # Add any remaining (circular deps)
    for f in all_files:
        if f not in ordered:
            ordered.append(f)

    return ordered


def _find_content_list(input_dir: Path) -> str:
    matches = list(input_dir.glob("*_content_list_v2.json"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No *_content_list_v2.json found in {input_dir}")


def _run_with_usage_accounting(coroutine, output_dir: str, metadata: dict) -> None:
    """Run one CLI invocation and always refresh its durable usage summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(
        os.environ.get("PTC_TOKEN_USAGE_LOG") or output / "token_usage.jsonl"
    )
    summary_path = Path(
        os.environ.get("PTC_TOKEN_USAGE_SUMMARY") or output / "token_usage_summary.json"
    )
    os.environ["PTC_TOKEN_USAGE_LOG"] = str(ledger_path)
    os.environ["PTC_TOKEN_USAGE_SUMMARY"] = str(summary_path)
    os.environ["PTC_PAPER"] = os.environ.get("PTC_PAPER") or output.name
    os.environ["PTC_RUN_ID"] = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-pid{os.getpid()}"
    )
    reset_usage()

    run_metadata = {
        **metadata,
        "paper": os.environ["PTC_PAPER"],
        "last_run_id": os.environ["PTC_RUN_ID"],
        "last_run_status": "running",
    }
    run_failed = False
    try:
        asyncio.run(coroutine)
        run_metadata["last_run_status"] = "completed"
    except BaseException as exc:
        run_failed = True
        run_metadata["last_run_status"] = "failed"
        run_metadata["last_run_error_type"] = type(exc).__name__
        raise
    finally:
        try:
            summary = write_usage_summary(
                ledger_path,
                summary_path,
                metadata=run_metadata,
            )
            totals = summary["totals"]
            print(
                f"Token accounting: {totals['calls']} responses, "
                f"in={totals['input_tokens']:,}, out={totals['output_tokens']:,}, "
                f"estimated cost=${totals['estimated_cost_usd']:.6f}"
            )
            print(f"  Ledger: {ledger_path}")
            print(f"  Summary: {summary_path}")
        except Exception as accounting_exc:
            if not run_failed:
                raise
            print(f"WARNING: failed to write token usage summary: {accounting_exc}")


def main():
    parser = argparse.ArgumentParser(description="PTC Agent — Paper to Code")
    parser.add_argument("--input", required=False, default="", help="Path to input directory containing content_list JSON and images/")
    parser.add_argument("--output", default="data/output", help="Output directory")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM figure analysis")
    parser.add_argument("--skip-debug", action="store_true", help="Skip debug loop (for CPU-only environments)")
    parser.add_argument("--skip-gaps", action="store_true", help="Skip gap analysis (Stage 2.5)")
    parser.add_argument("--only-debug", action="store_true", help="Skip stages 1-4 and run only the debug loop (requires existing output)")
    parser.add_argument("--unit-test", action="store_true", help="Generate unit tests after debug stage")
    parser.add_argument("--debug-iterations", type=int, default=20, help="Max debug iterations (default: 20)")
    parser.add_argument(
        "--target-codebase-blacklist",
        action="append",
        default=[],
        metavar="GITHUB_URL",
        help=(
            "Official target-paper GitHub repository to block before repository "
            "discovery (repeat for multiple URLs). Without this option, repository "
            "evidence is disabled."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Local random seed recorded with the run (default: 42)")
    args = parser.parse_args()
    random.seed(args.seed)
    os.environ["PTC_RANDOM_SEED"] = str(args.seed)

    if args.only_debug:
        _run_with_usage_accounting(
            run_pipeline(
                "",
                "",
                args.output,
                only_debug=True,
                enable_unittest=args.unit_test,
                debug_iterations=args.debug_iterations,
            ),
            args.output,
            {
                "mode": "only_debug",
                "enable_unittest": args.unit_test,
                "debug_iterations": args.debug_iterations,
                "random_seed": args.seed,
            },
        )
        return

    if not args.input:
        parser.error("--input is required unless --only-debug is set")

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        parser.error(f"Input path must be a directory: {args.input}")

    content_list_path = _find_content_list(input_dir)
    images_dir = str(input_dir / "images")
    enable_vlm = not args.no_vlm

    if enable_vlm and not Path(images_dir).is_dir():
        parser.error(f"Images directory not found: {images_dir} (use --no-vlm to skip)")

    _run_with_usage_accounting(
        run_pipeline(
            content_list_path,
            images_dir,
            args.output,
            enable_vlm=enable_vlm,
            skip_debug=args.skip_debug,
            skip_gaps=args.skip_gaps,
            enable_unittest=args.unit_test,
            debug_iterations=args.debug_iterations,
            target_codebase_blacklist=args.target_codebase_blacklist,
        ),
        args.output,
        {
            "mode": "pipeline",
            "input_path": str(input_dir),
            "enable_vlm": enable_vlm,
            "skip_gaps": args.skip_gaps,
            "skip_debug": args.skip_debug,
            "enable_unittest": args.unit_test,
            "debug_iterations": args.debug_iterations,
            "random_seed": args.seed,
            "target_codebase_blacklist": normalize_github_repository_blacklist(
                args.target_codebase_blacklist
            ),
        },
    )


if __name__ == "__main__":
    main()
