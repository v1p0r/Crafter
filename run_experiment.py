"""Run the PTC pipeline on a paper with each backend and collect token/cost data."""
import argparse
import asyncio
import json
import time
from pathlib import Path

from src.utils.llm_client import set_backend, get_usage, reset_usage, BACKENDS
from src.main import run_pipeline, _find_content_list


PAPER = "adaptive-pruning"
PARSED_DIR = "data/parsed"
OUTPUT_BASE = "data/experiments"


async def run_one(
    backend: str,
    input_dir: str,
    images_dir: str,
    output_dir: str,
    skip_debug: bool,
    target_codebase_blacklist: list[str],
):
    set_backend(backend)
    reset_usage()

    start = time.time()
    try:
        await run_pipeline(
            input_path=_find_content_list(Path(input_dir)),
            images_dir=images_dir,
            output_dir=output_dir,
            enable_vlm=True,
            skip_debug=skip_debug,
            target_codebase_blacklist=target_codebase_blacklist,
        )
        status = "success"
        error = None
    except Exception as e:
        status = "error"
        error = str(e)
        print(f"  [{backend}] ERROR: {e}")

    elapsed = time.time() - start
    usage = get_usage()
    result = usage.summary(backend)
    result["status"] = status
    result["error"] = error
    result["elapsed_seconds"] = round(elapsed, 1)
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=PAPER)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS.keys()))
    parser.add_argument("--skip-debug", action="store_true", default=True)
    parser.add_argument("--output-base", default=OUTPUT_BASE)
    parser.add_argument("--target-codebase-blacklist", action="append", default=[])
    args = parser.parse_args()

    mineru_dir = None
    for d in Path(PARSED_DIR, args.paper).rglob("*_content_list_v2.json"):
        mineru_dir = str(d.parent)
        break
    if not mineru_dir:
        print(f"No content_list_v2.json found for {args.paper}")
        return

    images_dir = str(Path(mineru_dir) / "images")
    results = {}

    for backend in args.backends:
        output_dir = str(Path(args.output_base) / f"{args.paper}_{backend}")
        print(f"\n{'='*60}")
        print(f"Running: {backend} ({BACKENDS[backend]['model']})")
        print(f"Output:  {output_dir}")
        print(f"{'='*60}")

        result = await run_one(
            backend,
            mineru_dir,
            images_dir,
            output_dir,
            args.skip_debug,
            args.target_codebase_blacklist,
        )
        results[backend] = result

        # Save incrementally
        report_path = Path(args.output_base) / f"{args.paper}_results.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2))

        print(f"  Status: {result['status']} | Time: {result['elapsed_seconds']}s")
        print(f"  Tokens: {result['total_input_tokens']} in / {result['total_output_tokens']} out")
        print(f"  Cost:   ${result['estimated_cost_usd']}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.paper}")
    print(f"{'='*60}")
    print(f"{'Backend':<12} {'Model':<28} {'In Tok':>8} {'Out Tok':>8} {'Cost':>8} {'Time':>7} {'Status'}")
    print("-" * 95)
    for name, r in results.items():
        print(f"{name:<12} {r['model']:<28} {r['total_input_tokens']:>8} {r['total_output_tokens']:>8} "
              f"${r['estimated_cost_usd']:>6.4f} {r['elapsed_seconds']:>6.1f}s {r['status']}")


if __name__ == "__main__":
    asyncio.run(main())
