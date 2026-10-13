"""Re-run only the VLM stage on the 10 papers that had VLM 503s during the
paperbench batch. Uses the new retry logic in call_vlm. Outputs to
data/output/MiniMax-M2.7/vlm-retest/<paper>/.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.vlm_agent import VLMAgent
from src.memory.global_memory import GlobalMemory
from src.utils.llm_client import get_usage, reset_usage

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PAPERS = [
    "ftrl", "lbcs", "lca-on-the-line", "pinn", "rice",
    "robust-clip", "sample-specific-masks", "sapg",
    "self-composing-policies", "sequential-neural-score-estimation",
]

INPUT_ROOT = Path("/Users/esther_lh/GitHub/paperbench_data/data/output_v2")
ORIGINAL_OUT = Path("/Users/esther_lh/GitHub/ptc_agent_merged/data/output/MiniMax-M2.7/skip-debug")
TEST_OUT = Path("/Users/esther_lh/GitHub/ptc_agent_merged/data/output/MiniMax-M2.7/vlm-retest")


def find_input_dir(paper: str) -> Path | None:
    for p in INPUT_ROOT.joinpath(paper).rglob("hybrid_auto"):
        if (p / "images").is_dir():
            return p
    return None


async def test_one(paper: str) -> dict:
    inp = find_input_dir(paper)
    if inp is None:
        return {"paper": paper, "error": "no hybrid_auto dir found"}

    images_dir = inp / "images"
    images_json = ORIGINAL_OUT / paper / "global_memory" / "images.json"
    if not images_json.exists():
        return {"paper": paper, "error": "no original images.json"}

    images = json.loads(images_json.read_text())

    test_paper_dir = TEST_OUT / paper
    test_paper_dir.mkdir(parents=True, exist_ok=True)

    gm = GlobalMemory(path=str(test_paper_dir / "global_memory"))
    gm.set("images", images)

    reset_usage()
    agent = VLMAgent(
        global_memory=gm,
        images_dir=str(images_dir),
        output_dir=str(test_paper_dir),
    )
    t0 = time.time()
    analyzed = await agent.run()
    dt = time.time() - t0

    # Pull original counts
    orig_fa = ORIGINAL_OUT / paper / "global_memory" / "figure_analysis.json"
    orig_count = 0
    if orig_fa.exists():
        try:
            d = json.loads(orig_fa.read_text())
            orig_count = len(d) if isinstance(d, (list, dict)) else 0
        except Exception:
            pass

    usage = get_usage().summary()
    return {
        "paper": paper,
        "n_images": len(images),
        "orig_analyzed": orig_count,
        "new_analyzed": len(analyzed),
        "elapsed_s": round(dt, 1),
        "vlm_tokens": usage.get("per_stage", {}).get("vlm", {}),
    }


async def main() -> None:
    results = []
    for p in PAPERS:
        print(f"\n=== {p} ===", flush=True)
        try:
            r = await test_one(p)
            print(json.dumps(r, indent=2), flush=True)
            results.append(r)
        except Exception as e:
            print(f"[{p}] FAILED: {e}", flush=True)
            results.append({"paper": p, "error": str(e)})

    print("\n=== SUMMARY ===")
    print(f"{'paper':<40} {'images':>6} {'orig':>5} {'new':>5} {'elapsed':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['paper']:<40} ERROR: {r['error']}")
        else:
            print(f"{r['paper']:<40} {r['n_images']:>6} {r['orig_analyzed']:>5} {r['new_analyzed']:>5} {r['elapsed_s']:>7.1f}s")

    out = TEST_OUT / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
