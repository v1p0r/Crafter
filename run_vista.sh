#!/bin/bash
# Full PTC pipeline on Vista (VLM + debug + unit tests all on by default).
# Usage: ANTHROPIC_API_KEY=<key> GEMINI_API_KEY=<key> ./run_vista.sh <input_dir> [--target-codebase-blacklist GitHub-URL] [--output dir] [--model name] [--paper name] [--no-vlm] [--skip-debug] [--no-unit-test] [--debug-iterations N]
#   <input_dir>     directory containing *_content_list_v2.json and images/
#   --output        explicit output dir; overrides auto path (default: data/output/<model>/<paper>)
#   --model         label for the output subdir (default: MiniMax-M2.7, matches BACKENDS["minimax"]["model"]). Does NOT change backend.
#   --paper         paper name for the output subdir (default: derived from input path)
#   --no-vlm        skip VLM figure analysis
#   --skip-debug    skip debug loop
#   --no-unit-test  skip unit-test generation
#   --only-debug    skip stages 1–4, only run debug (+ unit tests if --unit-test). Requires existing output dir.
#   --debug-iterations N  maximum debug cycles (default: 30)
#   --target-codebase-blacklist URL  block the target paper's official GitHub repo; repeatable

set -eo pipefail

# --- Load Vista modules (per global CLAUDE.md) ---
module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

# --- Activate ptc venv ---
source "$WORK/python_venvs/ptc/bin/activate"
echo "python: $(which python)"
echo "pip:    $(which pip)"

# --- Defaults: full pipeline (VLM + debug + unit tests all on) ---
VLM_FLAG=""
SKIP_DEBUG=""
UNIT_TEST="--unit-test"
ONLY_DEBUG=""
DEBUG_ITER="--debug-iterations 30"
OUTPUT_DIR=""
MODEL="MiniMax-M2.7"
PAPER_NAME=""
INPUT_DIR="data/parsed/adaptive-pruning/2401.12200v2/hybrid_auto/"
TARGET_CODEBASE_BLACKLIST=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --paper)        PAPER_NAME="$2"; shift 2 ;;
        --no-vlm)       VLM_FLAG="--no-vlm"; shift ;;
        --vlm)          VLM_FLAG=""; shift ;;
        --skip-debug)   SKIP_DEBUG="--skip-debug"; shift ;;
        --debug)        SKIP_DEBUG=""; shift ;;
        --no-unit-test) UNIT_TEST=""; shift ;;
        --unit-test)    UNIT_TEST="--unit-test"; shift ;;
        --only-debug)   ONLY_DEBUG="--only-debug"; shift ;;
        --debug-iterations) DEBUG_ITER="--debug-iterations $2"; shift 2 ;;
        --target-codebase-blacklist)
            if [ $# -lt 2 ]; then
                echo "Error: --target-codebase-blacklist requires a GitHub repository URL"
                exit 1
            fi
            TARGET_CODEBASE_BLACKLIST+=("--target-codebase-blacklist" "$2")
            shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *)
            INPUT_DIR="$1"
            shift ;;
    esac
done

# --- Auto-derive paper name from input path if not given ---
# Expected layout: .../<paper_name>/<paper_id>/<parse_method>/
# e.g. data/parsed/robust-clip/2402.12336v2/hybrid_auto/ -> robust-clip
if [ -z "$PAPER_NAME" ] && [ -n "$INPUT_DIR" ]; then
    _dir="${INPUT_DIR%/}"                     # strip trailing slash
    PAPER_NAME="$(basename "$(dirname "$(dirname "$_dir")")")"
fi

# --- Build output path if user didn't override ---
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="data/output/$MODEL/$PAPER_NAME"
fi

if [ -z "$INPUT_DIR" ] && [ -z "$ONLY_DEBUG" ]; then
    echo "Usage: ANTHROPIC_API_KEY=<key> GEMINI_API_KEY=<key> ./run_vista.sh <input_dir> [--output dir] [--no-vlm] [--skip-debug] [--no-unit-test] [--only-debug]"
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

if [ -z "$ONLY_DEBUG" ] && [ -z "$VLM_FLAG" ] && [ -z "$GEMINI_API_KEY" ] && [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: --vlm requires GEMINI_API_KEY or DASHSCOPE_API_KEY"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/$([ -n "$ONLY_DEBUG" ] && echo debug.log || echo pipeline.log)"

echo "PTC Agent — Vista"
if [ -n "$ONLY_DEBUG" ]; then
    echo "  Mode:      debug-only (skipping stages 1–4)"
    echo "  Output:    $OUTPUT_DIR"
    echo "  Log:       $LOG_FILE"
    echo "  UnitTest:  $([ -n "$UNIT_TEST" ] && echo enabled || echo disabled)"
    echo "  DebugIter: ${DEBUG_ITER##* }"
    python -u -m src.main --output "$OUTPUT_DIR" $ONLY_DEBUG $UNIT_TEST $DEBUG_ITER 2>&1 | tee "$LOG_FILE"
else
    echo "  Input:     $INPUT_DIR"
    echo "  Paper:     $PAPER_NAME"
    echo "  Model:     $MODEL"
    echo "  Output:    $OUTPUT_DIR"
    echo "  Log:       $LOG_FILE"
    echo "  VLM:       $([ -z "$VLM_FLAG" ] && echo enabled || echo disabled)"
    echo "  Debug:     $([ -z "$SKIP_DEBUG" ] && echo enabled || echo skipped)"
    echo "  DebugIter: ${DEBUG_ITER##* }"
    echo "  UnitTest:  $([ -n "$UNIT_TEST" ] && echo enabled || echo disabled)"
    python -u -m src.main --input "$INPUT_DIR" --output "$OUTPUT_DIR" "${TARGET_CODEBASE_BLACKLIST[@]}" $VLM_FLAG $SKIP_DEBUG $UNIT_TEST $DEBUG_ITER 2>&1 | tee "$LOG_FILE"
fi

# --- Run generated unit tests if they exist (don't fail the script on red tests) ---
if [ -d "$OUTPUT_DIR/codebase/tests" ]; then
    TEST_LOG="$OUTPUT_DIR/test_results.log"
    echo ""
    echo "=== Running generated unit tests ==="
    echo "Log: $TEST_LOG"
    ( cd "$OUTPUT_DIR/codebase" && python -u -m unittest discover tests -v ) 2>&1 | tee "$TEST_LOG" || true
fi
