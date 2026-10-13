#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ptc

set -e

# Defaults
VLM_FLAG="--no-vlm"
SKIP_DEBUG="--skip-debug"
OUTPUT_DIR="data/output"
DEBUG_ITERATIONS=30

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --paper)
            SINGLE_PAPER="$2"
            shift 2
            ;;
        --no-vlm)
            VLM_FLAG="--no-vlm"
            shift
            ;;
        --vlm)
            VLM_FLAG=""
            shift
            ;;
        --skip-debug)
            SKIP_DEBUG="--skip-debug"
            shift
            ;;
        --debug)
            SKIP_DEBUG=""
            shift
            ;;
        --only-debug)
            ONLY_DEBUG="--only-debug"
            shift
            ;;
        --debug-iterations)
            DEBUG_ITERATIONS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check env vars
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

if [ -z "$VLM_FLAG" ] && [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set (required for VLM, use --no-vlm to skip)"
    exit 1
fi

echo "PTC Agent Main — Local Run"
echo "Output: $OUTPUT_DIR"
echo "VLM: $([ -z "$VLM_FLAG" ] && echo 'enabled' || echo 'disabled')"
echo "Debug: $([ -z "$SKIP_DEBUG" ] && echo 'enabled' || echo 'skipped')"
echo "Debug max iterations: $DEBUG_ITERATIONS"
echo ""

# --only-debug mode: run debug on existing output only
if [ -n "$ONLY_DEBUG" ]; then
    echo "Running debug-only mode on $OUTPUT_DIR"
    python3 -m src.main \
        --output "$OUTPUT_DIR" \
        --only-debug \
        --debug-iterations "$DEBUG_ITERATIONS"
    exit 0
fi

# Build paper list
if [ -n "$SINGLE_PAPER" ]; then
    PAPERS=("$SINGLE_PAPER")
else
    PAPERS=($(ls -1 data/parsed/))
fi

echo "Papers to process (${#PAPERS[@]}): ${PAPERS[*]}"
echo ""

FAILED=()
SUCCEEDED=()

for PAPER in "${PAPERS[@]}"; do
    PARSED_DIR="data/parsed/$PAPER"
    PAPER_OUTPUT="$OUTPUT_DIR/$PAPER"

    if [ -d "$PAPER_OUTPUT" ]; then
        echo "[$PAPER] Output already exists, skipping."
        SUCCEEDED+=("$PAPER (skipped)")
        continue
    fi

    echo "============================================"
    echo "Processing: $PAPER"
    echo "============================================"

    MINERU_OUTPUT=$(find "$PARSED_DIR" -name "*_content_list_v2.json" -exec dirname {} \; | head -1)
    if [ -z "$MINERU_OUTPUT" ]; then
        echo "[$PAPER] ERROR: No content_list_v2.json found, skipping."
        FAILED+=("$PAPER")
        continue
    fi

    if python3 -m src.main \
        --input "$MINERU_OUTPUT" \
        --output "$PAPER_OUTPUT" \
        --debug-iterations "$DEBUG_ITERATIONS" \
        $VLM_FLAG \
        $SKIP_DEBUG; then
        echo "[$PAPER] Done."
        SUCCEEDED+=("$PAPER")
    else
        echo "[$PAPER] FAILED (exit code $?)."
        FAILED+=("$PAPER")
    fi

    echo ""
done

echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]}"
echo "Failed    (${#FAILED[@]}):    ${FAILED[*]}"

[ ${#FAILED[@]} -eq 0 ] || exit 1
