#!/bin/bash
set -e

# Defaults
VLM_FLAG=""
OUTPUT_DIR=""
SKIP_DEBUG=""
ONLY_DEBUG=""
UNIT_TEST=""
TARGET_CODEBASE_BLACKLIST=()

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-vlm)
            VLM_FLAG="--no-vlm"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-debug)
            SKIP_DEBUG="--skip-debug"
            shift
            ;;
        --only-debug)
            ONLY_DEBUG="--only-debug"
            shift
            ;;
        --unit-test)
            UNIT_TEST="--unit-test"
            shift
            ;;
        --target-codebase-blacklist)
            if [ $# -lt 2 ]; then
                echo "Error: --target-codebase-blacklist requires a GitHub repository URL"
                exit 1
            fi
            TARGET_CODEBASE_BLACKLIST+=("--target-codebase-blacklist" "$2")
            shift 2
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$ONLY_DEBUG" ] && [ -z "$INPUT_DIR" ]; then
    echo "Usage: ./run.sh <input_dir> [--output dir] [--target-codebase-blacklist GitHub-URL] [--no-vlm] [--skip-debug] [--only-debug] [--unit-test]"
    exit 1
fi

if [ -n "$ONLY_DEBUG" ] && [ -z "$INPUT_DIR" ] && [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --only-debug requires either <input_dir> or --output <dir> to know which paper to debug"
    exit 1
fi

# Derive output dir from input path if not explicitly set
# e.g. .../bridging-data-gaps/icml2024/hybrid_auto -> data/output/bridging-data-gaps
if [ -z "$OUTPUT_DIR" ]; then
    if [ -n "$INPUT_DIR" ]; then
        # Walk up from input dir to find the paper name (parent of venue dir like icml2024)
        _dir="$INPUT_DIR"
        # Strip trailing slash
        _dir="${_dir%/}"
        # Go up until we find a directory whose child matches a venue pattern (e.g. icml2024, neurips2023)
        _paper_name=""
        while [ -n "$_dir" ] && [ "$_dir" != "/" ]; do
            _base="$(basename "$_dir")"
            _parent="$(dirname "$_dir")"
            _parent_base="$(basename "$_parent")"
            # Check if current dir looks like a venue (contains a 4-digit year)
            if echo "$_base" | grep -qE '[a-z]+[0-9]{4}'; then
                _paper_name="$_parent_base"
                break
            fi
            _dir="$_parent"
        done
        if [ -n "$_paper_name" ] && [ "$_paper_name" != "/" ]; then
            OUTPUT_DIR="data/output/$_paper_name"
        else
            # Fallback: use the grandparent directory name
            OUTPUT_DIR="data/output/$(basename "$(dirname "$(dirname "$INPUT_DIR")")")"
        fi
    else
        OUTPUT_DIR="data/output"
    fi
fi

# Check env vars
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    exit 1
fi

if [ -z "$ONLY_DEBUG" ] && [ -z "$VLM_FLAG" ] && [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set (required for VLM, use --no-vlm to skip)"
    exit 1
fi

echo "PTC Agent — Paper to Code"
echo "Output: $OUTPUT_DIR"
if [ -n "$ONLY_DEBUG" ]; then
    echo "Mode: debug only"
else
    echo "Input: $INPUT_DIR"
    echo "VLM: $([ -z "$VLM_FLAG" ] && echo 'enabled' || echo 'disabled')"
fi
echo ""

if [ -n "$ONLY_DEBUG" ]; then
    python -m src.main --output "$OUTPUT_DIR" $ONLY_DEBUG $UNIT_TEST
else
    python -m src.main --input "$INPUT_DIR" --output "$OUTPUT_DIR" "${TARGET_CODEBASE_BLACKLIST[@]}" $VLM_FLAG $SKIP_DEBUG $UNIT_TEST
fi
