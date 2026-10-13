# PTC Agent

PTC Agent turns a parsed research paper into a runnable codebase. It partitions the paper, analyzes figures when enabled, fills implementation gaps, plans the project, generates files in dependency order, and runs a debug loop. Unit-test generation is optional.

## Pipeline

```text
paper partitioning
  -> figure analysis (optional)
  -> gap analysis
  -> project and experiment planning
  -> two-pass code generation
  -> debug loop (optional)
  -> unit-test generation (optional)
```

Run state is stored under the selected output directory. Reusing an output directory resumes from cached stages when possible.

## Requirements

- Python 3.10 or newer is required. The Vista setup uses Python 3.11.8.
- An API key is required for the selected main backend.
- A Gemini API key is required when figure analysis is enabled.

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the keys required by the selected run:

```bash
# Default backend used by run.sh
export ANTHROPIC_API_KEY=<minimax-key>

# Claude backend used through run_experiment.py
export CLAUDE_API_KEY=<claude-key>

# Figure analysis
export GEMINI_API_KEY=<gemini-key>
```

The main backend and the figure-analysis backend are separate. The current choices configured in `src/utils/llm_client.py` are MiniMax, Claude, GPT, DeepSeek, and Gemini. MiniMax is the default; its key is stored in `ANTHROPIC_API_KEY` because it uses an Anthropic-compatible endpoint. Claude uses `CLAUDE_API_KEY`.

`run.sh` uses the default MiniMax backend. Backend comparison runs can select one or more alternatives:

```bash
python run_experiment.py \
  --paper <paper-name> \
  --backends minimax claude \
  --output-base data/experiments
```

`run_experiment.py` looks for the paper under `data/parsed/<paper-name>`.

Figure analysis currently uses Gemini independently of the selected main backend. `--no-vlm` skips the figure-analysis stage entirely; it does not fall back to the main model.

## Input

The input argument is a directory, not a PDF file. It must contain one parsed content-list file:

```text
<input_dir>/
├── *_content_list_v2.json
└── images/                    # required unless --no-vlm is used
```

The content-list JSON is the paper representation consumed by the partition stage. The `images/` directory contains figures referenced by that representation.

## Run

Run the full pipeline:

```bash
./run.sh <input_dir> \
  --output data/output/<run-name> \
  --target-codebase-blacklist https://github.com/<owner>/<official-target-repo>
```

The target paper's official codebase must never be used as reproduction
evidence. Pass its GitHub URL with `--target-codebase-blacklist`; repeat the
option if the paper has multiple official repositories. Matching is performed
on canonical, case-insensitive `owner/repository` identity, so `.git` and paths
such as `/tree/main` cannot bypass it. The check happens before scope matching,
cloning, Git validation, or source inspection. When no blacklist is supplied,
repository evidence is disabled rather than risking target-code leakage.

Run without figure analysis:

```bash
./run.sh <input_dir> --output data/output/<run-name> --no-vlm
```

Generate code without entering the debug loop:

```bash
./run.sh <input_dir> --output data/output/<run-name> --skip-debug
```

Resume debugging an existing output:

```bash
./run.sh --only-debug --output data/output/<run-name>
```

Add `--unit-test` to generate tests after the debug stage.

### `run.sh` options

| Option | Meaning |
| --- | --- |
| `--output <dir>` | Select the output directory. If omitted, the script derives one from the input path. |
| `--target-codebase-blacklist <url>` | Block an official target-paper GitHub repository before any repository access; repeatable. |
| `--no-vlm` | Skip figure analysis entirely; the main model is not used as a vision fallback. |
| `--skip-debug` | Stop after code generation. |
| `--only-debug` | Run the debug loop against an existing output directory. |
| `--unit-test` | Generate unit tests after debugging. |

The Python entry point exposes two additional controls:

```bash
python -m src.main \
  --input <input_dir> \
  --output data/output/<run-name> \
  --skip-gaps \
  --debug-iterations 10
```

Use `python -m src.main --help` for the complete CLI reference.

## Vista

`run_vista.sh` loads the Vista GCC, CUDA, and Python modules, then activates `$WORK/python_venvs/ptc`. It enables figure analysis, debugging, and unit-test generation by default.

```bash
./run_vista.sh <input_dir> \
  --model <output-label> \
  --paper <paper-name>
```

Vista outputs default to `data/output/<model>/<paper>`. The `--model` value labels the output directory; it does not change the configured LLM backend. Useful Vista-only options include `--paper`, `--model`, `--no-unit-test`, and `--debug-iterations`.

## Output

```text
<output_dir>/
├── codebase/          # generated project and reproduce.sh
├── global_memory/     # shared pipeline state and plans
└── local_memory/      # per-agent messages, reasoning, and attempts
```

Vista runs also write `pipeline.log` or `debug.log`. When generated tests are present, `run_vista.sh` runs them and writes `test_results.log` without failing the wrapper on test failures.

Use a new output directory for a clean run. Use `--only-debug` when the generated codebase and saved planning state should be retained.

## Repository layout

```text
src/agents/       pipeline stages
src/memory/       persisted global and per-agent state
src/mcp/          document search, filesystem, LSP, and web-search helpers
src/utils/        model clients and paper-processing utilities
config/           model configuration
slurm/            Vista batch scripts
```

Additional implementation notes are in `architecture.md` and `docs/`.
