# Changelog

## 2026-08-02

Debug agent overhaul, driven by log analysis of all MiniMax (`data/output/MiniMax-M2.7/*/debug.log`) and Claude sonnet-4.6 (`fabrication_audit/fresh_run/extract/*/pipeline_debug_*.log`) runs.

### Added
- **Preflight stage**: before the run-crash loop, every file is compile-checked and local `from x import y` names are verified statically (pure AST, nothing executed); all findings are batch-fixed in one LLM call. Kills the one-ImportError-per-2-minute-run pattern (mech run burned 6+ iterations this way).
- **Debug journal**: per-attempt record (error signature → root cause → fix → outcome: fixed / error-changed / unchanged / reverted / rejected / no-fix) plus accumulated codebase notes the model appends via the new `notes_for_next_attempt` field. Injected into every investigation and regen — replaces the misleading 3-entry fix history (which labeled the pre-fix error "Error after fix" and showed never-applied fixes as failed approaches).
- **Root-cause discipline**: `submit_fix` now requires `root_cause` and requests `evidence`; system prompt requires hypothesis verification via reads when the cause isn't in the crash frame; module spec excerpt included when pipeline memory is available.
- **In-session fix application with feedback**: fixes are applied during the tool conversation; a fix that fails (syntax error with line detail, out-of-range edit, bad path) goes back as an `is_error` tool result and the model resubmits (≤2 retries). Previously invalid fixes were silently discarded and a whole iteration burned (4+ occurrences across runs).
- **Malformed tool-arg coercion**: `edits`/`file_rewrites` sent as strings are `json.loads`-decoded or rejected with one clear message — never iterated char-by-char (mech run produced 28,610 per-character warnings and lost its fixes to this).
- **Regression protection**: files are snapshotted (`debug_snapshots/iter_N/`) before every mutation; a fix that turns a runtime failure into an import-level failure (or silences all output) is auto-reverted and journaled; the best-progressing state is kept in `debug_snapshots/best/` and restored at the end if the final state scored worse. Anti-degradation rules (no stubs, no try/except-pass, no deleting failing paths) added to fix and regen prompts; regens that drop public names get that fed back into the retry.
- **Bare-codebase mode**: entry-point auto-detection (architecture → `reproduce.sh` → `main.py`/`run.py`/`train.py`/`run_experiments.py`), graceful degradation without `global_memory` (spec/architecture sections omitted; `files_to_create` codegen skipped). `run_pipeline` gains `codebase_dir_override`/`entry_point`/`run_timeout`/`strict_timeout` params (no new CLI flags).
- **Timeout policy**: run timeout configurable per agent (`run_timeout`, default 120s unchanged); on timeout the partial output is captured and classified — progress markers → `verified: true`, output without markers → `verified: false`, zero output → `verified: false, silent: true` (or a debuggable hang failure with `strict_timeout=True`). Success results now carry the evidence instead of a bare "timed out".
- **`src/utils/requirements.py`**: `PIP_NAME_MAP` (hydra→hydra-core, cv2→opencv-python, PIL→Pillow, sklearn→scikit-learn, googleapiclient→google-api-python-client, …) and `ensure_requirements()` — scans imports (AST with regex fallback), filters stdlib/local modules, preserves existing pinned entries. Called at debug start and after codegen, so every codebase ships the previously-missing-but-mandatory requirements.txt.
- **Prompt caching for the tool loop**: `call_minimax_with_tools(cache=True)` sets `cache_control` breakpoints (system + conversation tail, stale markers stripped) on backends with cache pricing (claude). Usage tracker already accounted cache tokens.
- `tests/test_debug_agent.py`: 40 unit tests over the pure helpers (coercion, apply-fix problems, line edits, error normalization, resource detection, auto-install, entry detection, timeout classification, requirements, snapshots/regression, static checks, journal).

### Changed
- **Regeneration trigger**: fires when the same *normalized error* (digits/addresses stripped) persists ≥3 runs — not when the same *file* crashes 3× (which fired mid-progress: all-in-one regenerated main.py at iteration 3 after two *different* bugs were fixed there, and regenerated files for missing-dataset errors it can never fix). Resource errors (missing dataset/checkpoint/credentials) never trigger regen and instead inject synthetic-fallback guidance. Errors without a codebase traceback are no longer attributed to the entry point.
- All debug-agent subprocesses (`_run_code`, smoke test, pip) use `sys.executable` instead of bare `python`, so the generated code runs under the same interpreter as the pipeline (identical on the cluster venv; fixes environments where `python` is only a shell alias). Tool-call `input` arriving as a JSON string (MiniMax anthropic-shim quirk) is decoded instead of dropped.
- **Auto-install**: uses `<sys.executable> -m pip` with the pip-name map, verifies the module imports afterward, never reinstalls the same module twice, records unresolvable dependencies for the LLM, appends verified installs to requirements.txt, and conservatively handles "<X> is required" messages via an allowlist. Previously `pip install hydra` (wrong package) re-ran every iteration — ~8 wasted iterations in the all-in-one Claude run.
- `call_minimax_with_tools` retries transient API errors (connection/429/5xx) with 2/5/15/45s backoff; each debug iteration is also exception-contained, so one API failure no longer kills the whole stage.
- Investigation loop: `stop=max_tokens` with no tool call retries instead of aborting the iteration (all-in-one iter 10 lost ~5 min to this); the budget-exhausted notice merges into the tool-result message (no consecutive `user` turns); empty assistant text gets a placeholder; nudges no longer consume the 5-call investigation budget; `search_codebase` merges memory and disk results.

## 2026-04-28

### Added
- `call_vlm` now retries on transient errors (`InternalServerError` 5xx, `RateLimitError` 429, `APIConnectionError`, `APITimeoutError`) with exponential backoff: up to 5 attempts, sleeping 2/5/15/45s ± 25% jitter between them. Non-retryable errors (bad request, auth, etc.) still fail-fast and return `""`. Motivated by the paperbench batch run where Gemini 503s caused ~70 silent figure-analysis failures across 10 papers.
- `scripts/vlm_retest.py`: re-runs only the VLM stage on the 10 affected papers using the new retry logic. Outputs to `data/output/MiniMax-M2.7/vlm-retest/<paper>/global_memory/figure_analysis.json`.
- `run_paperbench_rerun_vlm_failed.sh`: full pipeline re-run for the 10 affected papers, seeding `figure_analysis.json` from the vlm-retest cache so VLM is skipped (cached) and the rest of the pipeline (gap → planning → codegen) runs against the better figure analysis. Outputs to `data/output/MiniMax-M2.7/skip-debug-v2/<paper>/`.

## 2026-04-26

### Added
- `run_paperbench_skipdebug.sh`: batch runner for the 23 paperbench papers in `/Users/esther_lh/GitHub/paperbench_data/data/output_v2`. VLM on, gap analysis on, debug skipped. Outputs to `data/output/MiniMax-M2.7/skip-debug/<paper>/` with per-paper `run.log` and a `_batch.log`. Skips papers whose output dir already exists; sequential.

## 2026-04-13 (later)

### Added
- **Section role tagging** in partition agent: each section now has `section_role` in (`body`, `references`, `appendix`, `preamble`). Enables role-based filtering without re-scanning titles.
- **Tier 0.5 appendix resolution**: Gaps whose answer is in the paper's own appendix are now resolved from within the paper. The LLM emits an `internal_ref` field (e.g. `"Appendix A"`) in the gap identification call; `_resolve_from_appendix` matches it to the tagged appendix section and returns its text as the resolution. Downstream agents see the appendix text via `format_gaps_for_context` (rendered as `From appendix: ...`).

### Fixed
- **Bibliography no longer duplicated in gap agent LLM input**: `_get_relevant_text` now excludes references (which are sent separately under `_get_references_text`). Saves ~5K input tokens per run.
- **Planning agent stops sending references to its LLM**: `_get_relevant_text` filters out `section_role == "references"`.

### Changed
- `partition_agent.py` adds `_tag_section_roles()` (new method).
- `gap_agent.py`: `_resolve_gap` now takes `partitions`, adds Tier 0.5 path, new `_resolve_from_appendix` helper.
- `paper_content.format_gaps_for_context` renders `resolution.internal_text` for appendix-based resolutions.

## 2026-04-13

### Added
- **Reference-grounded gap resolution**: The gap agent's LLM call now parses the bibliography into structured entries (`ref_id`, exact title, authors, year, arxiv_id) in the same call. Each gap is linked to a specific reference entry, so S2 searches use titles copied verbatim from the bibliography instead of LLM guesses.
- **S2 match verification**: Title similarity check (`SequenceMatcher` + significant-word overlap) rejects false-positive S2 results. Threshold tuned to 0.55 with ≥40% word overlap fallback.
- **Fuzzy S2 search**: Tries full title first, then short name (before colon/dash) if full title fails verification.
- **OCR ligature cleanup**: Fixes PDF parsing artifacts (`E±cient` → `Efficient`, `ne-tuning` → `fine-tuning`) in reference titles before S2 lookup.
- **Per-stage token/cost logging**: `main.py` now prints cumulative usage after each pipeline stage and a final summary (`tokens in=X out=Y | calls=N | cost=$Z`).
- **Shared `src/utils/paper_content.py`**: Consolidates `item_to_text`, `extract_inline`, and `format_gaps_for_context` previously duplicated across three agents.
- **Shared `gap_relevant_to_target` + `GAP_TARGET_KEYWORDS`** in `gap_agent.py`: Centralized category→keyword mapping, replacing duplicated `_gap_relevant_to_*` methods in sub_planning and codegen agents.

### Fixed
- **Reference list extraction**: `item_to_text()` now handles `type: list` items (MinerU's `reference_list` format), which were previously invisible to all agents. Without this, the LLM was hallucinating titles since the references section appeared empty. Latent bugs in `planning_agent` and `sub_planning_agent` (missing `list`/`title`/`image` handling) are also fixed.

### Changed
- Removed duplicated `_item_to_text`/`_extract_inline` from `gap_agent`, `planning_agent`, `sub_planning_agent` — all three now import from `paper_content`.
- Removed duplicated supplementary-context formatting blocks from `sub_planning_agent`, `codegen_agent`, `planning_agent._plan_baselines` — all use `format_gaps_for_context`.
- Consolidated dataset-scanning logic in `gap_agent` into `_scan_text_for_datasets` helper.

## 2026-04-11

### Added
- **Gap Analysis Agent** (`src/agents/gap_agent.py`): New Stage 2.5 that identifies information gaps in the target paper (referenced architectures, baselines, datasets, missing hyperparameters) via one LLM call, then resolves each programmatically through Semantic Scholar API and a static dataset registry.
- **Semantic Scholar client** (`src/utils/semantic_scholar_client.py`): Async client with rate limiting. Fetches paper metadata by ArXiv ID, DOI, or title search.
- **Dataset registry** (`src/utils/dataset_registry.py`): 29 common ML datasets (ImageNet, CIFAR, SST-2, MMLU, etc.) with sources, sizes, and standard transforms.
- `--skip-gaps` CLI flag to bypass gap analysis.

### Changed
- `planning_agent.py`: `_design_architecture()` now shows unresolved gap warnings; `_plan_baselines()` injects resolved baseline abstracts/TLDRs; `_plan_dataset_loading()` injects dataset registry details.
- `sub_planning_agent.py`: Filters and injects resolved gaps relevant to each module.
- `codegen_agent.py`: Injects resolved gaps relevant to each file being generated.

## 2026-04-08
- Added `run_local.sh` — batch local runner using conda env `ptc`, modeled after `ptc_agent/run_local.sh`
  - Processes all papers in `data/parsed/` or a single paper via `--paper`
  - Supports `--vlm`, `--debug`, `--only-debug`, `--output` flags
  - Skips papers with existing output
  - Prints summary of succeeded/failed papers
- Swapped VLM backend from DashScope/qwen-vl-max to Gemini (`gemini-2.5-flash`)
  - Updated `llm_client.py`: `get_dashscope_client` -> `get_gemini_client`, uses `GEMINI_API_KEY`
  - Updated `config/settings.yaml`, `run.sh`, `run_local.sh` accordingly
- Fixed `_extract_text` in `llm_client.py` — was crashing on `ThinkingBlock` (no `.text` attr); now filters by `type == "text"`
