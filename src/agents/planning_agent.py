import asyncio
import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.sub_planning_agent import SubPlanningAgent
from src.memory.global_memory import GlobalMemory
from src.utils.gap_matching import attach_target_files_to_context
from src.utils.gap_resolution import is_gap_actionable
from src.utils.llm_client import call_minimax, parse_json
from src.utils.paper_content import format_gaps_for_context, item_to_text

logger = logging.getLogger("planning_agent")


class PlanningAgent(BaseAgent):
    def __init__(self, global_memory: GlobalMemory, output_dir: str = "data/output"):
        super().__init__(name="planning_main", global_memory=global_memory, output_dir=output_dir)
        self.output_dir = output_dir

    async def run(self, max_retries: int = 3) -> dict[str, Any]:
        partitions = self.global_memory.get("partitions", [])
        figure_analysis = self.global_memory.get("figure_analysis", [])

        self.log_reasoning("start", "Beginning planning phase")

        # Step 1: Extract goal, config, and paper analysis (parallel)
        extraction = await self._extract_all(partitions, figure_analysis)
        self.global_memory.set("goal_and_config", extraction.get("goal_and_config", ""))
        self.global_memory.set("paper_analysis", extraction.get("paper_analysis", ""))

        # Step 2: Decide architecture and file structure (retry if no files)
        architecture = await self._retry(
            lambda: self._design_architecture(partitions, figure_analysis, extraction),
            validate=lambda r: bool(r.get("files")),
            step_name="architecture",
            max_retries=max_retries,
        )

        # Step 2.5: Validate architecture and replan if needed
        for replan_attempt in range(2):
            valid, conflicts = self._validate_architecture(architecture)
            if valid:
                break
            self.log_reasoning("replan", f"Architecture conflicts (attempt {replan_attempt + 1}): {conflicts}")
            architecture = await self._replan_architecture(architecture, conflicts, extraction)

        self.global_memory.set("architecture", architecture)
        self.global_memory.set("architecture_md", self._render_architecture_md(architecture))

        # Gap identification happens before architecture design. Attach exact
        # file ownership now, before sub-planning or code generation can consume
        # any verified evidence.
        supplementary = self.global_memory.get("supplementary_context", {})
        if supplementary:
            supplementary = attach_target_files_to_context(supplementary, architecture)
            self.global_memory.set("supplementary_context", supplementary)

        # Step 3: Plan experiments, baselines, and dataset loading in parallel
        experiment_plan, baseline_plan, dataset_plan = await asyncio.gather(
            self._plan_experiments(extraction, architecture),
            self._plan_baselines(extraction, architecture),
            self._plan_dataset_loading(extraction, architecture),
        )
        self.global_memory.set("experiment_plan", experiment_plan)
        self.global_memory.set("baseline_plan", baseline_plan)
        self.global_memory.set("dataset_plan", dataset_plan)

        # Step 4: Group files into modules by directory
        sub_agent_plan = self._group_files_into_modules(architecture)

        # Step 5: Spawn sub-agents with asyncio
        sub_results = await self._run_sub_agents(sub_agent_plan, partitions)
        self.global_memory.set("module_specs", sub_results)

        # Step 6: Review pass for interface conflicts
        review = await self._review_pass(sub_results, architecture)
        self.global_memory.set("review", review)

        self.log_reasoning("done", "Planning complete")
        return {
            "architecture": architecture,
            "module_specs": sub_results,
            "review": review,
        }

    async def _retry(self, fn, validate, step_name: str, max_retries: int = 3):
        for attempt in range(1, max_retries + 1):
            result = await fn()
            if validate(result):
                return result
            logger.warning(f"Step '{step_name}' returned insufficient data (attempt {attempt}/{max_retries}), retrying...")
            self.log_reasoning(f"retry_{step_name}", f"Attempt {attempt} returned empty/invalid, retrying")
        logger.error(f"Step '{step_name}' failed after {max_retries} attempts, proceeding with last result")
        return result

    def _validate_architecture(self, architecture: dict) -> tuple[bool, str]:
        """Validate architecture for orchestrator/module conflicts. Returns (is_valid, conflicts)."""
        files = architecture.get("files", [])
        entry_point = architecture.get("entry_point", "main.py")
        conflicts = []

        # Enforce orchestrator defaults
        for f in files:
            if not isinstance(f, dict):
                continue
            path = f.get("path", "")
            if path in (entry_point, "reproduce.sh"):
                f["orchestrator"] = True
            elif "orchestrator" not in f:
                f["orchestrator"] = False

        # Check for duplicate implements across files
        implements_map: dict[str, list[str]] = {}
        for f in files:
            if not isinstance(f, dict):
                continue
            for impl in f.get("implements", []):
                implements_map.setdefault(impl, []).append(f.get("path", ""))

        for impl, paths in implements_map.items():
            if len(paths) > 1:
                conflicts.append(f"'{impl}' is claimed by multiple files: {paths}")

        # Check orchestrator files don't have implements that overlap with module files
        module_implements = set()
        for f in files:
            if not isinstance(f, dict) or f.get("orchestrator"):
                continue
            for impl in f.get("implements", []):
                module_implements.add(impl)

        for f in files:
            if not isinstance(f, dict) or not f.get("orchestrator"):
                continue
            path = f.get("path", "")
            orch_implements = f.get("implements", [])
            if orch_implements:
                overlaps = [i for i in orch_implements if i in module_implements]
                if overlaps:
                    conflicts.append(
                        f"Orchestrator '{path}' implements {overlaps} which already belong to module files"
                    )
                else:
                    conflicts.append(
                        f"Orchestrator '{path}' has implements={orch_implements} — "
                        f"orchestrators should have implements=[]"
                    )

        if not conflicts:
            return True, ""

        # Auto-fix: strip implements from orchestrator files
        for f in files:
            if isinstance(f, dict) and f.get("orchestrator") and f.get("implements"):
                f["implements"] = []

        self.log_reasoning("validate_autofix", f"Stripped implements from orchestrator files: {conflicts}")
        return True, ""

    async def _replan_architecture(self, architecture: dict, conflicts: str, extraction: dict) -> dict:
        """Ask the LLM to fix architecture conflicts."""
        system = (
            "You are a software architect fixing conflicts in a code architecture plan.\n\n"
            "The architecture below has conflicts. Fix them by:\n"
            "1. Removing ALL `implements` entries from orchestrator files (orchestrator=true)\n"
            "2. Ensuring each equation/algorithm is implemented in exactly ONE module file\n"
            "3. Orchestrator files should ONLY import and wire modules together\n\n"
            "Output ONLY the corrected JSON — same schema, no extra text."
        )

        context = (
            f"## Conflicts\n{conflicts}\n\n"
            f"## Current architecture\n{self._render_architecture_md(architecture)}"
        )

        messages = [{"role": "user", "content": context}]
        self.log_message("user", f"[replan] {context[:300]}...")

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="planning"
        )
        self.log_message("assistant", f"[replan] {response}")

        result = parse_json(response, fallback=None)
        if result and isinstance(result, dict) and result.get("files"):
            return result

        self.log_reasoning("replan_failed", "Replan failed to produce valid JSON, keeping original")
        return architecture

    async def _extract_all(self, partitions: list, figure_analysis: list) -> dict:
        relevant_text = self._get_relevant_text(partitions)

        figures_text = ""
        if figure_analysis:
            figures_text = "\n\nFigure analyses:\n" + "\n".join(
                f"- {f['caption']}: {f['analysis']}" for f in figure_analysis
            )

        paper_content = relevant_text + figures_text

        # Both calls are independent — run in parallel
        goal_config, paper_analysis = await asyncio.gather(
            self._extract_goal_and_config(paper_content),
            self._extract_paper_analysis(paper_content),
        )

        return {
            "goal_and_config": goal_config,
            "paper_analysis": paper_analysis,
        }

    async def _extract_goal_and_config(self, paper_content: str) -> str:
        system = (
            "You are an expert in AI research and reproducing scientific papers.\n\n"
            "Given the sections of a research paper, extract in markdown:\n\n"
            "## Goal\n"
            "What is the core method/system? What exactly needs to be implemented?\n"
            "Be specific about the model, training procedure, and evaluation.\n\n"
            "## Configuration\n"
            "Extract ALL hyperparameters and settings explicitly mentioned in the paper.\n"
            "Be EXHAUSTIVE. Include:\n"
            "- Training: learning rate, batch size, epochs, optimizer, weight decay, scheduler\n"
            "- Model: hidden dims, layers, attention heads, dropout\n"
            "- Data: dataset names, preprocessing, augmentation, splits, sample counts\n"
            "- Evaluation: metrics, eval frequency, number of samples\n"
            "- Loss coefficients: weighting factors, lambda/gamma values\n"
            "- Weight init: zero-init, Xavier, Kaiming — the EXACT scheme\n"
            "- Algorithm-specific: step sizes, noise schedules, clipping\n\n"
            "DO NOT FABRICATE — only use what the paper explicitly provides.\n"
            "Output in markdown."
        )

        messages = [{"role": "user", "content": paper_content}]
        self.log_message("user", f"[goal_config] {paper_content[:200]}...")

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="planning"
        )
        self.log_message("assistant", f"[goal_config] {response}")

        return response

    async def _extract_paper_analysis(self, paper_content: str) -> str:
        system = (
            "You are an expert in AI research. Given a research paper, extract the following in markdown:\n\n"
            "## Equations\n"
            "List EVERY numbered equation with its number, formula, and variables.\n\n"
            "## Algorithms\n"
            "List EVERY algorithm with its name/number and step-by-step description.\n\n"
            "## Figures and Tables to Reproduce\n"
            "List EVERY figure and table showing experimental results, "
            "with what data/models are needed to generate each one.\n\n"
            "## Baselines\n"
            "List EVERY method the paper compares against with:\n"
            "- Name, source (e.g. 'adapts StyleGAN2'), and description.\n\n"
            "## Ablations\n"
            "List EVERY ablation study or variant the paper runs.\n\n"
            "## Datasets\n"
            "List EVERY dataset used in any experiment with name, split, sample count, and usage.\n\n"
            "Output in markdown. Be thorough — missing a baseline or dataset means failed reproduction."
        )

        messages = [{"role": "user", "content": paper_content}]
        self.log_message("user", f"[paper_analysis] {paper_content[:200]}...")

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="planning"
        )
        self.log_message("assistant", f"[paper_analysis] {response}")

        return response

    async def _design_architecture(self, partitions: list, figure_analysis: list, extraction: dict) -> dict:
        system = (
            "You are an expert software architect specializing in ML/AI research code reproduction.\n\n"
            "Given a research paper's method, goal, configuration, equations, algorithms, and figures to reproduce, "
            "design the complete code architecture.\n"
            "The design must be complete and directly implementable. Avoid unnecessary complexity, "
            "and prefer publicly available libraries (PyTorch, numpy, etc.).\n\n"
            "## CRITICAL REQUIREMENTS FOR REPRODUCIBILITY\n"
            "The output codebase will be graded on whether it:\n"
            "a) Implements code for EVERY equation, algorithm, and method in the paper\n"
            "b) Successfully EXECUTES the full training and evaluation pipeline\n"
            "c) Produces all figures/tables from the paper's experiments\n\n"
            "Therefore you MUST include:\n"
            "- A `reproduce.sh` shell script that runs the ENTIRE pipeline end-to-end "
            "(training, evaluation, figure generation) and logs all output to `reproduce.log`.\n"
            "- A `main.py` entry point that orchestrates the full pipeline\n"
            "- Evaluation/sampling scripts that generate results for EVERY figure and table in the paper\n"
            "- Each numbered equation in the paper must have a clear corresponding function or code block\n"
            "- Each algorithm in the paper must have a clear corresponding implementation\n"
            "- A `baselines/` module implementing EVERY baseline method the paper compares against. "
            "Each baseline gets its own file. If a baseline adapts an existing codebase (e.g. StyleGAN2), "
            "implement a wrapper that loads/adapts it.\n"
            "- All ablation variants of the proposed method must be implementable via config flags or "
            "separate scripts — do not omit ablations\n"
            "- A `data/` module that handles downloading and loading EVERY dataset used in the paper\n\n"
            "Output ONLY valid JSON with the following structure:\n"
            "```json\n"
            "{\n"
            '  "files": [\n'
            '    {"path": "model.py", "purpose": "Main model", "dependencies": ["utils/config.py"], "implements": ["Eq. 3", "Algorithm 1"], "orchestrator": false},\n'
            '    {"path": "main.py", "purpose": "Orchestrate full pipeline", "dependencies": ["model.py", "training/trainer.py"], "implements": [], "orchestrator": true}\n'
            "  ],\n"
            '  "modules": ["model", "training", "data", "evaluation", "utils"],\n'
            '  "entry_point": "main.py"\n'
            "}\n"
            "```\n\n"
            "Requirements:\n"
            "1. Every file must have path, purpose, dependencies, implements, and orchestrator (true/false)\n"
            "2. reproduce.sh MUST be included in the files list\n"
            "3. Every baseline from the paper must appear as a file under baselines/\n"
            "4. Every ablation must be runnable via config flag or separate script\n"
            "5. Orchestrator files (entry_point, reproduce.sh, experiment runners) MUST have "
            "`orchestrator: true` and `implements: []`. Their job is importing and wiring modules, "
            "NOT implementing algorithms or equations. All algorithm/equation implementations belong "
            "in dedicated module files only\n"
            "6. No extra text — output ONLY the JSON"
        )

        goal_and_config_text = extraction.get("goal_and_config", "")
        paper_analysis = extraction.get("paper_analysis", "")

        context = (
            f"Goal and config:\n{goal_and_config_text}\n\n"
            f"Paper analysis:\n{paper_analysis}"
        )

        if figure_analysis:
            context += "\n\nFigure analyses:\n" + "\n".join(
                f"- {f['caption']}: {f['analysis']}" for f in figure_analysis
            )

        # Inject unresolved gap warnings from gap agent
        supplementary = self.global_memory.get("supplementary_context", {})
        unresolved = supplementary.get("unresolved_gaps", [])
        if unresolved:
            context += "\n\nWARNING — Unresolved information gaps (be conservative here):\n"
            for g in unresolved:
                status = g.get("resolution_status", "unresolved")
                source = (g.get("source_candidate") or {}).get("title")
                context += (
                    f"- [{g.get('category')} / {status}] "
                    f"{g.get('target_name') or g.get('description', '')}"
                )
                if source:
                    context += f" (source located: {source})"
                context += "\n"

        messages = [{"role": "user", "content": context}]
        self.log_message("user", context)

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="planning"
        )
        self.log_message("assistant", response)

        result = parse_json(response, fallback=None)
        if result and isinstance(result, dict) and result.get("files"):
            return result
        self.log_reasoning("parse_error", f"Failed to parse architecture: {response[:300]}")
        return {"files": [], "modules": [], "entry_point": "main.py"}

    async def _plan_experiments(self, extraction: dict, architecture: dict) -> str:
        goal_and_config = extraction.get("goal_and_config", "")
        paper_analysis = extraction.get("paper_analysis", "")

        system = (
            "You are an expert ML researcher planning the full experimental pipeline for reproducing a paper.\n\n"
            "Given the paper's goal and analysis, produce an experiment plan in markdown that covers "
            "EVERY experiment the paper runs.\n\n"
            "For each experiment include:\n"
            "- Name and description\n"
            "- Models to run (proposed method + all baselines + all ablations)\n"
            "- Datasets used\n"
            "- Metrics to compute\n"
            "- Output files to write (e.g. results/table1.json, results/fig2.png)\n"
            "- How to invoke it (e.g. `python evaluate.py --exp main_results`)\n\n"
            "Output in markdown."
        )

        context = (
            f"Goal and config:\n{goal_and_config}\n\n"
            f"Paper analysis:\n{paper_analysis}\n\n"
            f"Architecture files:\n{json.dumps([f.get('path') for f in architecture.get('files', [])], indent=2)}"
        )

        messages = [{"role": "user", "content": context}]
        self.log_message("user", f"[experiment_plan] {context[:200]}...")
        response = await call_minimax(
            messages=messages, system=system, max_tokens=8192, stage="planning"
        )
        self.log_message("assistant", f"[experiment_plan] {response}")
        return response

    async def _plan_baselines(self, extraction: dict, architecture: dict) -> str:
        goal_and_config = extraction.get("goal_and_config", "")
        paper_analysis = extraction.get("paper_analysis", "")

        system = (
            "You are an expert ML engineer planning the implementation of baseline methods for a paper.\n\n"
            "For each baseline the paper compares against, produce a detailed implementation plan in markdown:\n"
            "- Name and file path (e.g. baselines/tgan.py)\n"
            "- Base codebase (e.g. StyleGAN2, DDPM)\n"
            "- Training loop description\n"
            "- Loss function\n"
            "- Key hyperparameters\n"
            "- Differences from proposed method\n\n"
            "Also plan each ablation variant.\n"
            "Output in markdown."
        )

        context = (
            f"Goal and config:\n{goal_and_config}\n\n"
            f"Paper analysis:\n{paper_analysis}"
        )

        # Inject resolved baseline gaps from gap agent
        supplementary = self.global_memory.get("supplementary_context", {})
        baseline_gaps = [
            g for g in supplementary.get("actionable_gaps", [])
            if g.get("category") == "baseline_method" and is_gap_actionable(g)
        ]
        context += format_gaps_for_context(
            baseline_gaps,
            header="Supplementary context for baselines (from referenced papers):",
            abstract_chars=500,
        )

        messages = [{"role": "user", "content": context}]
        self.log_message("user", f"[baseline_plan] {context[:200]}...")
        response = await call_minimax(
            messages=messages, system=system, max_tokens=8192, stage="planning"
        )
        self.log_message("assistant", f"[baseline_plan] {response}")
        return response

    async def _plan_dataset_loading(self, extraction: dict, architecture: dict) -> str:
        goal_and_config = extraction.get("goal_and_config", "")
        paper_analysis = extraction.get("paper_analysis", "")

        system = (
            "You are an expert ML engineer planning dataset download and loading for a paper reproduction.\n\n"
            "For each dataset the paper uses, produce a plan in markdown:\n"
            "- Name and source (HuggingFace ID, torchvision class, or download URL)\n"
            "- Target directory\n"
            "- Number of samples and splits needed\n"
            "- Preprocessing steps\n"
            "- Authentication requirements\n\n"
            "Prefer HuggingFace > torchvision > direct URL.\n"
            "Output in markdown."
        )

        context = (
            f"Goal and config:\n{goal_and_config}\n\n"
            f"Paper analysis:\n{paper_analysis}"
        )

        # Inject dataset details from gap agent's registry
        supplementary = self.global_memory.get("supplementary_context", {})
        dataset_details = supplementary.get("dataset_details", {})
        if dataset_details:
            context += "\n\nDataset details from known registries:\n"
            for name, info in dataset_details.items():
                context += f"\n- **{name}**: source={info.get('source')}"
                if info.get("train_size"):
                    context += f", train_size={info['train_size']}"
                if info.get("standard_transforms"):
                    context += f"\n  transforms: {info['standard_transforms']}"
                context += "\n"

        messages = [{"role": "user", "content": context}]
        self.log_message("user", f"[dataset_plan] {context[:200]}...")
        response = await call_minimax(
            messages=messages, system=system, max_tokens=4096, stage="planning"
        )
        self.log_message("assistant", f"[dataset_plan] {response}")
        return response

    def _group_files_into_modules(self, architecture: dict) -> list[dict]:
        """Group architecture files into modules by directory. No LLM call needed."""
        files = architecture.get("files", [])
        modules: dict[str, list[str]] = {}
        for f in files:
            path = f.get("path", "") if isinstance(f, dict) else f
            # Group by top-level directory, or "root" for top-level files
            parts = path.split("/")
            module = parts[0] if len(parts) > 1 else "root"
            modules.setdefault(module, []).append(path)

        return [{"module": mod, "files": fls} for mod, fls in modules.items()]

    @staticmethod
    def _render_architecture_md(architecture: dict) -> str:
        """Render architecture JSON as human/LLM-readable markdown."""
        lines = [f"# Architecture\n"]
        lines.append(f"**Entry point:** `{architecture.get('entry_point', 'main.py')}`\n")

        modules = architecture.get("modules", [])
        if modules:
            lines.append(f"**Modules:** {', '.join(modules)}\n")

        files = architecture.get("files", [])
        if files:
            lines.append("## Files\n")
            for f in files:
                if not isinstance(f, dict):
                    continue
                path = f.get("path", "")
                purpose = f.get("purpose", "")
                orch = " (orchestrator)" if f.get("orchestrator") else ""
                lines.append(f"### `{path}`{orch}")
                lines.append(f"**Purpose:** {purpose}")
                deps = f.get("dependencies", [])
                if deps:
                    lines.append(f"**Dependencies:** {', '.join(f'`{d}`' for d in deps)}")
                implements = f.get("implements", [])
                if implements:
                    lines.append(f"**Implements:** {', '.join(implements)}")
                lines.append("")

        return "\n".join(lines)

    async def _run_sub_agents(self, sub_agent_plan: list[dict], partitions: list) -> dict[str, Any]:
        # Run all sub-agents in parallel — no dependency ordering needed
        tasks = []
        for plan in sub_agent_plan:
            agent = SubPlanningAgent(
                module_name=plan["module"],
                files=plan["files"],
                relevant_partitions=partitions,
                global_memory=self.global_memory,
                output_dir=self.output_dir,
            )
            tasks.append((plan["module"], agent.run()))

        batch_results = await asyncio.gather(*[t[1] for t in tasks])

        results = {}
        for (module, _), result in zip(tasks, batch_results):
            results[module] = result

        self.log_reasoning("batch_complete", f"Completed: {[t[0] for t in tasks]}")
        return results

    async def _review_pass(self, module_specs: dict, architecture: dict) -> str:
        system = (
            "You are a senior code reviewer performing a final consistency check before implementation.\n\n"
            "Given module specifications from multiple independent planning agents, verify that all "
            "interfaces are consistent and compatible. Check for:\n\n"
            "1. Signature mismatches between modules\n"
            "2. Return type conflicts\n"
            "3. Missing dependencies\n"
            "4. Inconsistent data structures\n"
            "5. Circular dependencies\n"
            "6. Config conflicts\n\n"
            "For each conflict found, provide a specific resolution.\n"
            "Output in markdown."
        )

        # Build text summary of module specs
        specs_text = ""
        for module, spec in module_specs.items():
            if isinstance(spec, str):
                specs_text += f"\n### Module: {module}\n{spec}\n"
            elif isinstance(spec, dict) and spec.get("spec_text"):
                specs_text += f"\n### Module: {module}\n{spec['spec_text']}\n"

        context = (
            f"Architecture:\n{self._render_architecture_md(architecture)}\n\n"
            f"Module specs:\n{specs_text}"
        )

        messages = [{"role": "user", "content": context}]
        self.log_message("user", context)

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="planning"
        )
        self.log_message("assistant", response)

        return response

    def _get_relevant_text(self, partitions: list) -> str:
        parts = []
        for section in partitions:
            # Skip references — planning doesn't need the bibliography.
            # Prefer the section_role tag; fall back to title-match for
            # partitions from runs before the tag existed.
            role = section.get("section_role")
            if role == "references":
                continue
            if role is None and any(
                kw in section.get("title", "").lower()
                for kw in ("reference", "bibliography", "cited")
            ):
                continue
            parts.append(f"## {section['title']}")
            for item in section["items"]:
                text = item_to_text(item, include_table_html=True)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
