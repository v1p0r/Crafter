import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.memory.global_memory import GlobalMemory
from src.utils.gap_matching import gap_relevant_to_target
from src.utils.gap_resolution import is_gap_actionable
from src.utils.llm_client import call_minimax
from src.utils.paper_content import format_gaps_for_context, item_to_text

logger = logging.getLogger("sub_planning_agent")


class SubPlanningAgent(BaseAgent):
    def __init__(
        self,
        module_name: str,
        files: list[str],
        relevant_partitions: list[dict],
        global_memory: GlobalMemory,
        output_dir: str = "data/output",
    ):
        super().__init__(
            name=f"planning_sub_{module_name}",
            global_memory=global_memory,
            output_dir=output_dir,
        )
        self.module_name = module_name
        self.files = files
        self.relevant_partitions = relevant_partitions

    async def run(self) -> dict[str, Any]:
        self.log_reasoning("start", f"Planning module: {self.module_name}, files: {self.files}")

        architecture = self.global_memory.get("architecture", {})
        goal_and_config = self.global_memory.get("goal_and_config", "")
        figure_analysis = self.global_memory.get("figure_analysis", [])
        paper_analysis = self.global_memory.get("paper_analysis", "")
        experiment_plan = self.global_memory.get("experiment_plan", "")
        baseline_plan = self.global_memory.get("baseline_plan", "")
        dataset_plan = self.global_memory.get("dataset_plan", "")

        paper_context = self._build_paper_context()

        system = (
            f"You are an expert researcher and software engineer designing the '{self.module_name}' module.\n\n"
            "Your task is to produce a COMPLETE and DETAILED implementation specification in markdown "
            "that can be used directly for code generation.\n\n"
            "## Requirements\n"
            "1. STRICTLY align with the paper's methodology\n"
            "2. FOLLOW the overall architecture's interfaces — do not change the design\n"
            "3. Reference settings from config — do NOT fabricate values\n"
            "4. For equations: specify the EXACT formula and variable mappings\n"
            "5. For weight init: use the EXACT scheme from the paper\n"
            "6. For hyperparameters: use EXACT values from config\n"
            "7. For loss functions: write the full formula\n\n"
            "## Output Format (Markdown)\n"
            "For each file in this module, write:\n\n"
            "### `file_path.py`\n"
            "**Imports**: list imports\n"
            "**Classes**:\n"
            "- `ClassName(base_class)`: description\n"
            "  - `__init__(args)`: what to initialize, weight init scheme\n"
            "  - `method(args) -> return_type`: step-by-step logic, tensor shapes, equations\n"
            "**Functions**:\n"
            "- `func(args) -> return_type`: description\n\n"
            "Include tensor shapes (e.g. `[B, C, H, W]`), exact formulas, and default values.\n"
            "Be detailed enough that a developer can implement without reading the paper.\n\n"
            "## Module-specific rules\n"
            "If this module is 'baselines': spec EVERY baseline the paper compares against.\n"
            "If this module is 'data': spec download for EVERY dataset, prefer HuggingFace/torchvision.\n"
            "If a file is marked as `orchestrator: true` in the architecture, its spec should ONLY cover:\n"
            "- Which modules to import and from where\n"
            "- The sequence of operations (data loading -> training -> evaluation -> output)\n"
            "- Command-line argument parsing if applicable\n"
            "- It should NOT specify algorithm implementations, loss formulas, or model architectures — "
            "those belong in the module files. The orchestrator only imports and calls them."
        )

        context = (
            f"Goal and config:\n{goal_and_config}\n\n"
            f"Overall architecture:\n{self.global_memory.get('architecture_md', '')}\n\n"
            f"My assigned files: {json.dumps(self.files)}\n\n"
            f"Paper content relevant to this module:\n{paper_context}"
        )

        if paper_analysis:
            context += f"\n\nPaper analysis:\n{paper_analysis}"
        if experiment_plan:
            context += f"\n\nExperiment plan:\n{experiment_plan}"
        if baseline_plan:
            context += f"\n\nBaseline plan:\n{baseline_plan}"
        if dataset_plan:
            context += f"\n\nDataset plan:\n{dataset_plan}"

        if figure_analysis:
            context += "\n\nFigure analyses:\n" + "\n".join(
                f"- {f['caption']}: {f['analysis']}" for f in figure_analysis
            )

        # Inject resolved gaps relevant to this module
        supplementary = self.global_memory.get("supplementary_context", {})
        requirement_ids = tuple(
            requirement_id
            for file_spec in architecture.get("files", [])
            if isinstance(file_spec, dict) and file_spec.get("path") in self.files
            for requirement_id in file_spec.get("implements", [])
        )
        relevant_gaps = [
            g for g in supplementary.get("actionable_gaps", [])
            if is_gap_actionable(g)
            and gap_relevant_to_target(
                g,
                self.module_name,
                requirement_ids=requirement_ids,
                target_files=tuple(self.files),
            )
        ]
        context += format_gaps_for_context(relevant_gaps)

        messages = [{"role": "user", "content": context}]
        self.log_message("user", context)

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="planning"
        )
        self.log_message("assistant", response)

        # Store as markdown spec
        spec = {"module_name": self.module_name, "spec_text": response}

        self.log_reasoning("done", f"Completed spec for {self.module_name}")
        return spec

    def _build_paper_context(self) -> str:
        parts = []
        for section in self.relevant_partitions:
            parts.append(f"## {section['title']}")
            for item in section["items"]:
                text = item_to_text(item, include_table_html=True)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
