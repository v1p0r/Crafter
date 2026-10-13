import ast
import json
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.memory.global_memory import GlobalMemory
from src.utils.gap_matching import gap_relevant_to_target
from src.utils.gap_resolution import is_gap_actionable
from src.utils.llm_client import call_minimax
from src.utils.paper_content import format_gaps_for_context


class CodegenAgent(BaseAgent):
    def __init__(
        self,
        file_path: str,
        global_memory: GlobalMemory,
        output_dir: str = "data/output",
        codebase_dir: str = "data/output/codebase",
        is_orchestrator: bool = False,
    ):
        safe_name = file_path.replace("/", "_").replace(".", "_")
        super().__init__(
            name=f"codegen_{safe_name}",
            global_memory=global_memory,
            output_dir=output_dir,
        )
        self.file_path = file_path
        self.codebase_dir = Path(codebase_dir)
        self.is_orchestrator = is_orchestrator

    async def run(self) -> dict[str, Any]:
        self.log_reasoning("start", f"Generating code for {self.file_path} (orchestrator={self.is_orchestrator})")

        module_specs = self.global_memory.get("module_specs", {})
        architecture = self.global_memory.get("architecture", {})
        goal_and_config = self.global_memory.get("goal_and_config", "")
        review = self.global_memory.get("review", "")
        generated_code = self.global_memory.get("generated_code", {})

        # Find the module spec for this file
        module_spec_text = self._find_module_spec(module_specs)

        if self.is_orchestrator:
            system = self._build_orchestrator_system_prompt()
        else:
            system = self._build_module_system_prompt()

        # Build context with previously generated files
        prev_code_text = ""
        if generated_code:
            if self.is_orchestrator:
                # Orchestrators receive ALL generated module files so they can
                # correctly import and call into every module
                relevant_code = dict(generated_code)
            else:
                # Module files only receive their direct dependencies
                relevant_paths = self._get_relevant_prev_files(architecture)
                relevant_code = {p: generated_code[p] for p in relevant_paths if p in generated_code}
            if relevant_code:
                prev_code_text = "\n\nPreviously generated files:\n"
                for path, code in relevant_code.items():
                    prev_code_text += f"\n--- {path} ---\n{code}\n"

        # paper_analysis and review are markdown strings
        paper_analysis = self.global_memory.get("paper_analysis", "")

        context = (
            f"File to generate: {self.file_path}\n\n"
            f"Module spec:\n{module_spec_text}\n\n"
            f"Overall architecture:\n{self.global_memory.get('architecture_md', '')}\n\n"
            f"Goal and config:\n{goal_and_config}"
        )

        if paper_analysis:
            context += f"\n\nPaper analysis:\n{paper_analysis}"

        if isinstance(review, str) and review:
            context += f"\n\nReview notes:\n{review}"

        if self.file_path == "reproduce.sh":
            context += self._build_reproduce_context()

        # Inject resolved gaps relevant to this file.
        # Target is file_path + file purpose (from architecture) so both the
        # path-keyword match (e.g. "baseline" in "baselines/lora.py") AND the
        # semantic match (e.g. "attention" in purpose "attention-based encoder") work.
        supplementary = self.global_memory.get("supplementary_context", {})
        file_entry = {}
        for f in architecture.get("files", []):
            if isinstance(f, dict) and f.get("path") == self.file_path:
                file_entry = f
                break
        file_purpose = file_entry.get("purpose", "") or ""
        target = f"{self.file_path} {file_purpose}"
        relevant_gaps = [
            g for g in supplementary.get("actionable_gaps", [])
            if is_gap_actionable(g)
            and gap_relevant_to_target(
                g,
                target,
                requirement_ids=tuple(file_entry.get("implements", [])),
                target_files=(self.file_path,),
            )
        ]
        context += format_gaps_for_context(relevant_gaps)

        context += prev_code_text

        messages = [{"role": "user", "content": context}]
        self.log_message("user", context)

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="codegen"
        )
        self.log_message("assistant", response)

        # Clean response — strip markdown fences if present
        code = self._clean_code(response)

        # Cheap syntax self-correction: if Python doesn't parse, retry once with the error.
        # Saves 1–2 debug iterations per paper on trivial bugs (missing colons, bad lambdas).
        if self.file_path.endswith(".py"):
            try:
                ast.parse(code)
            except SyntaxError as e:
                self.log_reasoning("syntax_retry", f"{self.file_path}: {e}; retrying once")
                retry_messages = messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Your previous output had a SyntaxError: {e}. Output the full corrected file again."},
                ]
                response = await call_minimax(
                    messages=retry_messages,
                    system=system,
                    max_tokens=16384,
                    stage="codegen",
                )
                self.log_message("assistant", f"[retry] {response}")
                code = self._clean_code(response)

        # Post-generation fix for reproduce.sh
        if self.file_path == "reproduce.sh":
            code = self._fix_reproduce_sh(code)

        # Write to codebase
        out_path = self.codebase_dir / self.file_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code)

        # Store in global memory for subsequent agents
        if generated_code is None:
            generated_code = {}
        generated_code[self.file_path] = code
        self.global_memory.set("generated_code", generated_code)

        self.log_reasoning("done", f"Generated {len(code)} chars for {self.file_path}")
        return {"file_path": self.file_path, "code": code}

    def _build_module_system_prompt(self) -> str:
        return (
            "You are an expert researcher and software engineer with a deep understanding of "
            "experimental design and reproducibility in scientific research.\n\n"
            "You will receive a file specification, the overall architecture, configuration, and any "
            "previously generated code files. Your task is to write code that reproduces the paper's method.\n\n"
            "The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines.\n"
            "The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics.\n\n"
            f"You MUST write only the file: {self.file_path}\n\n"
            "## Rules (FOLLOW ALL OF THESE)\n"
            "1. **Only ONE file**: Do your best to implement THIS ONLY ONE FILE\n"
            "2. **COMPLETE CODE**: Your code will be part of the entire project, so implement complete, "
            "reliable, reusable code. No TODOs, no placeholders, no '...', no 'pass' in non-abstract methods\n"
            "3. **Set default value**: If there is any setting, ALWAYS SET A DEFAULT VALUE, "
            "ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE\n"
            "4. **Follow design**: YOU MUST FOLLOW the data structures and interfaces from the spec. "
            "DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in the spec\n"
            "5. **CAREFULLY CHECK** that you don't miss any necessary class/function in this file\n"
            "6. **Import first**: Before using an external variable/module, make sure you import it first. "
            "AVOID circular imports\n"
            "7. **Write out EVERY CODE DETAIL**: Don't leave any TODO or unimplemented section\n"
            "8. **REFER TO CONFIGURATION**: You must use configuration from the provided config. "
            "DO NOT FABRICATE any configuration values — only use what is explicitly provided\n"
            "9. **Type hints**: Include type hints for all function arguments and return values\n"
            "10. **Previously generated files**: Reference the already generated files for correct imports "
            "and interface compatibility\n"
            "11. **EXACT paper values**: Weight initialization, hyperparameters, and loss coefficients "
            "MUST match the paper EXACTLY. If the spec says zero-init, use `nn.init.zeros_()` — "
            "do NOT substitute Xavier or Kaiming. If gamma=5, hardcode 5 as the default, not 1.0\n"
            "12. **Equation fidelity**: If this file implements a paper equation, the code MUST compute "
            "every term in that equation. Do not simplify or omit terms\n"
            "13. **DO NOT DUPLICATE**: If a class or function already exists in a previously generated file, "
            "IMPORT it — do NOT redefine it\n\n"
            "## Output Format\n"
            "Write the code with triple quotes. Output format:\n"
            f"```python\n"
            f"## {self.file_path}\n"
            "...\n"
            "```"
        )

    def _build_orchestrator_system_prompt(self) -> str:
        reproduce_rules = ""
        if self.file_path == "reproduce.sh":
            reproduce_rules = (
                "\n## reproduce.sh rules\n"
                "- The script MUST start with EXACTLY these lines (no custom logging, no argument parsing):\n"
                "  ```\n"
                "  #!/bin/bash\n"
                "  set -e\n"
                "  exec > >(tee -a reproduce.log) 2>&1\n"
                "  python -m pip install -r requirements.txt\n"
                "  ```\n"
                "- ALL output MUST go to `reproduce.log` via the exec redirect above\n"
                "- Do NOT create custom log files, log functions, or argument parsers\n"
                "- Do NOT use bare `pip` — always use `python -m pip`\n"
                "- Run EVERY experiment: every model variant, every dataset, every baseline\n"
                "- Loop over ALL source→target pairs, not just one default\n"
                "- Save generated images to `results/images/`, metrics to `results/metrics.json`\n"
                "- The script MUST write files to disk — a no-op script scores 0\n"
                "- Must be runnable with `bash reproduce.sh` from the codebase dir, no arguments\n"
                "- Print progress: `echo '=== Running experiment X ==='`\n"
                "- Keep it SIMPLE — a flat sequence of commands, not functions/stages/parsers\n"
            )

        return (
            "You are writing an ORCHESTRATOR file. Your ONLY job is to import from existing modules "
            "and wire them together.\n\n"
            f"You MUST write only the file: {self.file_path}\n\n"
            "## CRITICAL RULES\n"
            "1. **IMPORT, DON'T IMPLEMENT**: You MUST NOT define any model, trainer, dataset, loss, "
            "evaluation, or algorithm logic. Every class and function you need already exists in the "
            "generated module files provided below. IMPORT them.\n"
            "2. **Verify interfaces**: For every function call you make, verify the import path and "
            "function signature against the actual source code provided. Do not guess at interfaces.\n"
            "3. **No duplication**: If you find yourself writing >20 lines of logic that isn't "
            "argument parsing or control flow, STOP — you're reimplementing something that belongs "
            "in a module file. Import it instead.\n"
            "4. **COMPLETE CODE**: No TODOs, no placeholders, no 'pass' in non-abstract methods\n"
            "5. **Set default values**: ALWAYS SET DEFAULT VALUES for settings, "
            "ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE\n"
            "6. **Follow design**: Follow the architecture spec's interfaces exactly\n"
            "7. **REFER TO CONFIGURATION**: Use config from the provided config. "
            "DO NOT FABRICATE any configuration values\n"
            "8. **Type hints**: Include type hints for all function arguments and return values\n"
            "9. **Previously generated files**: Reference the provided source code for correct imports. "
            "Use the EXACT class names, function names, and argument signatures from the source.\n"
            + reproduce_rules +
            "\n## Output Format\n"
            "Write the code with triple quotes. Output format:\n"
            f"```python\n"
            f"## {self.file_path}\n"
            "...\n"
            "```"
        )

    def _get_relevant_prev_files(self, architecture: dict) -> list[str]:
        """Return only previously generated files this file depends on."""
        relevant = set()

        # From architecture dependency graph
        arch_files = architecture.get("files", [])
        for f in arch_files:
            if isinstance(f, dict) and f.get("path") == self.file_path:
                for dep in f.get("dependencies", []):
                    relevant.add(dep)

        return list(relevant)

    def _build_reproduce_context(self) -> str:
        architecture = self.global_memory.get("architecture", {})
        goal_and_config = self.global_memory.get("goal_and_config", "")
        paper_analysis = self.global_memory.get("paper_analysis", "")
        generated_code = self.global_memory.get("generated_code", {})

        # Extract entry point and all generated file paths
        entry_point = architecture.get("entry_point", "main.py")
        all_files = list(generated_code.keys())

        context = "\n\n## reproduce.sh Requirements\n"
        context += (
            "## MANDATORY STRUCTURE — the script MUST start with exactly:\n"
            "```bash\n"
            "#!/bin/bash\n"
            "set -e\n"
            "exec > >(tee -a reproduce.log) 2>&1\n"
            "python -m pip install -r requirements.txt\n"
            "```\n\n"
            "Then run ALL experiments as a FLAT list of commands. NO functions, NO argument parsing, "
            "NO custom log files. The `exec` redirect sends everything to `reproduce.log` automatically.\n\n"
            "Use `python -m pip` (not bare `pip`). Only include REAL packages in requirements.txt — "
            "verify every package name is a real PyPI package.\n\n"
        )

        context += f"Entry point: {entry_point}\n"
        context += f"All generated files: {json.dumps(all_files)}\n"

        if goal_and_config:
            context += f"\nGoal and config:\n{goal_and_config}\n"

        if paper_analysis:
            context += f"\nPaper analysis:\n{paper_analysis}\n"

        # Pass dataset plan so reproduce.sh knows to call setup_data.py first
        dataset_plan = self.global_memory.get("dataset_plan", "")
        if dataset_plan:
            context += f"\n\nDataset loading plan:\n{dataset_plan}"

        context += (
            "\n\n## Critical reproduce.sh structure:\n"
            "1. Install dependencies: `python -m pip install -r requirements.txt` (the exec redirect handles logging)\n"
            "2. Download datasets: `python setup_data.py` "
            "(this downloads real data — do NOT skip this step)\n"
            "3. Run all experiments in order\n"
            "4. Save all outputs to disk\n\n"
            "The debug loop uses dummy data so the code runs without crashing. "
            "reproduce.sh must download real data via setup_data.py before running experiments. "
            "A script that exits without writing output files will score 0.\n"
        )

        return context

    def _find_module_spec(self, module_specs: dict) -> str:
        """Find the module spec markdown text that covers this file."""
        # Check which module this file belongs to by matching file path
        for module_name, spec in module_specs.items():
            if isinstance(spec, dict) and spec.get("spec_text"):
                if self.file_path in spec["spec_text"]:
                    return spec["spec_text"]
        # Fallback: return all specs concatenated
        parts = []
        for module_name, spec in module_specs.items():
            if isinstance(spec, dict) and spec.get("spec_text"):
                parts.append(spec["spec_text"])
        return "\n\n".join(parts) if parts else f"No spec found for {self.file_path} — generate based on architecture"

    def _clean_code(self, response: str) -> str:
        import re
        # Search for a ```python ... ``` block anywhere in the response
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # No fence — find first line that looks like Python and strip leading prose
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("import ", "from ", "#!", "# ", "class ", "def ", '"""', "'''")):
                return "\n".join(lines[i:]).strip()
        return response.strip()

    def _fix_reproduce_sh(self, code: str) -> str:
        """Ensure reproduce.sh has correct structure regardless of what the LLM generated."""
        import re

        lines = code.split("\n")

        # Ensure shebang
        if not lines or not lines[0].startswith("#!/"):
            lines.insert(0, "#!/bin/bash")

        # Replace bare `pip` with `python -m pip`
        fixed = []
        for line in lines:
            # Replace `pip install` but not `python -m pip install`
            if re.match(r'\s*pip\s+install', line):
                line = line.replace("pip install", "python -m pip install", 1)
            fixed.append(line)
        lines = fixed

        # Ensure `exec > >(tee -a reproduce.log) 2>&1` is near the top
        has_exec_redirect = any("reproduce.log" in l and "exec" in l for l in lines)
        if not has_exec_redirect:
            # Insert after shebang and set -e
            insert_at = 1
            for i, line in enumerate(lines):
                s = line.strip()
                if s.startswith("set ") or s.startswith("#") or s == "":
                    insert_at = i + 1
                else:
                    break
            lines.insert(insert_at, 'exec > >(tee -a reproduce.log) 2>&1')

        return "\n".join(lines)
