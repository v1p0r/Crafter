import ast
import logging
import re
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.memory.global_memory import GlobalMemory
from src.memory.local_memory import LocalMemory
from src.utils.llm_client import call_minimax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("unittest_agent")


class UnitTestAgent(BaseAgent):
    def __init__(
        self,
        global_memory: GlobalMemory,
        output_dir: str = "data/output",
        codebase_dir: str = "data/output/codebase",
    ):
        super().__init__(name="unittest_agent", global_memory=global_memory, output_dir=output_dir)
        self.codebase_dir = Path(codebase_dir)
        self.output_dir = output_dir

    async def run(self) -> dict[str, Any]:
        logger.info("=== Starting unit test generation ===")
        self.log_reasoning("start", "Beginning unit test generation")

        generated_code = self.global_memory.get("generated_code", {})
        if not generated_code:
            logger.warning("No generated code found — skipping unit test generation")
            return {"success": False, "reason": "no generated code"}

        # Filter to Python files that are testable (skip reproduce.sh, __init__.py, etc.)
        testable_files = [
            fp for fp in generated_code
            if fp.endswith(".py")
            and not fp.endswith("__init__.py")
            and not fp.startswith("test_")
        ]

        if not testable_files:
            logger.warning("No testable Python files found")
            return {"success": False, "reason": "no testable files"}

        logger.info(f"Found {len(testable_files)} testable files")

        # Skip if every test file already exists on disk — otherwise regenerate all.
        expected_tests = [self._make_test_path(fp) for fp in testable_files]
        if expected_tests and all((self.codebase_dir / t).exists() for t in expected_tests):
            logger.info(f"Skipping unit test generation ({len(expected_tests)} test files already exist)")
            return {"success": True, "test_files": expected_tests, "skipped": True}

        test_results: dict[str, str] = {}

        for file_path in testable_files:
            logger.info(f"Generating tests for: {file_path}")
            code = generated_code[file_path]

            # Load the codegen agent's local memory for this file
            codegen_context = self._load_codegen_memory(file_path)

            # Generate unit tests
            test_code = await self._generate_tests(file_path, code, codegen_context)

            if test_code:
                test_file_path = self._make_test_path(file_path)
                # Write test file to codebase
                out_path = self.codebase_dir / test_file_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(test_code)

                test_results[test_file_path] = test_code
                logger.info(f"  Written: {test_file_path} ({len(test_code)} chars)")
                self.log_reasoning(f"generated_{file_path}", f"Created {test_file_path}")
            else:
                logger.warning(f"  No tests generated for {file_path}")

        # Store test code in global memory
        if test_results:
            self.global_memory.set("generated_tests", test_results)

        logger.info(f"=== Unit test generation complete: {len(test_results)} test files ===")
        self.log_reasoning("done", f"Generated {len(test_results)} test files")
        return {"success": True, "test_files": list(test_results.keys())}

    def _load_codegen_memory(self, file_path: str) -> str:
        """Load the codegen agent's local memory for context on how the file was created."""
        safe_name = file_path.replace("/", "_").replace(".", "_")
        agent_name = f"codegen_{safe_name}"

        local_mem = LocalMemory(agent_name=agent_name, output_dir=self.output_dir)
        data = local_mem.all()

        context_parts = []

        # Extract the conversation history (the prompt and response that created this file)
        history = data.get("conversation_history", [])
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                # This contains the spec, architecture, goal, etc. used to generate the file
                context_parts.append(f"## Generation context (what the codegen agent was told):\n{content[:3000]}")
            elif role == "assistant":
                # This is the LLM's response — usually just the code, less useful
                pass

        # Extract reasoning trace
        reasoning = data.get("reasoning_trace", [])
        if reasoning:
            trace_text = "\n".join(f"- {r.get('step', '')}: {r.get('reasoning', '')}" for r in reasoning)
            context_parts.append(f"## Codegen reasoning:\n{trace_text}")

        return "\n\n".join(context_parts) if context_parts else ""

    async def _generate_tests(self, file_path: str, code: str, codegen_context: str) -> str | None:
        """Ask the LLM to break the file into testable blocks and generate unit tests."""

        generated_code = self.global_memory.get("generated_code", {})
        symbol_table = self._build_symbol_table(file_path, code, generated_code)

        system = (
            "You are an expert Python test engineer. Your task is to analyze a source file, "
            "break it into independently testable code blocks (functions, methods, classes), "
            "and write thorough unit tests for each block.\n\n"
            "## Process\n"
            "1. Read the source file and identify every testable unit: functions, class methods, "
            "   property logic, data transformations, etc.\n"
            "2. For each testable unit, write one or more test cases covering:\n"
            "   - Normal/happy path\n"
            "   - Edge cases (empty inputs, boundary values, None, etc.)\n"
            "   - Error conditions (if applicable)\n"
            "3. Use `unittest` (standard library). Use `unittest.mock` for external dependencies "
            "   (file I/O, network, GPU operations, heavy ML training loops).\n\n"
            "## Rules\n"
            "- Mock heavy dependencies (torch training loops, file downloads, API calls) but test "
            "  the actual logic (data processing, shape transformations, config parsing, etc.)\n"
            "- For ML code: test tensor shapes, dtype handling, forward pass output shapes, "
            "  loss computation shapes — use small random tensors, NOT real data\n"
            "- Every test must be runnable without GPU, without real data, without network access\n"
            "- Import ONLY symbols listed in the 'Exact importable symbols' section. "
            "  If a name you'd want isn't there, work around it — do NOT invent class or function names.\n"
            "- Write COMPLETE test code — no TODOs, no placeholders\n"
            "- Each test method should test ONE thing and have a descriptive name\n"
            "- Add a brief comment above each test class explaining what code block it tests\n\n"
            "## Output Format\n"
            "```python\n"
            "import unittest\n"
            "...\n"
            "```\n"
            "Output ONLY the test file code, nothing else."
        )

        # Build the user message with full context
        user_msg = f"## Source file: {file_path}\n```python\n{code}\n```\n"

        if symbol_table:
            user_msg += f"\n## Exact importable symbols (closed set — do NOT invent names)\n{symbol_table}\n"

        if codegen_context:
            user_msg += (
                f"\n## Context from when this file was generated:\n"
                f"Use this to understand the PURPOSE of each function/class — "
                f"what it's supposed to do according to the paper spec.\n\n"
                f"{codegen_context}\n"
            )

        # Include module spec if available
        module_specs = self.global_memory.get("module_specs", {})
        spec_text = self._find_module_spec(file_path, module_specs)
        if spec_text:
            user_msg += f"\n## Module specification:\n{spec_text[:2000]}\n"

        messages = [{"role": "user", "content": user_msg}]
        self.log_message("user", user_msg[:2000] + "..." if len(user_msg) > 2000 else user_msg)

        response = await call_minimax(
            messages=messages, system=system, max_tokens=16384, stage="unittest"
        )
        self.log_message("assistant", response[:2000] + "..." if len(response) > 2000 else response)

        test_code = self._clean_code(response)

        if not test_code or not self._is_valid_python(test_code):
            logger.warning(f"Generated test code for {file_path} is not valid Python")
            self.log_reasoning(f"invalid_{file_path}", "Generated test code failed syntax check")
            return None

        return test_code

    @staticmethod
    def _extract_public_symbols(code: str) -> list[str]:
        """Top-level class/function names defined in `code`, excluding underscores."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        return [
            node.name for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]

    def _build_symbol_table(self, file_path: str, code: str, generated_code: dict) -> str:
        """Compact listing of what's importable from the target file and its deps.

        Tests hallucinate class names when given only raw source; this pins them
        to an explicit closed set. Only includes deps actually referenced by `code`."""
        lines = [f"`{file_path}` exports: {', '.join(self._extract_public_symbols(code)) or '(none)'}"]
        for fp, fp_code in generated_code.items():
            if fp == file_path or not fp.endswith(".py"):
                continue
            module = fp.replace("/", ".").removesuffix(".py")
            # Check if target file references this module (import or dotted path)
            if module not in code and module.split(".")[-1] not in code:
                continue
            syms = self._extract_public_symbols(fp_code)
            if syms:
                lines.append(f"`{module}` exports: {', '.join(syms)}")
        return "\n".join(lines)

    def _find_module_spec(self, file_path: str, module_specs: dict) -> str:
        """Find the module spec that covers this file."""
        for spec in module_specs.values():
            if isinstance(spec, dict) and spec.get("spec_text"):
                if file_path in spec["spec_text"]:
                    return spec["spec_text"]
        return ""

    def _make_test_path(self, file_path: str) -> str:
        """Convert a source file path to its test file path.

        e.g. 'models/unet.py' -> 'tests/test_unet.py'
             'train.py' -> 'tests/test_train.py'
        """
        p = Path(file_path)
        return f"tests/test_{p.stem}.py"

    def _clean_code(self, response: str) -> str:
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("import ", "from ", "#!", "# ", "class ", "def ")):
                return "\n".join(lines[i:]).strip()
        return response.strip()

    def _is_valid_python(self, code: str) -> bool:
        if not code or not code.strip():
            return False
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False
