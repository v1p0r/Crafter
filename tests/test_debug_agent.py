import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.debug_agent import DebugAgent
from src.memory.global_memory import GlobalMemory
from src.utils.requirements import ensure_requirements, module_to_pip_name


def make_agent(tmp: Path, **kwargs) -> DebugAgent:
    (tmp / "codebase").mkdir(parents=True, exist_ok=True)
    gm = GlobalMemory(path=str(tmp / "global_memory"))
    return DebugAgent(
        global_memory=gm,
        output_dir=str(tmp / "out"),
        codebase_dir=str(tmp / "codebase"),
        **kwargs,
    )


class TestCoerceFixArgs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)

    def test_proper_list_passthrough(self):
        inp = {"edits": [{"path": "a.py", "start_line": 1, "end_line": 1, "new_code": "x=1"}]}
        problems = self.agent._coerce_fix_args(inp)
        self.assertEqual(problems, [])
        self.assertEqual(len(inp["edits"]), 1)

    def test_json_string_decoded(self):
        inp = {"file_rewrites": '[{"path": "a.py", "content": "x = 1"}]'}
        problems = self.agent._coerce_fix_args(inp)
        self.assertEqual(problems, [])
        self.assertIsInstance(inp["file_rewrites"], list)
        self.assertEqual(inp["file_rewrites"][0]["path"], "a.py")

    def test_xml_leak_rejected_not_iterated(self):
        # The mech run bug: a leaked-XML string must produce ONE problem,
        # never be iterated character by character
        inp = {"edits": '<parameter name="path">utils/api.py</parameter>'}
        problems = self.agent._coerce_fix_args(inp)
        self.assertEqual(len(problems), 1)
        self.assertIn("JSON array", problems[0])
        self.assertEqual(inp["edits"], [])


class TestApplyFix(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)
        self.cb = self.tmp / "codebase"
        (self.cb / "mod.py").write_text("def f():\n    return 1\n")

    def test_valid_rewrite_applied(self):
        applied, problems = self.agent._apply_fix(
            {"file_rewrites": [{"path": "mod.py", "content": "def f():\n    return 2\n"}]}, 1)
        self.assertTrue(applied)
        self.assertEqual(problems, [])
        self.assertIn("return 2", (self.cb / "mod.py").read_text())

    def test_invalid_rewrite_reports_syntax_detail(self):
        applied, problems = self.agent._apply_fix(
            {"file_rewrites": [{"path": "mod.py", "content": "def f(:\n    pass\n"}]}, 1)
        self.assertFalse(applied)
        self.assertEqual(len(problems), 1)
        self.assertIn("not valid Python", problems[0])
        self.assertIn("return 1", (self.cb / "mod.py").read_text())  # unchanged

    def test_missing_file_reported(self):
        applied, problems = self.agent._apply_fix(
            {"edits": [{"path": "ghost.py", "start_line": 1, "end_line": 1, "new_code": "x=1"}]}, 1)
        self.assertFalse(applied)
        self.assertTrue(any("not found" in p for p in problems))

    def test_empty_fix_reported(self):
        applied, problems = self.agent._apply_fix({}, 1)
        self.assertFalse(applied)
        self.assertEqual(len(problems), 1)

    def test_out_of_range_edit_reported(self):
        applied, problems = self.agent._apply_fix(
            {"edits": [{"path": "mod.py", "start_line": 50, "end_line": 60, "new_code": "x=1"}]}, 1)
        self.assertFalse(applied)
        self.assertTrue(any("out of bounds" in p for p in problems))


class TestLineEdits(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)

    def test_simple_replace(self):
        code = "a = 1\nb = 2\nc = 3\n"
        out = self.agent._apply_line_edits(code, [(2, 2, "b = 20")])
        self.assertEqual(out, "a = 1\nb = 20\nc = 3\n")

    def test_reindent_fallback(self):
        code = "def f():\n    x = 1\n    return x\n"
        out = self.agent._apply_line_edits(code, [(2, 2, "x = 2")])
        self.assertIn("    x = 2", out)
        self.assertTrue(self.agent._is_valid_python(out))

    def test_out_of_range_skipped(self):
        code = "a = 1\n"
        out = self.agent._apply_line_edits(code, [(5, 6, "b = 2")])
        self.assertEqual(out, code)


class TestErrorClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)

    def test_normalize_stable_across_line_numbers(self):
        e1 = 'File "m.py", line 42, in f\nTypeError: bad shape (3, 128) at 0x7f01ab'
        e2 = 'File "m.py", line 57, in f\nTypeError: bad shape (3, 256) at 0x7f99cd'
        self.assertEqual(self.agent._normalize_error(e1), self.agent._normalize_error(e2))

    def test_normalize_distinguishes_different_errors(self):
        self.assertNotEqual(
            self.agent._normalize_error("TypeError: bad shape"),
            self.agent._normalize_error("ValueError: bad value"),
        )

    def test_resource_error_positive(self):
        err = ("FileNotFoundError: ImageNet data path not found: ./data/imagenet/train. "
               "Please download ImageNet dataset")
        self.assertTrue(self.agent._is_resource_error(err))

    def test_resource_error_negative(self):
        self.assertFalse(self.agent._is_resource_error(
            "TypeError: forward() missing 1 required positional argument"))

    def test_score_import_vs_runtime(self):
        import_err = {"success": False, "stdout": "",
                      "error": 'Traceback ...\nModuleNotFoundError: No module named "x"'}
        runtime_err = {"success": False, "stdout": "training...",
                       "error": "Traceback ...\nTypeError: bad operand"}
        self.assertEqual(self.agent._score_run(import_err)[0], 0)
        self.assertEqual(self.agent._score_run(runtime_err)[0], 1)
        self.assertEqual(self.agent._score_run({"success": True, "stdout": "ok"})[0], 2)

    def test_clear_regression(self):
        self.assertTrue(self.agent._is_clear_regression((1, 500), (0, 0)))     # runtime -> import
        self.assertTrue(self.agent._is_clear_regression((1, 500), (1, 0)))     # output vanished
        self.assertFalse(self.agent._is_clear_regression((1, 500), (1, 300)))  # just different
        self.assertFalse(self.agent._is_clear_regression((0, 0), (1, 200)))    # improvement
        self.assertFalse(self.agent._is_clear_regression((1, 500), (2, 0)))    # success


class TestAutoInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)
        (self.tmp / "out").mkdir(exist_ok=True)

    def test_pip_name_mapping(self):
        self.assertEqual(module_to_pip_name("hydra"), "hydra-core")
        self.assertEqual(module_to_pip_name("cv2"), "opencv-python")
        self.assertEqual(module_to_pip_name("numpy"), "numpy")
        self.assertEqual(module_to_pip_name("hydra.core"), "hydra-core")

    def test_installs_mapped_package_once(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("src.agents.debug_agent.subprocess.run", side_effect=fake_run):
            self.agent._install_missing_packages("ModuleNotFoundError: No module named 'hydra'")
            self.agent._install_missing_packages("ModuleNotFoundError: No module named 'hydra'")

        pip_calls = [c for c in calls if "pip" in c]
        self.assertEqual(len(pip_calls), 1)  # never reinstall the same module
        self.assertIn("hydra-core", pip_calls[0])
        self.assertIn("-m", pip_calls[0])  # python -m pip, not bare pip
        # verified install lands in requirements.txt
        req = (self.tmp / "codebase" / "requirements.txt").read_text()
        self.assertIn("hydra-core", req)

    def test_failed_import_after_install_noted(self):
        def fake_run(cmd, **kwargs):
            if "pip" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="nope")

        with patch("src.agents.debug_agent.subprocess.run", side_effect=fake_run):
            self.agent._install_missing_packages("No module named 'weirdmod'")
        self.assertTrue(any("weirdmod" in n for n in self.agent.pip_notes))

    def test_is_required_allowlist(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("src.agents.debug_agent.subprocess.run", side_effect=fake_run):
            self.agent._install_missing_packages("ERROR: Hydra is required to run this")
            self.agent._install_missing_packages("A valid config is required")  # must NOT install

        pip_calls = [c for c in calls if "pip" in c and "install" in c]
        self.assertEqual(len(pip_calls), 1)
        self.assertIn("hydra-core", pip_calls[0])


class TestEntryPointDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)
        self.cb = self.tmp / "codebase"

    def test_override_wins(self):
        agent = make_agent(self.tmp, entry_point="custom.py")
        self.assertEqual(agent._detect_entry_point(), "custom.py")

    def test_reproduce_sh_parsed(self):
        (self.cb / "train_model.py").write_text("print('hi')\n")
        (self.cb / "reproduce.sh").write_text(
            "#!/bin/bash\npip install -r requirements.txt\npython train_model.py --epochs 1\n")
        self.assertEqual(self.agent._detect_entry_point(), "train_model.py")

    def test_conventional_fallback(self):
        (self.cb / "run_experiments.py").write_text("print('hi')\n")
        self.assertEqual(self.agent._detect_entry_point(), "run_experiments.py")

    def test_none_when_nothing_found(self):
        self.assertIsNone(self.agent._detect_entry_point())


class TestRunTimeout(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cb = self.tmp / "codebase"

    def _agent_with_entry(self, **kwargs):
        agent = make_agent(self.tmp, **kwargs)
        (self.cb / "main.py").write_text("print('x')\n")
        return agent

    def _timeout(self, stdout="", stderr=""):
        return subprocess.TimeoutExpired(cmd="python", timeout=120,
                                         output=stdout, stderr=stderr)

    def test_timeout_with_progress_verified(self):
        agent = self._agent_with_entry()
        with patch("src.agents.debug_agent.subprocess.run",
                   side_effect=self._timeout(stdout="epoch 1: loss 0.53\n")):
            result = agent._run_code()
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])

    def test_silent_timeout_flagged_not_failed(self):
        agent = self._agent_with_entry()
        with patch("src.agents.debug_agent.subprocess.run", side_effect=self._timeout()):
            result = agent._run_code()
        self.assertTrue(result["success"])
        self.assertFalse(result["verified"])
        self.assertTrue(result["silent"])

    def test_silent_timeout_strict_fails(self):
        agent = self._agent_with_entry(strict_timeout=True)
        with patch("src.agents.debug_agent.subprocess.run", side_effect=self._timeout()):
            result = agent._run_code()
        self.assertFalse(result["success"])
        self.assertIn("no output", result["error"])


class TestRequirements(unittest.TestCase):
    def test_scan_maps_and_filters(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "utils").mkdir()
        (tmp / "main.py").write_text("import os\nimport numpy as np\nimport cv2\n"
                                     "from utils.helpers import f\n")
        (tmp / "utils" / "helpers.py").write_text("import torch\ndef f():\n    pass\n")
        req = ensure_requirements(tmp).read_text()
        self.assertIn("numpy", req)
        self.assertIn("opencv-python", req)
        self.assertIn("torch", req)
        self.assertNotIn("os", req.split())     # stdlib filtered
        self.assertNotIn("utils", req)          # local module filtered

    def test_existing_pins_preserved(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "main.py").write_text("import numpy\nimport scipy\n")
        (tmp / "requirements.txt").write_text("numpy==1.24.0\n")
        req = ensure_requirements(tmp).read_text()
        self.assertIn("numpy==1.24.0", req)
        self.assertIn("scipy", req)
        self.assertEqual(req.count("numpy"), 1)

    def test_syntax_error_file_still_scanned(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "broken.py").write_text("import pandas\ndef f(:\n")
        req = ensure_requirements(tmp).read_text()
        self.assertIn("pandas", req)


class TestSnapshots(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)
        self.cb = self.tmp / "codebase"
        (self.cb / "a.py").write_text("original = 1\n")

    def test_restore_iteration_reverts_edit_and_removes_created(self):
        self.agent._snapshot_file(3, "a.py")
        (self.cb / "a.py").write_text("broken = 1\n")
        self.agent._snapshot_file(3, "new.py")  # doesn't exist yet -> "created"
        (self.cb / "new.py").write_text("junk = 1\n")

        self.agent._restore_iteration(3)
        self.assertEqual((self.cb / "a.py").read_text(), "original = 1\n")
        self.assertFalse((self.cb / "new.py").exists())

    def test_best_state_roundtrip(self):
        self.agent._save_best_state()
        (self.cb / "a.py").write_text("worse = 1\n")
        self.agent._restore_best_state()
        self.assertEqual((self.cb / "a.py").read_text(), "original = 1\n")


class TestStaticChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = make_agent(self.tmp)
        self.cb = self.tmp / "codebase"

    def test_syntax_error_detected(self):
        (self.cb / "bad.py").write_text("def f(:\n    pass\n")
        issues = self.agent._collect_static_issues()
        self.assertTrue(any("bad.py" == f and "SyntaxError" in p for f, p in issues))

    def test_broken_local_import_detected(self):
        # The mech run bug: main imports a name the module doesn't define
        (self.cb / "ablation.py").write_text("def run_ablation_alpha_sweep():\n    pass\n")
        (self.cb / "main.py").write_text("from ablation import run_alpha_sweep\n")
        issues = self.agent._collect_static_issues()
        self.assertTrue(any("run_alpha_sweep" in p for _, p in issues))

    def test_valid_import_not_flagged(self):
        (self.cb / "ablation.py").write_text("def run():\n    pass\nCONST = 3\n")
        (self.cb / "main.py").write_text("from ablation import run, CONST\nimport numpy\n")
        self.assertEqual(self.agent._collect_static_issues(), [])

    def test_public_names(self):
        names = self.agent._public_names(
            "def keep():\n    pass\nclass Model:\n    pass\ndef _private():\n    pass\n")
        self.assertEqual(names, {"keep", "Model"})


class TestJournal(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "out").mkdir(exist_ok=True)
        self.agent = make_agent(self.tmp)

    def test_finalize_unchanged_vs_changed(self):
        err = "TypeError: bad shape at line 42"
        self.agent.journal.append({"iteration": 1, "error_before": err,
                                   "outcome": "pending", "applied": True, "files": []})
        self.agent._finalize_last_entry(False, "TypeError: bad shape at line 57")
        self.assertEqual(self.agent.journal[-1]["outcome"], "unchanged")

        self.agent.journal.append({"iteration": 2, "error_before": err,
                                   "outcome": "pending", "applied": True, "files": []})
        self.agent._finalize_last_entry(False, "ValueError: something else entirely")
        self.assertEqual(self.agent.journal[-1]["outcome"], "error-changed")

        self.agent.journal.append({"iteration": 3, "error_before": err,
                                   "outcome": "pending", "applied": True, "files": []})
        self.agent._finalize_last_entry(True, "")
        self.assertEqual(self.agent.journal[-1]["outcome"], "fixed")

    def test_context_rendering_and_notes(self):
        self.agent.journal.append({"iteration": 1, "error_before": "E", "root_cause": "rc",
                                   "fix": "fx", "files": ["a.py"], "outcome": "unchanged",
                                   "applied": True})
        self.agent.codebase_notes.append("Trainer expects TrainerConfig, not kwargs")
        ctx = self.agent._journal_context()
        self.assertIn("do not repeat this approach", ctx)
        self.assertIn("TrainerConfig", ctx)

    def test_write_journal_file(self):
        self.agent.journal.append({"iteration": 1, "error_before": "E", "root_cause": "rc",
                                   "fix": "fx", "files": ["a.py"], "outcome": "fixed",
                                   "applied": True})
        self.agent._write_journal()
        content = (self.tmp / "out" / "debug_journal.md").read_text()
        self.assertIn("Iteration 1", content)
        self.assertIn("fixed", content)


if __name__ == "__main__":
    unittest.main()
