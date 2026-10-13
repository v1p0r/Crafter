"""Shared helpers for mapping imports to pip packages and maintaining requirements.txt.

Used by the debug agent (auto-install, requirements guarantee) and by the
pipeline after codegen so every generated codebase ships a requirements.txt.
"""

import ast
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger("requirements")

# Import name -> PyPI package name, for the common cases where they differ.
# `pip install <import name>` would install the wrong package (or garbage) for these.
PIP_NAME_MAP = {
    "hydra": "hydra-core",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "googleapiclient": "google-api-python-client",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "git": "GitPython",
}

# Module names we trust enough to pip-install when an error says "<X> is required"
# (no ModuleNotFoundError to confirm it's actually an import). Conservative on
# purpose: "config is required" must not trigger `pip install config`.
REQUIRED_MSG_ALLOWLIST = {
    "hydra", "wandb", "timm", "transformers", "datasets", "accelerate",
    "einops", "flax", "optax", "autoattack",
}


def module_to_pip_name(module: str) -> str:
    """Map a top-level import name to its PyPI package name."""
    top = module.split(".")[0]
    return PIP_NAME_MAP.get(top, top)


def _local_module_names(codebase_dir: Path) -> set[str]:
    """Top-level names importable from the codebase root (files and dirs)."""
    names: set[str] = set()
    for p in codebase_dir.iterdir():
        if p.is_file() and p.suffix == ".py":
            names.add(p.stem)
        elif p.is_dir() and not p.name.startswith((".", "__")):
            names.add(p.name)
    return names


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)|import\s+([A-Za-z_][\w.]*))", re.M)


def _file_imports(path: Path) -> set[str]:
    """Top-level module names imported by a file. AST first, regex fallback
    so files with syntax errors (common in generated code) still contribute."""
    try:
        text = path.read_text()
    except Exception:
        return set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        found = set()
        for m in _IMPORT_RE.finditer(text):
            name = m.group(1) or m.group(2)
            found.add(name.split(".")[0])
        return found
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # skip relative imports
                found.add(node.module.split(".")[0])
    return found


def scan_third_party_imports(codebase_dir: Path) -> set[str]:
    """All third-party top-level modules imported anywhere in the codebase."""
    codebase_dir = Path(codebase_dir)
    local = _local_module_names(codebase_dir)
    stdlib = set(sys.stdlib_module_names)
    modules: set[str] = set()
    for py_file in codebase_dir.rglob("*.py"):
        modules |= _file_imports(py_file)
    return {m for m in modules if m not in stdlib and m not in local and m != "__future__"}


def _canonical(requirement_line: str) -> str:
    """Canonical package name of a requirements line (strip extras/version/comments)."""
    name = re.split(r"[=<>!~\[;#\s]", requirement_line.strip(), 1)[0]
    return name.lower().replace("_", "-")


def ensure_requirements(codebase_dir: str | Path) -> Path | None:
    """Guarantee requirements.txt exists and covers every third-party import.

    Existing entries (user/LLM-written, possibly version-pinned) are preserved;
    missing packages are appended. Returns the path, or None if the dir is missing.
    """
    codebase_dir = Path(codebase_dir)
    if not codebase_dir.is_dir():
        return None
    req_path = codebase_dir / "requirements.txt"

    existing_lines: list[str] = []
    existing_names: set[str] = set()
    if req_path.exists():
        try:
            existing_lines = req_path.read_text().splitlines()
            existing_names = {_canonical(l) for l in existing_lines
                              if l.strip() and not l.strip().startswith("#")}
        except Exception:
            pass

    packages = sorted({module_to_pip_name(m) for m in scan_third_party_imports(codebase_dir)})
    missing = [p for p in packages if _canonical(p) not in existing_names]

    if not missing and req_path.exists():
        return req_path

    lines = [l for l in existing_lines if l.strip()]
    lines.extend(missing)
    req_path.write_text("\n".join(lines) + "\n")
    logger.info(f"requirements.txt: {len(lines)} entries ({len(missing)} added) at {req_path}")
    return req_path


def append_requirement(codebase_dir: str | Path, package: str) -> None:
    """Add a package to requirements.txt if not already present."""
    req_path = Path(codebase_dir) / "requirements.txt"
    existing = ""
    if req_path.exists():
        try:
            existing = req_path.read_text()
        except Exception:
            return
    names = {_canonical(l) for l in existing.splitlines()
             if l.strip() and not l.strip().startswith("#")}
    if _canonical(package) in names:
        return
    sep = "" if (not existing or existing.endswith("\n")) else "\n"
    req_path.write_text(existing + sep + package + "\n")
