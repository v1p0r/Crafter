import json
import threading
from pathlib import Path
from typing import Any

# Keys whose values are structured data (dicts/lists) — stored as JSON.
# Everything else is markdown/text/code — stored as .md or raw files.
_JSON_KEYS = {"partitions", "images", "figure_analysis", "architecture", "generated_code", "module_specs"}


class GlobalMemory:
    """Shared memory across all agents.

    Each key is persisted as its own file inside a directory:
    - Structured data (lists, dicts) → key.json
    - Text content (markdown, code) → key.md

    The in-memory API (get/set/update/delete/keys/all) is unchanged.
    """

    def __init__(self, path: str = "global_memory"):
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

        # Migrate from legacy single-file format if it exists
        legacy_json = self.path.with_suffix(".json") if not self.path.suffix else None
        if legacy_json and legacy_json.is_file() and not self.path.is_dir():
            self._migrate_from_json(legacy_json)
            return

        # Load from directory
        if self.path.is_dir():
            self._load_from_dir()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save_key_async(key, value)

    def update(self, data: dict[str, Any]) -> None:
        self._data.update(data)
        for key, value in data.items():
            self._save_key_async(key, value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        # Remove the file in a background thread
        threading.Thread(target=self._delete_file, args=(key,), daemon=True).start()

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def all(self) -> dict[str, Any]:
        return self._data.copy()

    # ── Persistence ───────────────────────────────────────────────────

    def _save_key_async(self, key: str, value: Any) -> None:
        """Serialize in the calling (asyncio) thread, write in background."""
        if self._is_json_key(key, value):
            data_str = json.dumps(value, indent=2, ensure_ascii=False)
            ext = ".json"
        else:
            data_str = str(value)
            ext = ".md"
        threading.Thread(
            target=self._write_file, args=(key, data_str, ext), daemon=True
        ).start()

    def _write_file(self, key: str, data_str: str, ext: str) -> None:
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)
            # Remove old file with opposite extension if it exists
            old_ext = ".json" if ext == ".md" else ".md"
            old_file = self.path / f"{key}{old_ext}"
            if old_file.exists():
                old_file.unlink()
            file_path = self.path / f"{key}{ext}"
            file_path.write_text(data_str, encoding="utf-8")

    def _delete_file(self, key: str) -> None:
        with self._lock:
            for ext in (".json", ".md"):
                fp = self.path / f"{key}{ext}"
                if fp.exists():
                    fp.unlink()

    def _load_from_dir(self) -> None:
        """Load all key files from the memory directory."""
        for fp in self.path.iterdir():
            if fp.is_file() and fp.suffix in (".json", ".md"):
                key = fp.stem
                try:
                    text = fp.read_text(encoding="utf-8")
                    if fp.suffix == ".json":
                        self._data[key] = json.loads(text)
                    else:
                        self._data[key] = text
                except (json.JSONDecodeError, ValueError, OSError):
                    pass  # skip corrupted files

    def _migrate_from_json(self, legacy_path: Path) -> None:
        """One-time migration from single global_memory.json to directory format."""
        try:
            with open(legacy_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            return

        self._data = data
        self.path.mkdir(parents=True, exist_ok=True)

        # Write each key as its own file
        for key, value in data.items():
            if self._is_json_key(key, value):
                ext = ".json"
                content = json.dumps(value, indent=2, ensure_ascii=False)
            else:
                ext = ".md"
                content = str(value)
            (self.path / f"{key}{ext}").write_text(content, encoding="utf-8")

        # Rename legacy file so migration doesn't re-run
        legacy_path.rename(legacy_path.with_suffix(".json.bak"))

    @staticmethod
    def _is_json_key(key: str, value: Any = None) -> bool:
        """Determine if a key should be stored as JSON."""
        if key in _JSON_KEYS:
            return True
        # Fallback: store dicts and lists as JSON, strings as markdown
        if value is not None:
            return isinstance(value, (dict, list))
        return False
