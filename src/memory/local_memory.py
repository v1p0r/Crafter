import json
import threading
from pathlib import Path
from typing import Any


class LocalMemory:
    def __init__(self, agent_name: str, output_dir: str = "data/output"):
        self.agent_name = agent_name
        self.path = Path(output_dir) / "local_memory" / f"{agent_name}.json"
        self._data: dict[str, Any] = {
            "agent_name": agent_name,
            "conversation_history": [],
            "reasoning_trace": [],
            "intermediate_attempts": [],
        }
        self._lock = threading.Lock()

        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass  # corrupted file — start fresh

    def add_message(self, role: str, content: str) -> None:
        self._data["conversation_history"].append({
            "role": role,
            "content": content,
        })
        self._save_async()

    def add_reasoning(self, step: str, reasoning: str) -> None:
        self._data["reasoning_trace"].append({
            "step": step,
            "reasoning": reasoning,
        })
        self._save_async()

    def add_attempt(self, description: str, result: Any) -> None:
        self._data["intermediate_attempts"].append({
            "description": description,
            "result": result,
        })
        self._save_async()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save_async()

    def get_conversation_history(self) -> list[dict]:
        return self._data["conversation_history"]

    def all(self) -> dict[str, Any]:
        return self._data.copy()

    def _save_async(self) -> None:
        data_str = json.dumps(self._data, indent=2, ensure_ascii=False)
        threading.Thread(target=self._write_to_disk, args=(data_str,), daemon=True).start()

    def _write_to_disk(self, data_str: str) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                f.write(data_str)
