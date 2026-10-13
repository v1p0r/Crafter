from abc import ABC, abstractmethod
from typing import Any

from src.memory.global_memory import GlobalMemory
from src.memory.local_memory import LocalMemory


class BaseAgent(ABC):
    def __init__(self, name: str, global_memory: GlobalMemory, output_dir: str = "data/output"):
        self.name = name
        self.global_memory = global_memory
        self.local_memory = LocalMemory(agent_name=name, output_dir=output_dir)

    @abstractmethod
    async def run(self) -> Any:
        pass

    def log_reasoning(self, step: str, reasoning: str) -> None:
        self.local_memory.add_reasoning(step, reasoning)

    def log_message(self, role: str, content: str) -> None:
        self.local_memory.add_message(role, content)

    def log_attempt(self, description: str, result: Any) -> None:
        self.local_memory.add_attempt(description, result)
