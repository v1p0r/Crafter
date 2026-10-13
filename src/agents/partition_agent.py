import json
import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.memory.global_memory import GlobalMemory


class PartitionAgent(BaseAgent):
    def __init__(self, global_memory: GlobalMemory, input_path: str, output_dir: str = "data/output"):
        super().__init__(name="partition_agent", global_memory=global_memory, output_dir=output_dir)
        self.input_path = input_path

    async def run(self) -> dict[str, Any]:
        self.log_reasoning("start", "Loading and partitioning content list")

        content_list = self._load_content_list()
        flat_items = self._flatten_and_filter(content_list)
        sections = self._split_by_sections(flat_items)
        self._tag_section_roles(sections)
        images = self._extract_images(flat_items)

        self.log_reasoning("partition", f"Found {len(sections)} sections, {len(images)} images")

        self.global_memory.set("partitions", sections)
        self.global_memory.set("images", images)

        self.log_reasoning("done", f"Partitioned {len(sections)} sections, wrote to global memory")
        return {"sections": sections, "images": images}

    def _load_content_list(self) -> list[list[dict]]:
        with open(self.input_path, "r") as f:
            return json.load(f)

    def _flatten_and_filter(self, content_list: list[list[dict]]) -> list[dict]:
        items = []
        for page_idx, page in enumerate(content_list):
            for item in page:
                if item["type"] in ("page_header", "page_number"):
                    continue
                item["page_idx"] = page_idx
                items.append(item)
        return items

    def _split_by_sections(self, items: list[dict]) -> list[dict]:
        sections = []
        current_section = None

        for item in items:
            if item["type"] == "title" and item["content"].get("level") == 1:
                title_text = self._extract_text(item["content"].get("title_content", []))
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": title_text.strip(),
                    "items": [item],
                }
            elif current_section is not None:
                current_section["items"].append(item)
            else:
                # Content before first section heading
                current_section = {
                    "title": "preamble",
                    "items": [item],
                }

        if current_section:
            sections.append(current_section)

        return sections

    @staticmethod
    def _tag_section_roles(sections: list[dict]) -> None:
        """Tag each section with a `section_role`: body, references, appendix, or preamble.

        Rules applied in order:
        1. References / bibliography sections → "references"
        2. Explicit appendix markers (title starts with "Appendix", "supplementary",
           or single-letter numbering like "A." / "A ") → "appendix"
        3. Everything AFTER the references section → "appendix" (common paper layout)
        4. Preamble placeholder → "preamble"
        5. Default → "body"
        """
        seen_references = False
        for s in sections:
            title = s.get("title", "")
            lower = title.lower().strip()

            if any(kw in lower for kw in ("reference", "bibliography", "cited")):
                s["section_role"] = "references"
                seen_references = True
                continue

            # Appendix patterns: "Appendix X", "supplementary material",
            # "A. Title" (letter + dot + space), "A Title" (letter + space + capital).
            is_appendix_pattern = (
                lower.startswith("appendix")
                or "supplementary" in lower
                or re.match(r"^[A-Z]\.?\s+[A-Z]", title) is not None
            )
            if is_appendix_pattern or seen_references:
                s["section_role"] = "appendix"
            elif title == "preamble":
                s["section_role"] = "preamble"
            else:
                s["section_role"] = "body"

    def _extract_images(self, items: list[dict]) -> list[dict]:
        images = []
        for item in items:
            if item["type"] == "image":
                content = item["content"]
                caption = self._extract_text(content.get("image_caption", []))
                images.append({
                    "path": content.get("image_source", {}).get("path", ""),
                    "caption": caption,
                    "bbox": item.get("bbox"),
                    "page_idx": item.get("page_idx"),
                })
        return images

    def _extract_text(self, content_list: list[dict]) -> str:
        parts = []
        for item in content_list:
            if item["type"] == "text":
                parts.append(item["content"])
            elif item["type"] == "equation_inline":
                parts.append(f"${item['content']}$")
        return "".join(parts)
