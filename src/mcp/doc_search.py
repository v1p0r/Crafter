import json
from typing import Any


class DocSearchMCP:
    """Search within paper content stored in global memory.

    This is a local search over the partitioned paper content,
    not an external MCP server.
    """

    def __init__(self, partitions: list[dict]):
        self.partitions = partitions

    def search(self, query: str) -> list[dict]:
        query_lower = query.lower()
        results = []

        for section in self.partitions:
            score = 0
            matches = []

            # Check title
            title = section.get("title", "").lower()
            if query_lower in title:
                score += 10

            # Check items
            for item in section.get("items", []):
                text = self._extract_text(item).lower()
                if query_lower in text:
                    score += 1
                    matches.append(self._extract_text(item))

            if score > 0:
                results.append({
                    "section": section.get("title", ""),
                    "score": score,
                    "tags": section.get("tags", []),
                    "matches": matches[:5],
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_by_tag(self, tag: str) -> list[dict]:
        results = []
        for section in self.partitions:
            if tag in section.get("tags", []):
                results.append({
                    "section": section.get("title", ""),
                    "tags": section.get("tags", []),
                    "key_content": section.get("key_content", []),
                })
        return results

    def _extract_text(self, item: dict) -> str:
        content = item.get("content", {})
        item_type = item.get("type", "")

        if item_type == "paragraph":
            return self._inline_to_text(content.get("paragraph_content", []))
        elif item_type == "title":
            return self._inline_to_text(content.get("title_content", []))
        elif item_type == "equation_interline":
            return content.get("math_content", "")
        elif item_type == "algorithm":
            caption = self._inline_to_text(content.get("algorithm_caption", []))
            body = self._inline_to_text(content.get("algorithm_content", []))
            return f"{caption} {body}"
        elif item_type == "table":
            return self._inline_to_text(content.get("table_caption", []))
        elif item_type == "image":
            return self._inline_to_text(content.get("image_caption", []))
        return ""

    def _inline_to_text(self, items: list[dict]) -> str:
        parts = []
        for item in items:
            if item["type"] == "text":
                parts.append(item["content"])
            elif item["type"] == "equation_inline":
                parts.append(item["content"])
        return "".join(parts)
