import base64
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.memory.global_memory import GlobalMemory
from src.utils.llm_client import call_vlm


class VLMAgent(BaseAgent):
    def __init__(self, global_memory: GlobalMemory, images_dir: str, output_dir: str = "data/output"):
        super().__init__(name="vlm_agent", global_memory=global_memory, output_dir=output_dir)
        self.images_dir = Path(images_dir)

    async def run(self) -> list[dict]:
        images = self.global_memory.get("images", [])
        self.log_reasoning("start", f"Processing {len(images)} images")

        filtered = await self._filter_figures(images)
        self.log_reasoning("filter", f"Kept {len(filtered)} relevant figures out of {len(images)}")

        analyzed = []
        for fig in filtered:
            result = await self._analyze_figure(fig)
            if result:
                analyzed.append(result)

        self.global_memory.set("figure_analysis", analyzed)
        self.log_reasoning("done", f"Analyzed {len(analyzed)} figures, wrote to global memory")
        return analyzed

    async def _filter_figures(self, images: list[dict]) -> list[dict]:
        filtered = []
        for img in images:
            # Strip leading "images/" prefix if present — images_dir already points there
            rel = img["path"]
            if rel.startswith("images/"):
                rel = rel[len("images/"):]
            img_path = self.images_dir / rel
            if not img_path.exists():
                self.log_reasoning("skip", f"Image not found: {img_path}")
                continue

            classification = await self._classify_figure(img_path, img.get("caption", ""))
            if classification in ("architecture", "pipeline", "system_workflow", "algorithm"):
                img["classification"] = classification
                filtered.append(img)
            else:
                self.log_reasoning("filter_out", f"Filtered out: {img.get('caption', '')[:80]} -> {classification}")

        return filtered

    async def _classify_figure(self, img_path: Path, caption: str) -> str:
        img_b64 = self._image_to_base64(img_path)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"Caption: {caption}\n\n"
                        "You are a strict figure classifier for a research paper.\n\n"
                        "Classify this figure into exactly ONE of the following categories:\n"
                        "- architecture: diagrams showing model layers, blocks, components of a network\n"
                        "- pipeline: diagrams showing a sequence of processing steps or modules\n"
                        "- system_workflow: diagrams showing interactions between components, agents, services\n"
                        "- algorithm: diagrams showing algorithm steps or pseudocode\n"
                        "- result_plot: experimental results, performance graphs, charts, curves\n"
                        "- example: input/output examples, generated samples, visualizations\n"
                        "- other: none of the above\n\n"
                        "You MUST EXCLUDE (classify as result_plot, example, or other) figures that are primarily:\n"
                        "- Metrics, curves, score plots, training curves\n"
                        "- Performance comparisons, ablations, bar charts\n"
                        "- Qualitative examples, galleries of results, synthesized samples\n"
                        "- Dataset examples, failure cases, case studies\n"
                        "- UI screenshots, block catalogs without a full system pipeline\n\n"
                        "Use ONLY the provided caption text and the visual content of the image.\n"
                        "Reply with ONLY the category name, nothing else."
                    ),
                },
            ],
        }]

        response = await call_vlm(messages=messages, max_tokens=500)
        response = (response or "").strip()
        self.log_message("assistant", f"classify {img_path.name}: {response}")
        return response.lower()

    async def _analyze_figure(self, img: dict) -> dict | None:
        rel = img["path"]
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        img_path = self.images_dir / rel
        img_b64 = self._image_to_base64(img_path)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"Caption: {img.get('caption', '')}\n\n"
                        "You are an expert vision-language model specialized in understanding figures "
                        "from computer science and machine learning papers.\n\n"
                        "Extract structured information from this figure that would help RECONSTRUCT "
                        "the model, pipeline, or system in code. Be as concrete and implementation-oriented "
                        "as possible:\n\n"
                        "1. Components/blocks/modules: list each with a concise name, type, and role\n"
                        "2. Data flows: source -> target relationships, what data is passed (tensors, features, etc.)\n"
                        "3. Sequential stages: steps in order, if applicable\n"
                        "4. Inputs and outputs: data types, shapes, files, models, APIs\n"
                        "5. Control flow: loops, conditions, branching, interactions\n\n"
                        "Rules:\n"
                        "- Prefer information grounded in the VISUAL content; use the caption only as supporting context\n"
                        "- Do NOT invent components that are not clearly implied by the figure\n"
                        "- Be specific about dimensions, layer types, activation functions if visible\n\n"
                        "Output MUST be valid JSON only. No extra text.\n\n"
                        "JSON schema:\n"
                        "{\n"
                        '  "category": "model_architecture | pipeline | system_workflow | other",\n'
                        '  "extracted": {\n'
                        '    "components": [{"name": "str", "type": "str", "role": "str"}],\n'
                        '    "flows": [{"source": "str", "target": "str", "data": "str"}],\n'
                        '    "stages": [{"name": "str", "description": "str"}],\n'
                        '    "inputs": ["str"],\n'
                        '    "outputs": ["str"]\n'
                        "  }\n"
                        "}"
                    ),
                },
            ],
        }]

        response = await call_vlm(messages=messages)
        self.log_message("assistant", f"analyze {img_path.name}: {response}")

        return {
            "path": img["path"],
            "caption": img.get("caption", ""),
            "classification": img.get("classification", ""),
            "analysis": response,
        }

    def _image_to_base64(self, img_path: Path) -> str:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
