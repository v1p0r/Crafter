import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger("llm_client")

# ---------------------------------------------------------------------------
# Backend configuration & token tracking
# ---------------------------------------------------------------------------

BACKENDS = {
    "minimax": {
        "api_style": "anthropic", "model": "MiniMax-M2.7-highspeed",
        "base_url": "https://api.minimax.io/anthropic", "api_key_env": "ANTHROPIC_API_KEY",
        "input_cost_per_1m": 0.50, "output_cost_per_1m": 2.00,
    },
    "claude": {
        "api_style": "anthropic", "model": "claude-sonnet-4-6",
        "base_url": None, "api_key_env": "CLAUDE_API_KEY_2",
        "input_cost_per_1m": 3.00, "output_cost_per_1m": 15.00,
        "cache_write_5m_cost_per_1m": 3.75,
        "cache_write_1h_cost_per_1m": 6.00,
        "cache_read_cost_per_1m": 0.30,
        "pricing_source_url": "https://platform.claude.com/docs/en/about-claude/pricing",
        "pricing_checked_on": "2026-07-24",
    },
    "gpt": {
        "api_style": "openai", "model": "gpt-4.1",
        "base_url": None, "api_key_env": "OPENAI_API_KEY",
        "input_cost_per_1m": 2.00, "output_cost_per_1m": 8.00,
    },
    "deepseek": {
        "api_style": "openai", "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY",
        "input_cost_per_1m": 0.27, "output_cost_per_1m": 1.10,
        "max_tokens_limit": 8192,
    },
    "gemini": {
        "api_style": "openai", "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "input_cost_per_1m": 0.30, "output_cost_per_1m": 2.50,
        "cache_read_cost_per_1m": 0.03,
        "pricing_source_url": "https://ai.google.dev/gemini-api/docs/pricing",
        "pricing_checked_on": "2026-07-24",
    },
}

_current_backend: str = os.environ.get("LLM_BACKEND", "minimax")


def set_backend(name: str) -> None:
    global _current_backend
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    _current_backend = name
    logger.info(f"Backend set to: {name} ({BACKENDS[name]['model']})")


@dataclass
class TokenUsage:
    """In-memory usage for the current Python process.

    ``input_tokens`` is the total logical input, including cache creation/read
    tokens. The component fields make the billable categories explicit.
    """

    input_tokens: int = 0
    uncached_input_tokens: int = 0
    output_tokens: int = 0
    visible_output_tokens: int = 0
    reasoning_output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0
    estimated_cost_usd: float = 0.0
    cost_breakdown_usd: dict = field(default_factory=dict)
    per_stage: dict = field(default_factory=dict)

    def add(
        self,
        input_tok: int,
        output_tok: int,
        stage: str = "unknown",
        *,
        uncached_input_tok: int | None = None,
        visible_output_tok: int | None = None,
        reasoning_output_tok: int = 0,
        cache_creation_input_tok: int = 0,
        cache_read_input_tok: int = 0,
        estimated_cost_usd: float = 0.0,
        cost_breakdown_usd: dict | None = None,
    ):
        uncached = input_tok if uncached_input_tok is None else uncached_input_tok
        self.input_tokens += int(input_tok)
        self.uncached_input_tokens += int(uncached)
        self.output_tokens += int(output_tok)
        self.visible_output_tokens += int(
            output_tok if visible_output_tok is None else visible_output_tok
        )
        self.reasoning_output_tokens += int(reasoning_output_tok)
        self.cache_creation_input_tokens += int(cache_creation_input_tok)
        self.cache_read_input_tokens += int(cache_read_input_tok)
        self.calls += 1
        self.estimated_cost_usd += float(estimated_cost_usd)
        breakdown = cost_breakdown_usd or {}
        for key, value in breakdown.items():
            self.cost_breakdown_usd[key] = self.cost_breakdown_usd.get(key, 0.0) + float(value)
        if stage not in self.per_stage:
            self.per_stage[stage] = {
                "input_tokens": 0,
                "uncached_input_tokens": 0,
                "output_tokens": 0,
                "visible_output_tokens": 0,
                "reasoning_output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "calls": 0,
                "estimated_cost_usd": 0.0,
                "cost_breakdown_usd": {},
            }
        stage_usage = self.per_stage[stage]
        stage_usage["input_tokens"] += int(input_tok)
        stage_usage["uncached_input_tokens"] += int(uncached)
        stage_usage["output_tokens"] += int(output_tok)
        stage_usage["visible_output_tokens"] += int(
            output_tok if visible_output_tok is None else visible_output_tok
        )
        stage_usage["reasoning_output_tokens"] += int(reasoning_output_tok)
        stage_usage["cache_creation_input_tokens"] += int(cache_creation_input_tok)
        stage_usage["cache_read_input_tokens"] += int(cache_read_input_tok)
        stage_usage["calls"] += 1
        stage_usage["estimated_cost_usd"] += float(estimated_cost_usd)
        for key, value in breakdown.items():
            stage_breakdown = stage_usage["cost_breakdown_usd"]
            stage_breakdown[key] = stage_breakdown.get(key, 0.0) + float(value)

    def cost(self, backend: str | None = None) -> float:
        # Cost is computed per response because one process can use both the
        # selected text backend and Gemini VLM.
        return self.estimated_cost_usd

    def summary(self, backend: str | None = None) -> dict:
        cfg = BACKENDS[backend or _current_backend]
        return {
            "backend": backend or _current_backend,
            "model": cfg["model"],
            "total_input_tokens": self.input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "total_output_tokens": self.output_tokens,
            "visible_output_tokens": self.visible_output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_calls": self.calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "cost_breakdown_usd": _round_costs(self.cost_breakdown_usd),
            "per_stage": self.per_stage,
        }

    def reset(self):
        self.input_tokens = 0
        self.uncached_input_tokens = 0
        self.output_tokens = 0
        self.visible_output_tokens = 0
        self.reasoning_output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.calls = 0
        self.estimated_cost_usd = 0.0
        self.cost_breakdown_usd = {}
        self.per_stage = {}


_usage = TokenUsage()
get_usage = lambda: _usage
_usage_sequence = 0


def reset_usage() -> None:
    global _usage_sequence
    _usage.reset()
    _usage_sequence = 0


def _round_costs(values: dict) -> dict:
    return {key: round(float(value), 8) for key, value in values.items()}


def _get_value(value: Any, key: str, default: Any = 0) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _usage_as_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump(mode="json")
        except TypeError:
            return usage.model_dump()
    result = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cache_creation",
        "prompt_tokens_details",
    ):
        if hasattr(usage, key):
            result[key] = getattr(usage, key)
    return result


def _normalize_usage(usage: Any, api_style: str) -> dict:
    """Return provider-neutral token categories for one API response."""
    if api_style == "anthropic":
        uncached = int(_get_value(usage, "input_tokens", 0) or 0)
        output = int(_get_value(usage, "output_tokens", 0) or 0)
        visible_output = output
        reasoning_output = 0
        cache_read = int(_get_value(usage, "cache_read_input_tokens", 0) or 0)
        cache_creation_total = int(_get_value(usage, "cache_creation_input_tokens", 0) or 0)
        cache_creation = _get_value(usage, "cache_creation", {}) or {}
        cache_write_5m = int(_get_value(cache_creation, "ephemeral_5m_input_tokens", 0) or 0)
        cache_write_1h = int(_get_value(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
        categorized_creation = cache_write_5m + cache_write_1h
        # Older SDKs expose only cache_creation_input_tokens; Anthropic's
        # default cache lifetime is five minutes.
        if categorized_creation < cache_creation_total:
            cache_write_5m += cache_creation_total - categorized_creation
        cache_creation_total = max(cache_creation_total, cache_write_5m + cache_write_1h)
        total_input = uncached + cache_creation_total + cache_read
    else:
        total_input = int(_get_value(usage, "prompt_tokens", 0) or 0)
        visible_output = int(_get_value(usage, "completion_tokens", 0) or 0)
        provider_total = int(_get_value(usage, "total_tokens", 0) or 0)
        # Gemini's OpenAI compatibility layer reports visible candidate tokens
        # in completion_tokens, while total_tokens also includes thinking.
        # Output pricing includes those thinking tokens.
        output = max(visible_output, provider_total - total_input)
        reasoning_output = max(0, output - visible_output)
        details = _get_value(usage, "prompt_tokens_details", {}) or {}
        cache_read = int(_get_value(details, "cached_tokens", 0) or 0)
        cache_write_5m = 0
        cache_write_1h = 0
        cache_creation_total = 0
        # OpenAI-compatible APIs include cached tokens in prompt_tokens.
        uncached = max(0, total_input - cache_read)

    return {
        "input_tokens": total_input,
        "uncached_input_tokens": uncached,
        "output_tokens": output,
        "visible_output_tokens": visible_output,
        "reasoning_output_tokens": reasoning_output,
        "total_tokens": total_input + output,
        "cache_creation_input_tokens": cache_creation_total,
        "cache_creation_5m_input_tokens": cache_write_5m,
        "cache_creation_1h_input_tokens": cache_write_1h,
        "cache_read_input_tokens": cache_read,
    }


def _calculate_cost(tokens: dict, pricing_backend: str) -> tuple[float, dict, dict]:
    cfg = BACKENDS[pricing_backend]
    rates = {
        "uncached_input": float(cfg["input_cost_per_1m"]),
        "output": float(cfg["output_cost_per_1m"]),
        "cache_creation_5m": float(
            cfg.get("cache_write_5m_cost_per_1m", cfg["input_cost_per_1m"])
        ),
        "cache_creation_1h": float(
            cfg.get("cache_write_1h_cost_per_1m", cfg["input_cost_per_1m"])
        ),
        "cache_read": float(cfg.get("cache_read_cost_per_1m", cfg["input_cost_per_1m"])),
    }
    breakdown = {
        "uncached_input": tokens["uncached_input_tokens"] * rates["uncached_input"] / 1_000_000,
        "output": tokens["output_tokens"] * rates["output"] / 1_000_000,
        "cache_creation_5m": (
            tokens["cache_creation_5m_input_tokens"] * rates["cache_creation_5m"] / 1_000_000
        ),
        "cache_creation_1h": (
            tokens["cache_creation_1h_input_tokens"] * rates["cache_creation_1h"] / 1_000_000
        ),
        "cache_read": tokens["cache_read_input_tokens"] * rates["cache_read"] / 1_000_000,
    }
    return sum(breakdown.values()), breakdown, rates


def _track_api_response(
    response: Any,
    *,
    stage: str,
    call_kind: str,
    requested_model: str,
    pricing_backend: str,
    stop_reason: str | None = None,
) -> None:
    """Account for one completed provider response and append one JSONL event."""
    global _usage_sequence
    usage = getattr(response, "usage", None)
    api_style = BACKENDS[pricing_backend]["api_style"]
    tokens = _normalize_usage(usage, api_style)
    cost, cost_breakdown, rates = _calculate_cost(tokens, pricing_backend)
    _usage.add(
        tokens["input_tokens"],
        tokens["output_tokens"],
        stage,
        uncached_input_tok=tokens["uncached_input_tokens"],
        visible_output_tok=tokens["visible_output_tokens"],
        reasoning_output_tok=tokens["reasoning_output_tokens"],
        cache_creation_input_tok=tokens["cache_creation_input_tokens"],
        cache_read_input_tok=tokens["cache_read_input_tokens"],
        estimated_cost_usd=cost,
        cost_breakdown_usd=cost_breakdown,
    )

    _usage_sequence += 1
    event = {
        "schema_version": 2,
        "sequence": _usage_sequence,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("PTC_RUN_ID", ""),
        "pid": os.getpid(),
        "paper": os.environ.get("PTC_PAPER", ""),
        "random_seed": os.environ.get("PTC_RANDOM_SEED"),
        "backend": _current_backend,
        "pricing_backend": pricing_backend,
        "stage": stage,
        "call_kind": call_kind,
        "requested_model": requested_model,
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "stop_reason": stop_reason,
        "tokens": tokens,
        "provider_usage": _usage_as_dict(usage),
        "pricing_usd_per_1m_tokens": rates,
        "pricing_source_url": BACKENDS[pricing_backend].get("pricing_source_url"),
        "pricing_checked_on": BACKENDS[pricing_backend].get("pricing_checked_on"),
        "estimated_cost_usd": round(cost, 10),
        "cost_breakdown_usd": _round_costs(cost_breakdown),
    }

    log_path = os.environ.get("PTC_TOKEN_USAGE_LOG")
    if log_path:
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                handle.flush()
        except Exception as exc:
            # Usage logging must never turn a successful model response into a
            # failed experiment call.
            logger.exception("Failed to append token usage to %s: %s", log_path, exc)

    logger.info(
        "TOKEN_USAGE stage=%s model=%s input=%d output=%d cost=$%.8f",
        stage,
        event["response_model"] or requested_model,
        tokens["input_tokens"],
        tokens["output_tokens"],
        cost,
    )


def _empty_aggregate() -> dict:
    return {
        "calls": 0,
        "input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "visible_output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_creation_5m_input_tokens": 0,
        "cache_creation_1h_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cost_breakdown_usd": {},
    }


def _add_event_to_aggregate(aggregate: dict, event: dict) -> None:
    tokens = event.get("tokens", {})
    aggregate["calls"] += 1
    for key in (
        "input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "visible_output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "cache_creation_input_tokens",
        "cache_creation_5m_input_tokens",
        "cache_creation_1h_input_tokens",
        "cache_read_input_tokens",
    ):
        aggregate[key] += int(tokens.get(key, 0) or 0)
    aggregate["estimated_cost_usd"] += float(event.get("estimated_cost_usd", 0.0) or 0.0)
    for key, value in event.get("cost_breakdown_usd", {}).items():
        costs = aggregate["cost_breakdown_usd"]
        costs[key] = costs.get(key, 0.0) + float(value)


def _finalize_aggregate(aggregate: dict) -> dict:
    result = dict(aggregate)
    result["estimated_cost_usd"] = round(float(result["estimated_cost_usd"]), 8)
    result["cost_breakdown_usd"] = _round_costs(result["cost_breakdown_usd"])
    return result


def _upgrade_usage_event_for_aggregation(event: dict) -> tuple[dict, bool]:
    """Recompute schema-v1 events whose Gemini thinking tokens were not split out."""
    if event.get("schema_version", 0) >= 2 or not event.get("provider_usage"):
        return event, False
    pricing_backend = event.get("pricing_backend")
    if pricing_backend not in BACKENDS:
        return event, False
    tokens = _normalize_usage(
        event["provider_usage"],
        BACKENDS[pricing_backend]["api_style"],
    )
    cost, cost_breakdown, rates = _calculate_cost(tokens, pricing_backend)
    upgraded = dict(event)
    upgraded["tokens"] = tokens
    upgraded["estimated_cost_usd"] = round(cost, 10)
    upgraded["cost_breakdown_usd"] = _round_costs(cost_breakdown)
    upgraded["pricing_usd_per_1m_tokens"] = rates
    return upgraded, True


def summarize_usage_log(log_path: str | Path, metadata: dict | None = None) -> dict:
    """Aggregate the durable response ledger across all invocations in a run directory."""
    path = Path(log_path)
    total = _empty_aggregate()
    per_stage = {
        stage: _empty_aggregate()
        for stage in ("partition", "vlm", "gap", "planning", "codegen", "debug", "unittest")
    }
    per_model: dict[str, dict] = {}
    per_run: dict[str, dict] = {}
    parse_errors = []
    response_ids = set()
    duplicate_response_ids = []
    recomputed_schema_v1_events = 0

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append({"line": line_number, "error": str(exc)})
                    continue
                event, recomputed = _upgrade_usage_event_for_aggregation(event)
                recomputed_schema_v1_events += int(recomputed)
                _add_event_to_aggregate(total, event)
                stage = event.get("stage") or "unknown"
                per_stage.setdefault(stage, _empty_aggregate())
                _add_event_to_aggregate(per_stage[stage], event)
                model = event.get("response_model") or event.get("requested_model") or "unknown"
                per_model.setdefault(model, _empty_aggregate())
                _add_event_to_aggregate(per_model[model], event)
                run_id = event.get("run_id") or "unknown"
                per_run.setdefault(run_id, _empty_aggregate())
                _add_event_to_aggregate(per_run[run_id], event)
                response_id = event.get("response_id")
                if response_id:
                    if response_id in response_ids:
                        duplicate_response_ids.append(response_id)
                    response_ids.add(response_id)

    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_path": str(path),
        "metadata": metadata or {},
        "totals": _finalize_aggregate(total),
        "per_stage": {key: _finalize_aggregate(value) for key, value in per_stage.items()},
        "per_model": {key: _finalize_aggregate(value) for key, value in per_model.items()},
        "per_run": {key: _finalize_aggregate(value) for key, value in per_run.items()},
        "integrity": {
            "parse_errors": parse_errors,
            "duplicate_response_ids": sorted(set(duplicate_response_ids)),
            "recomputed_schema_v1_events": recomputed_schema_v1_events,
        },
        "notes": [
            "input_tokens includes uncached, cache-creation, and cache-read input tokens.",
            "output_tokens includes visible completion and reasoning/thinking output tokens when the provider exposes them.",
            "estimated_cost_usd is computed from the per-response model pricing recorded in the ledger.",
            "The ledger contains token counts and metadata only; prompts and responses are not stored.",
        ],
    }


def write_usage_summary(
    log_path: str | Path,
    summary_path: str | Path,
    metadata: dict | None = None,
) -> dict:
    summary = summarize_usage_log(log_path, metadata=metadata)
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return summary


def parse_json(response: str, fallback: Any = None) -> Any:
    """Robustly extract JSON from LLM response, handling markdown fences and preamble."""
    # Try 1: Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", response, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 2: Find outermost JSON container — whichever of { or [ appears first in the string
    obj_pos = response.find("{")
    arr_pos = response.find("[")
    # Build candidate list ordered by position (earliest first)
    candidates = []
    if obj_pos != -1:
        candidates.append((obj_pos, "{", "}"))
    if arr_pos != -1:
        candidates.append((arr_pos, "[", "]"))
    candidates.sort(key=lambda x: x[0])
    for start_pos, start_char, end_char in candidates:
        depth = 0
        in_string = False
        escape = False
        for i in range(start_pos, len(response)):
            c = response[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[start_pos:i + 1])
                    except json.JSONDecodeError:
                        break

    # Try 3: Brute force the whole response
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    logger.error(f"Failed to parse JSON from response (first 300 chars): {response[:300]}")
    return fallback


def get_minimax_client() -> anthropic.AsyncAnthropic:
    cfg = BACKENDS[_current_backend]
    api_key = os.environ[cfg["api_key_env"]]
    if not cfg["base_url"]:
        return anthropic.AsyncAnthropic(api_key=api_key)
    # Bearer-auth endpoint (MiniMax): pop ANTHROPIC_API_KEY so the SDK doesn't auto-add X-Api-Key.
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        return anthropic.AsyncAnthropic(auth_token=api_key, base_url=cfg["base_url"])
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def get_gemini_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )


def _extract_text(response) -> str:
    """Extract text from response, skipping ThinkingBlocks."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""
def _extract_content_blocks(response) -> tuple[str, list[dict]]:
    """Extract text and tool_use blocks from a response.

    Returns (text, tool_calls) where tool_calls is a list of
    {"id": str, "name": str, "input": dict} dicts.
    """
    text = ""
    tool_calls = []
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
        elif getattr(block, "type", None) == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return text, tool_calls


def _with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with a cache_control breakpoint on the last
    content block of the last message, and stale breakpoints stripped (the API
    allows at most 4)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [dict(b) if isinstance(b, dict) else b for b in content]
            for b in content:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
        out.append({**m, "content": content})
    if out:
        last_content = out[-1].get("content")
        if isinstance(last_content, str) and last_content:
            out[-1]["content"] = [{
                "type": "text", "text": last_content,
                "cache_control": {"type": "ephemeral"},
            }]
        elif isinstance(last_content, list) and last_content and isinstance(last_content[-1], dict):
            last_content[-1]["cache_control"] = {"type": "ephemeral"}
    return out


async def call_minimax_with_tools(
    messages: list[dict],
    tools: list[dict],
    system: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    stage: str = "unknown",
    cache: bool = False,
) -> tuple[str, list[dict], str]:
    """Call the current Anthropic-style backend with tool definitions.

    Only works for backends where api_style == "anthropic" (minimax, claude).
    Retries transient API errors with backoff. With cache=True, adds prompt-cache
    breakpoints (system + conversation tail) on backends that price caching.
    Returns (text, tool_calls, stop_reason).
    """
    cfg = BACKENDS[_current_backend]
    if cfg["api_style"] != "anthropic":
        raise RuntimeError(
            f"Tool calling requires an Anthropic-style backend; "
            f"current backend '{_current_backend}' uses {cfg['api_style']}."
        )
    client = get_minimax_client()

    system_param: Any = system
    request_messages = messages
    if cache and "cache_read_cost_per_1m" in cfg:
        if system:
            system_param = [{
                "type": "text", "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        request_messages = _with_cache_breakpoint(messages)

    delays = [2, 5, 15, 45]  # exponential backoff before retries 1..4
    for attempt in range(len(delays) + 1):
        try:
            response = await client.messages.create(
                model=cfg["model"],
                max_tokens=max_tokens,
                system=system_param,
                messages=request_messages,
                tools=tools,
                temperature=temperature,
            )
            break
        except (anthropic.APIConnectionError, anthropic.RateLimitError,
                anthropic.InternalServerError) as e:
            if attempt >= len(delays):
                raise
            delay = delays[attempt] * random.uniform(0.75, 1.25)
            logger.warning(
                f"Tool call transient error (attempt {attempt + 1}/{len(delays) + 1}, "
                f"retrying in {delay:.1f}s): {e}"
            )
            await asyncio.sleep(delay)
    text, tool_calls = _extract_content_blocks(response)
    _track_api_response(
        response,
        stage=stage,
        call_kind="anthropic_tools",
        requested_model=cfg["model"],
        pricing_backend=_current_backend,
        stop_reason=response.stop_reason,
    )
    return text, tool_calls, response.stop_reason


async def _call_minimax_with_truncation(
    messages: list[dict],
    system: str,
    max_tokens: int,
    temperature: float,
    max_continuations: int,
    max_truncation_attempts: int = 5,
    truncation_factor: float = 0.8,
    stage: str = "unknown",
) -> str:
    """Retry call_minimax with progressively shorter input on overflow."""
    messages = [dict(m) for m in messages]

    for attempt in range(max_truncation_attempts):
        # Find the largest user message and truncate it
        largest_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            key=lambda i: len(messages[i].get("content", "")),
            default=None,
        )
        if largest_idx is None:
            logger.error("Cannot truncate — no user messages found.")
            return ""

        original = messages[largest_idx]["content"]
        truncated = original[:int(len(original) * truncation_factor)]
        messages[largest_idx]["content"] = truncated

        logger.warning(
            f"Truncation attempt {attempt + 1}/{max_truncation_attempts}: "
            f"reduced largest message from {len(original)} to {len(truncated)} chars."
        )

        try:
            # Use _allow_truncation_fallback=False to prevent recursive overflow handling
            return await call_minimax(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                max_continuations=max_continuations,
                _allow_truncation_fallback=False,
                stage=stage,
            )
        except anthropic.BadRequestError as e:
            if "prompt is too long" in str(e) or "context" in str(e).lower():
                continue
            raise

    logger.error(f"Input still too large after {max_truncation_attempts} truncation attempts.")
    return ""


async def call_minimax(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_continuations: int = 10,
    _allow_truncation_fallback: bool = True,
    stage: str = "unknown",
) -> str:
    cfg = BACKENDS[_current_backend]
    # Route OpenAI-style backends to _call_openai
    if cfg["api_style"] == "openai":
        return await _call_openai(
            messages=messages, system=system, max_tokens=max_tokens,
            temperature=temperature, max_continuations=max_continuations, stage=stage,
        )
    model = cfg["model"]
    client = get_minimax_client()
    messages = [dict(m) for m in messages]  # avoid mutating caller's list
    full_text = ""
    response = None

    try:
        for attempt in range(1 + max_continuations):
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=temperature,
            )
            chunk = _extract_text(response)
            full_text += chunk
            _track_api_response(
                response,
                stage=stage,
                call_kind="anthropic_messages",
                requested_model=model,
                pricing_backend=_current_backend,
                stop_reason=response.stop_reason,
            )

            if response.stop_reason != "max_tokens":
                break

            logger.info(f"Output truncated (attempt {attempt + 1}/{max_continuations}), continuing...")

            # Detect if we're mid-JSON so continuation prompt is more specific
            stripped = full_text.strip()
            # Also check for JSON inside markdown fences (```json\n{...)
            looks_like_json = (
                stripped.startswith("{") or stripped.startswith("[")
                or re.search(r"```(?:json)?\s*\n?\s*[\{\[]", stripped) is not None
            )
            if looks_like_json:
                continuation_prompt = (
                    "Your JSON output was truncated. Continue the JSON EXACTLY from where you left off. "
                    "Do not restart or repeat any prior content. Do not add any explanation."
                )
            else:
                continuation_prompt = (
                    "Your output was truncated. Continue EXACTLY from where you left off. "
                    "Do not repeat any prior content."
                )

            # Append only the latest chunk — not full_text — so the conversation
            # history doesn't duplicate earlier chunks on each continuation turn.
            # This builds a proper multi-turn conversation:
            #   user: original prompt
            #   assistant: chunk1  (partial)
            #   user: continue
            #   assistant: chunk2  (continuation, not chunk1+chunk2)
            #   user: continue
            #   assistant: chunk3  (final)
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": continuation_prompt})

        if response and response.stop_reason == "max_tokens":
            logger.warning(f"Output still truncated after {max_continuations} continuations")

    except anthropic.BadRequestError as e:
        if _allow_truncation_fallback and ("prompt is too long" in str(e) or "context" in str(e).lower()):
            # Input overflow — progressively truncate the largest user message and retry
            logger.warning(f"Input context overflow detected. Attempting truncated retry...")
            return await _call_minimax_with_truncation(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                max_continuations=max_continuations,
                stage=stage,
            )
        raise

    return full_text


# ---------------------------------------------------------------------------
# OpenAI-style call (GPT, DeepSeek, Gemini as main backend)
# ---------------------------------------------------------------------------

async def _call_openai(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_continuations: int = 10,
    stage: str = "unknown",
) -> str:
    cfg = BACKENDS[_current_backend]
    kwargs = {"api_key": os.environ[cfg["api_key_env"]]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    client = AsyncOpenAI(**kwargs)
    model = cfg["model"]

    # Clamp max_tokens to backend limit if set
    limit = cfg.get("max_tokens_limit")
    if limit and max_tokens > limit:
        max_tokens = limit

    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)

    full_text = ""

    for attempt in range(1 + max_continuations):
        response = await client.chat.completions.create(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        chunk = choice.message.content or ""
        full_text += chunk
        _track_api_response(
            response,
            stage=stage,
            call_kind="openai_chat_completions",
            requested_model=model,
            pricing_backend=_current_backend,
            stop_reason=choice.finish_reason,
        )

        if choice.finish_reason != "length":
            break

        logger.info(f"Output truncated (attempt {attempt + 1}/{max_continuations}), continuing...")
        stripped = full_text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            cont = ("Your JSON output was truncated. Continue the JSON EXACTLY from where you left off. "
                    "Do not restart or repeat any prior content. Do not add any explanation.")
        else:
            cont = ("Your output was truncated. Continue EXACTLY from where you left off. "
                    "Do not repeat any prior content.")
        oai_messages.append({"role": "assistant", "content": chunk})
        oai_messages.append({"role": "user", "content": cont})

    return full_text


async def call_vlm(
    messages: list[dict],
    model: str = "gemini-2.5-flash",
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    client = get_gemini_client()
    delays = [2, 5, 15, 45]  # exponential backoff before retries 1..4
    last_err = None
    for attempt in range(len(delays) + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            _track_api_response(
                response,
                stage="vlm",
                call_kind="openai_chat_completions_vlm",
                requested_model=model,
                pricing_backend="gemini",
                stop_reason=response.choices[0].finish_reason,
            )
            return response.choices[0].message.content or ""
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
            last_err = e
            if attempt >= len(delays):
                break
            delay = delays[attempt] * random.uniform(0.75, 1.25)
            logger.warning(f"VLM transient error (attempt {attempt + 1}/5, retrying in {delay:.1f}s): {e}")
            await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"VLM call failed (non-retryable): {e}")
            return ""
    logger.error(f"VLM call failed after 5 attempts: {last_err}")
    return ""
