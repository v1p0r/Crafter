import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils import llm_client


def _anthropic_response(
    response_id,
    *,
    input_tokens,
    output_tokens,
    stop_reason,
    cache_creation_input_tokens=0,
    cache_read_input_tokens=0,
    cache_5m=0,
    cache_1h=0,
):
    return SimpleNamespace(
        id=response_id,
        model="claude-sonnet-4-6",
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=response_id)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=cache_5m,
                ephemeral_1h_input_tokens=cache_1h,
            ),
        ),
    )


class _QueuedAnthropicMessages:
    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, **kwargs):
        return self.responses.pop(0)


class _QueuedOpenAICompletions:
    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, **kwargs):
        return self.responses.pop(0)


class TokenUsageTests(unittest.TestCase):
    def test_per_response_stage_tokens_cache_and_cost(self):
        claude_responses = [
            _anthropic_response(
                "claude-1",
                input_tokens=1000,
                output_tokens=100,
                stop_reason="max_tokens",
                cache_creation_input_tokens=50,
                cache_read_input_tokens=20,
                cache_5m=30,
                cache_1h=20,
            ),
            _anthropic_response(
                "claude-2",
                input_tokens=500,
                output_tokens=50,
                stop_reason="end_turn",
            ),
        ]
        anthropic_client = SimpleNamespace(
            messages=_QueuedAnthropicMessages(claude_responses)
        )
        gemini_response = SimpleNamespace(
            id="gemini-1",
            model="gemini-2.5-flash",
            usage=SimpleNamespace(
                prompt_tokens=2000,
                completion_tokens=200,
                total_tokens=2300,
                prompt_tokens_details=SimpleNamespace(cached_tokens=100),
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="gemini"),
                )
            ],
        )
        gemini_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=_QueuedOpenAICompletions([gemini_response])
            )
        )

        original_backend = llm_client._current_backend
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "token_usage.jsonl"
            summary_path = Path(temp_dir) / "token_usage_summary.json"
            env = {
                "PTC_TOKEN_USAGE_LOG": str(ledger),
                "PTC_RUN_ID": "test-run",
                "PTC_PAPER": "test-paper",
            }
            try:
                with patch.dict(os.environ, env, clear=False):
                    llm_client.set_backend("claude")
                    llm_client.reset_usage()
                    with patch.object(
                        llm_client,
                        "get_minimax_client",
                        return_value=anthropic_client,
                    ):
                        text = asyncio.run(
                            llm_client.call_minimax(
                                [{"role": "user", "content": "test"}],
                                max_continuations=1,
                                stage="codegen",
                            )
                        )
                    self.assertEqual(text, "claude-1claude-2")

                    with patch.object(
                        llm_client,
                        "get_gemini_client",
                        return_value=gemini_client,
                    ):
                        vlm_text = asyncio.run(
                            llm_client.call_vlm(
                                [{"role": "user", "content": "image"}]
                            )
                        )
                    self.assertEqual(vlm_text, "gemini")

                    summary = llm_client.write_usage_summary(
                        ledger,
                        summary_path,
                        metadata={"mode": "test"},
                    )
            finally:
                llm_client.set_backend(original_backend)

            events = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(events), 3)
            self.assertEqual([event["sequence"] for event in events], [1, 2, 3])
            self.assertEqual(
                [event["stage"] for event in events],
                ["codegen", "codegen", "vlm"],
            )

            totals = summary["totals"]
            self.assertEqual(totals["calls"], 3)
            self.assertEqual(totals["input_tokens"], 3570)
            self.assertEqual(totals["uncached_input_tokens"], 3400)
            self.assertEqual(totals["output_tokens"], 450)
            self.assertEqual(totals["visible_output_tokens"], 350)
            self.assertEqual(totals["reasoning_output_tokens"], 100)
            self.assertEqual(totals["cache_creation_input_tokens"], 50)
            self.assertEqual(totals["cache_read_input_tokens"], 120)
            self.assertAlmostEqual(totals["estimated_cost_usd"], 0.0083115)

            self.assertEqual(summary["per_stage"]["codegen"]["calls"], 2)
            self.assertEqual(summary["per_stage"]["vlm"]["calls"], 1)
            self.assertEqual(summary["per_stage"]["partition"]["calls"], 0)
            self.assertEqual(summary["integrity"]["parse_errors"], [])
            self.assertEqual(summary["integrity"]["duplicate_response_ids"], [])
            self.assertEqual(summary["integrity"]["recomputed_schema_v1_events"], 0)

    def test_schema_v1_gemini_event_is_recomputed_from_raw_usage(self):
        event, upgraded = llm_client._upgrade_usage_event_for_aggregation(
            {
                "schema_version": 1,
                "pricing_backend": "gemini",
                "provider_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "total_tokens": 500,
                },
                "tokens": {"input_tokens": 100, "output_tokens": 1},
                "estimated_cost_usd": 0.0,
            }
        )
        self.assertTrue(upgraded)
        self.assertEqual(event["tokens"]["input_tokens"], 100)
        self.assertEqual(event["tokens"]["visible_output_tokens"], 1)
        self.assertEqual(event["tokens"]["reasoning_output_tokens"], 399)
        self.assertEqual(event["tokens"]["output_tokens"], 400)
        self.assertAlmostEqual(event["estimated_cost_usd"], 0.00103)


if __name__ == "__main__":
    unittest.main()
