from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import patch

from toolmem.cli import interactive_args, make_model, parser
from toolmem.models import OpenRouterAdapter


class OpenRouterTests(unittest.TestCase):
    def test_defaults_and_attribution_headers(self) -> None:
        adapter = OpenRouterAdapter(
            model="openai/example",
            api_key="secret",
            referer="https://example.test",
            app_title="Benchmark",
        )
        self.assertEqual(adapter.endpoint, "https://openrouter.ai/api/v1")
        self.assertEqual(adapter.extra_headers["HTTP-Referer"], "https://example.test")
        self.assertEqual(adapter.extra_headers["X-OpenRouter-Title"], "Benchmark")

    def test_attribution_headers_are_optional(self) -> None:
        adapter = OpenRouterAdapter(
            model="openai/example",
            api_key="secret",
            app_title="",
        )
        self.assertEqual(adapter.extra_headers, {})

    def test_cli_constructs_openrouter_adapter(self) -> None:
        args = Namespace(
            model_provider="openrouter",
            model="anthropic/example",
            api_key="secret",
            temperature=0,
            max_tokens=1024,
            referer="",
            app_title="ToolMem Bench",
            endpoint="",
        )
        adapter = make_model(args)
        self.assertIsInstance(adapter, OpenRouterAdapter)
        self.assertEqual(adapter.model, "anthropic/example")

    def test_cli_accepts_openrouter_provider(self) -> None:
        args = parser().parse_args(
            ["run", "--model-provider", "openrouter", "--model", "openai/example"]
        )
        self.assertEqual(args.model_provider, "openrouter")

    def test_interactive_menu_can_browse_tasks(self) -> None:
        with patch("builtins.input", return_value="2"):
            args = interactive_args()
        self.assertEqual(args.command, "list-tasks")


if __name__ == "__main__":
    unittest.main()
