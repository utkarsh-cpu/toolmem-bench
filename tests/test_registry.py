from __future__ import annotations

import tempfile
import unittest

from toolmem.registry import ToolRegistry
from toolmem.types import ToolSpec


def spec(
    name: str = "sum_numbers",
    source: str | None = None,
    description: str = "Sum numeric values from a numbers array",
) -> ToolSpec:
    return ToolSpec.from_dict(
        {
            "name": name,
            "description": description,
            "source": source or "import json,sys\nx=json.load(sys.stdin)\nprint(sum(x['numbers']))\n",
            "runtime": "python",
            "input_schema": {
                "type": "object",
                "properties": {"numbers": {"type": "array"}},
            },
            "output_schema": {"type": "number"},
            "tags": ["math"],
        }
    )


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry(self.temp.name)

    def tearDown(self) -> None:
        self.registry.close()
        self.temp.cleanup()

    def test_versions_are_immutable_and_restorable(self) -> None:
        first = self.registry.save(spec())
        second = self.registry.save(spec(source="print(42)\n"), first.tool_id)
        self.assertEqual(second.version, 2)
        self.assertNotEqual(self.registry.get(first.tool_id, 1).source_hash, second.source_hash)
        restored = self.registry.restore(first.tool_id, 1)
        self.assertEqual(restored.version, 3)
        self.assertEqual(restored.restored_from, 1)

    def test_delete_version_and_tool(self) -> None:
        first = self.registry.save(spec())
        self.registry.save(spec(source="print(42)\n"), first.tool_id)
        self.registry.delete(first.tool_id, 2)
        self.assertEqual(self.registry.get(first.tool_id).version, 1)
        self.registry.delete(first.tool_id)
        with self.assertRaises(KeyError):
            self.registry.get(first.tool_id)

    def test_search_strategies_and_metrics(self) -> None:
        self.registry.save(spec())
        self.registry.save(
            spec("reverse_text", "print('x')\n", "Reverse characters in a text string")
        )
        for strategy in ("lexical", "semantic", "hybrid"):
            results = self.registry.search("sum a list of numbers", strategy=strategy)
            self.assertTrue(results)
            self.assertEqual(results[0].name, "sum_numbers")
        metrics = self.registry.metrics()
        self.assertEqual(metrics["active_saved_tools"], 2)
        self.assertEqual(metrics["total_versions"], 2)

    def test_duplicate_storage_is_measured_not_rejected(self) -> None:
        self.registry.save(spec())
        self.registry.save(spec())
        metrics = self.registry.metrics()
        self.assertEqual(metrics["active_saved_tools"], 2)
        self.assertEqual(metrics["exact_duplicate_versions"], 1)
        self.assertGreater(metrics["exact_duplicate_rate"], 0)


if __name__ == "__main__":
    unittest.main()
