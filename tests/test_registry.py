from __future__ import annotations

import tempfile
import unittest

from toolmem.registry import ToolRegistry, source_hash
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

    def test_update_deleted_tool_raises(self) -> None:
        saved = self.registry.save(spec())
        self.registry.delete(saved.tool_id)
        with self.assertRaises(KeyError):
            self.registry.save(spec(), tool_id=saved.tool_id)

    def test_restore_deleted_tool_works(self) -> None:
        saved = self.registry.save(spec())
        self.registry.save(spec(source="print(42)\n"), tool_id=saved.tool_id)
        self.registry.delete(saved.tool_id)
        restored = self.registry.restore(saved.tool_id, 1)
        self.assertEqual(restored.version, 3)
        self.assertEqual(restored.state.value, "active")
        self.assertEqual(self.registry.get(saved.tool_id).version, 3)

    def test_has_tool_idempotent_seed(self) -> None:
        tool_spec = spec()
        self.registry.save(tool_spec)
        self.registry.save(tool_spec)
        self.assertTrue(
            self.registry.has_tool(
                tool_spec.name,
                source_hash(tool_spec.source),
            )
        )
        self.assertEqual(self.registry.metrics()["active_saved_tools"], 2)

    def test_near_duplicate_stats_separate(self) -> None:
        self.registry.save(spec())
        self.registry.save(spec(source="print(42)\n"))
        metrics = self.registry.metrics()
        self.assertEqual(metrics["near_duplicate_pairs"], 0)
        stats = self.registry.near_duplicate_stats()
        self.assertEqual(stats["near_duplicate_pairs"], 1)
        self.assertEqual(stats["near_duplicate_rate"], 1.0)

    def test_empty_query_fts_does_not_crash(self) -> None:
        self.registry.save(spec())
        self.assertIsInstance(self.registry.search(""), list)

    def test_stale_embedding_dimension_scores_zero(self) -> None:
        saved = self.registry.save(spec())
        self.registry.connection.execute(
            "UPDATE versions SET embedding = ? WHERE tool_id = ?",
            ("[0.0]", saved.tool_id),
        )
        self.registry.connection.commit()
        results = self.registry.search("sum numbers", strategy="semantic")
        self.assertEqual(results[0].semantic_score, 0.0)


if __name__ == "__main__":
    unittest.main()
