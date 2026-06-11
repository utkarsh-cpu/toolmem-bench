from __future__ import annotations

import asyncio
import tempfile
import unittest

from toolmem.executor import LocalExecutor
from toolmem.harness import BenchmarkHarness
from toolmem.models import DeterministicFakeModel
from toolmem.tasks import BenchmarkTask, starter_suite
from toolmem.types import ModelResponse


class HarnessTests(unittest.TestCase):
    def test_fake_model_end_to_end(self) -> None:
        task = BenchmarkTask(
            "simple",
            "Return 4",
            {"type": "number", "expected": 4},
        )
        model = DeterministicFakeModel([ModelResponse(content="4", final_answer="4")])
        with tempfile.TemporaryDirectory() as directory:
            harness = BenchmarkHarness(model, directory, LocalExecutor())
            result = asyncio.run(harness.run_task(task))
            harness.close()
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["meta_tool_calls"], 0)
        self.assertGreater(result.score["composite"], 0.9)

    def test_tool_call_trace_and_memory(self) -> None:
        task = BenchmarkTask(
            "tool",
            "Add values",
            {"type": "number", "expected": 5},
            lifecycle_targets=["create", "run"],
        )
        create = ModelResponse(
            tool_call={
                "name": "create_tool",
                "arguments": {
                    "spec": {
                        "name": "add",
                        "description": "add numbers",
                        "source": "import json,sys\nx=json.load(sys.stdin)\nprint(sum(x['numbers']))\n",
                        "runtime": "python",
                    },
                    "save": True,
                    "run": True,
                    "input": {"numbers": [2, 3]},
                },
            }
        )
        model = DeterministicFakeModel(
            [create, ModelResponse(content="5", final_answer="5")]
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = BenchmarkHarness(model, directory, LocalExecutor())
            result = asyncio.run(harness.run_task(task))
            harness.close()
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["active_saved_tools"], 1)
        self.assertEqual(result.metrics["total_executions"], 1)

    def test_starter_suite_has_twenty_tasks(self) -> None:
        self.assertEqual(len(starter_suite()), 20)
        self.assertEqual(len({task.task_id for task in starter_suite()}), 20)


if __name__ == "__main__":
    unittest.main()
