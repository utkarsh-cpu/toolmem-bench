from __future__ import annotations

import tempfile
import unittest

from toolmem.api import ToolMemoryAPI
from toolmem.executor import LocalExecutor
from toolmem.registry import ToolRegistry


SPEC = {
    "name": "sum_numbers",
    "description": "Sum numbers from an input array",
    "source": "import json,sys\nx=json.load(sys.stdin)\nprint(json.dumps(sum(x['numbers'])))\n",
    "runtime": "python",
    "input_schema": {
        "type": "object",
        "properties": {"numbers": {"type": "array"}},
    },
    "output_schema": {"type": "number"},
    "tags": ["math"],
}


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry(self.temp.name)
        self.api = ToolMemoryAPI(self.registry, LocalExecutor())

    def tearDown(self) -> None:
        self.registry.close()
        self.temp.cleanup()

    def test_exactly_three_tools(self) -> None:
        self.assertEqual(
            [tool["name"] for tool in self.api.definitions()],
            ["create_tool", "find_tool", "update_tool"],
        )

    def test_create_save_and_run_combinations(self) -> None:
        saved = self.api.create_tool(SPEC, save=True, run=False)
        self.assertEqual(saved["storage"], "saved")
        ephemeral = self.api.create_tool(
            {**SPEC, "name": "one_off"}, save=False, run=True, input={"numbers": [1, 2]}
        )
        self.assertEqual(ephemeral["execution"]["payload"], 3)
        self.assertEqual(self.registry.metrics()["active_saved_tools"], 1)

    def test_find_only_and_find_run(self) -> None:
        created = self.api.create_tool(SPEC)
        found = self.api.find_tool("add numeric list", run=False)
        self.assertTrue(found["matches"])
        self.assertNotIn("source", found["matches"][0])
        ran = self.api.find_tool("sum numbers", run=True, input={"numbers": [2, 4]})
        self.assertEqual(ran["execution"]["payload"], 6)
        self.assertEqual(ran["selected_tool_id"], created["tool"]["tool_id"])

    def test_ephemeral_and_saved_update(self) -> None:
        created = self.api.create_tool(SPEC)
        tool_id = created["tool"]["tool_id"]
        ephemeral = self.api.update_tool(
            tool_id,
            "update",
            changes={"source": "print(7)\n"},
            save=False,
            run=True,
            input={},
        )
        self.assertEqual(ephemeral["execution"]["payload"], 7)
        self.assertEqual(self.registry.metrics()["total_versions"], 1)
        saved = self.api.update_tool(
            tool_id,
            "update",
            changes={"source": "print(8)\n"},
            save=True,
            run=False,
        )
        self.assertEqual(saved["tool"]["version"], 2)

    def test_delete_is_an_update_action(self) -> None:
        created = self.api.create_tool(SPEC)
        tool_id = created["tool"]["tool_id"]
        result = self.api.update_tool(tool_id, "delete")
        self.assertEqual(result["status"], "deleted")
        self.assertFalse(self.api.find_tool("sum numbers")["matches"])


if __name__ == "__main__":
    unittest.main()
