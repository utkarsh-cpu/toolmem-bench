from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ToolSpec


@dataclass(slots=True)
class BenchmarkTask:
    task_id: str
    prompt: str
    grader: dict[str, Any]
    lifecycle_targets: list[str] = field(default_factory=list)
    seeded_tools: list[ToolSpec] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def grade_answer(answer: Any, grader: dict[str, Any]) -> bool:
    kind = grader["type"]
    expected = grader.get("expected")
    if kind == "exact":
        return answer == expected
    if kind == "json":
        try:
            actual = json.loads(answer) if isinstance(answer, str) else answer
        except json.JSONDecodeError:
            return False
        return actual == expected
    if kind == "contains":
        return str(expected).lower() in str(answer).lower()
    if kind == "number":
        try:
            return abs(float(answer) - float(expected)) <= float(grader.get("tolerance", 1e-9))
        except (TypeError, ValueError):
            return False
    raise ValueError(f"unknown grader type: {kind}")


def _python_tool(name: str, description: str, body: str, tags: list[str]) -> ToolSpec:
    return ToolSpec.from_dict(
        {
            "name": name,
            "description": description,
            "source": body,
            "runtime": "python",
            "input_schema": {"type": "object"},
            "output_schema": {},
            "tags": tags,
        }
    )


def starter_suite() -> list[BenchmarkTask]:
    seeded_sum = _python_tool(
        "sum_numbers",
        "Sum a list of numeric values supplied as numbers",
        "import json,sys\nx=json.load(sys.stdin)\nprint(json.dumps(sum(x['numbers'])))\n",
        ["math", "reusable"],
    )
    broken_csv = _python_tool(
        "csv_total",
        "Sum the amount column from CSV text",
        "import json,sys\nx=json.load(sys.stdin)\nprint(sum(int(r.split(',')[1]) for r in x['csv'].splitlines()))\n",
        ["csv", "broken"],
    )
    distractor = _python_tool(
        "reverse_text",
        "Reverse a text string",
        "import json,sys\nx=json.load(sys.stdin)\nprint(json.dumps(x['text'][::-1]))\n",
        ["text"],
    )
    definitions = [
        ("normalize-names", "Normalize ['  ADA  ','Grace HOPPER'] to title-cased trimmed names.", ["create", "ephemeral"], "json", ["Ada", "Grace Hopper"]),
        ("aggregate-sales", "Sum sales records [{'amount':12.5},{'amount':7.5}] and return 20.", ["create"], "number", 20),
        ("extract-emails", "Extract sorted unique emails from 'a@x.com A@x.com b@y.org'.", ["create"], "json", ["a@x.com", "b@y.org"]),
        ("word-frequency", "Return word frequencies for 'red blue red green blue red'.", ["create"], "json", {"red": 3, "blue": 2, "green": 1}),
        ("date-conversion", "Convert 06/11/2026 to ISO date 2026-06-11.", ["create"], "exact", "2026-06-11"),
        ("checksum", "Compute the SHA-256 of the UTF-8 text 'tool-memory'.", ["create"], "exact", "4997b1787137b1d263c64b60a9f872d14729288a776fb479eccb1dbcfae8a360"),
        ("json-flatten", "Flatten {'a':{'b':2},'c':3} using dotted keys.", ["create"], "json", {"a.b": 2, "c": 3}),
        ("markdown-headings", "Extract heading texts from '# Alpha\\ntext\\n## Beta'.", ["create"], "json", ["Alpha", "Beta"]),
        ("prime-check", "Determine whether 104729 is prime. Return true.", ["create"], "json", True),
        ("deduplicate-records", "Deduplicate records by id, keeping the last: [{'id':1,'v':'a'},{'id':1,'v':'b'}].", ["create"], "json", [{"id": 1, "v": "b"}]),
        ("reuse-sum", "Use the saved numeric summation capability for [3,5,8,13].", ["find", "reuse"], "number", 29),
        ("find-distractors", "Find and use the numeric summation tool, ignoring unrelated tools, for [40,2].", ["find"], "number", 42),
        ("repair-csv", "Repair the saved CSV total tool so it handles a header and decimals, then total 'name,amount\\na,1.5\\nb,2.25'.", ["find", "update"], "number", 3.75),
        ("version-choice", "Use the most reliable current saved summation tool for [10,20,30].", ["find"], "number", 60),
        ("no-tool-needed", "Answer directly: what is 2 + 2?", [], "number", 4),
        ("one-off-transform", "For this one-off task, reverse the list [1,2,3] without polluting persistent memory.", ["ephemeral"], "json", [3, 2, 1]),
        ("delete-duplicate", "Remove an obsolete duplicate tool if present, then return 'clean'.", ["delete"], "exact", "clean"),
        ("reusable-slug", "Create a reusable slugification tool and slugify 'Agentic Tool Memory'.", ["create", "save"], "exact", "agentic-tool-memory"),
        ("multi-step", "Normalize [3,1,3,2], remove duplicates, sort, and return their sum.", ["create", "run"], "number", 6),
        ("schema-retrieval", "Find a tool accepting a numbers array and use it for [1,1,2,3,5,8].", ["find", "schema"], "number", 20),
    ]
    tasks: list[BenchmarkTask] = []
    for index, (task_id, prompt, targets, grader_type, expected) in enumerate(definitions):
        seeds: list[ToolSpec] = []
        if index in {10, 11, 13, 19}:
            seeds = [seeded_sum, distractor]
        elif index == 12:
            seeds = [broken_csv, distractor]
        elif index == 16:
            seeds = [seeded_sum, seeded_sum]
        tasks.append(
            BenchmarkTask(
                task_id=task_id,
                prompt=prompt,
                grader={"type": grader_type, "expected": expected},
                lifecycle_targets=targets,
                seeded_tools=seeds,
                tags=["synthetic", *targets],
            )
        )
    return tasks


def load_suite(path: str | Path) -> list[BenchmarkTask]:
    data = json.loads(Path(path).read_text())
    return [
        BenchmarkTask(
            task_id=item["task_id"],
            prompt=item["prompt"],
            grader=item["grader"],
            lifecycle_targets=item.get("lifecycle_targets", []),
            seeded_tools=[ToolSpec.from_dict(spec) for spec in item.get("seeded_tools", [])],
            tags=item.get("tags", []),
        )
        for item in data
    ]
