from __future__ import annotations

import asyncio
import json
import numbers
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .api import ToolMemoryAPI
from .executor import DockerExecutor, LocalExecutor, ToolExecutor
from .models import ModelAdapter
from .registry import ToolRegistry, source_hash
from .scoring import score_episode
from .tasks import BenchmarkTask, grade_answer
from .types import EpisodeResult


SYSTEM_PROMPT = """You are being evaluated on task completion and management of your own tool memory.
You have exactly three meta-tools: create_tool, find_tool, and update_tool.
You decide what to save, keep ephemeral, update, reuse, and delete. Avoid unnecessary stored tools.
Generated tools read one JSON value from stdin and should print their result as JSON.
When the task is complete, return the final answer without calling another tool."""


class BenchmarkHarness:
    def __init__(
        self,
        model: ModelAdapter,
        root: str | Path,
        executor: ToolExecutor | None = None,
        persistent: bool = False,
        max_turns: int = 20,
        max_tool_calls: int = 30,
        wall_time_seconds: int = 600,
    ):
        self.model = model
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = executor or DockerExecutor()
        self.persistent = persistent
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.wall_time_seconds = wall_time_seconds
        self._persistent_registry: ToolRegistry | None = None
        if persistent:
            self._persistent_registry = ToolRegistry(self.root / "persistent-memory")

    def close(self) -> None:
        if self._persistent_registry:
            self._persistent_registry.close()

    def _registry_for(self, task: BenchmarkTask) -> ToolRegistry:
        if self._persistent_registry:
            registry = self._persistent_registry
        else:
            task_root = self.root / "episodes" / task.task_id
            if task_root.exists():
                shutil.rmtree(task_root)
            registry = ToolRegistry(task_root)
        for spec in task.seeded_tools:
            if not registry.has_tool(spec.name, source_hash(spec.source)):
                registry.save(spec)
        return registry

    async def run_task(self, task: BenchmarkTask) -> EpisodeResult:
        registry = self._registry_for(task)
        api = ToolMemoryAPI(registry, self.executor)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]
        trace: list[dict[str, Any]] = []
        started = time.monotonic()
        final_answer: Any = None
        constraints_ok = True
        total_usage: dict[str, int | float] = {}
        for turn in range(self.max_turns):
            if time.monotonic() - started > self.wall_time_seconds:
                constraints_ok = False
                trace.append({"type": "limit", "reason": "wall_time"})
                break
            response = await self.model.complete(messages, api.definitions())
            for key, value in response.usage.items():
                if isinstance(value, numbers.Real):
                    total_usage[key] = total_usage.get(key, 0) + value
            trace.append(
                {
                    "type": "model",
                    "turn": turn,
                    "content": response.content,
                    "tool_call": response.tool_call,
                    "usage": response.usage,
                }
            )
            if response.tool_call:
                if api.tool_calls >= self.max_tool_calls:
                    constraints_ok = False
                    trace.append({"type": "limit", "reason": "tool_calls"})
                    break
                name = response.tool_call["name"]
                arguments = response.tool_call.get("arguments", {})
                try:
                    result = api.dispatch(name, arguments)
                except Exception as error:
                    result = {
                        "status": "error",
                        "error_category": type(error).__name__,
                        "message": str(error),
                    }
                trace.append(
                    {"type": "tool_call", "name": name, "arguments": arguments, "result": result}
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": response.tool_call.get("id") or f"call-{turn}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": response.tool_call.get("id") or f"call-{turn}",
                        "content": json.dumps(result, default=str),
                    }
                )
                continue
            final_answer = response.final_answer if response.final_answer is not None else response.content
            break
        passed = grade_answer(
            final_answer,
            task.grader,
            context=api.registry.metrics(),
        )
        metrics = {
            **api.metrics(),
            "turns": len([entry for entry in trace if entry["type"] == "model"]),
            "wall_time_ms": (time.monotonic() - started) * 1000,
            "token_usage": total_usage,
        }
        metrics.update(registry.near_duplicate_stats())
        metrics["tool_calls_per_successful_task"] = (
            metrics["meta_tool_calls"] if passed else None
        )
        metrics["unnecessary_tool_calls"] = (
            metrics["meta_tool_calls"] if not task.lifecycle_targets else 0
        )
        metrics["repair_success"] = (
            passed if "update" in task.lifecycle_targets else None
        )
        score = score_episode(
            passed,
            task.lifecycle_targets,
            trace,
            metrics,
            constraints_ok,
            persistent=self.persistent,
        )
        result = EpisodeResult(task.task_id, final_answer, passed, score, metrics, trace)
        if not self.persistent:
            registry.close()
        return result

    async def run_suite(self, tasks: list[BenchmarkTask]) -> list[EpisodeResult]:
        return [await self.run_task(task) for task in tasks]
