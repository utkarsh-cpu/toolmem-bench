from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .executor import ToolExecutor
from .registry import ToolRegistry
from .types import Runtime, ToolSpec, UpdateAction


class ToolMemoryAPI:
    """The exactly-three meta-tool interface exposed to benchmarked models."""

    TOOL_NAMES = ("create_tool", "find_tool", "update_tool")

    def __init__(self, registry: ToolRegistry, executor: ToolExecutor):
        self.registry = registry
        self.executor = executor
        self.tool_calls = 0
        self.search_calls = 0
        self.search_latency_ms = 0.0
        self.search_results_returned = 0
        self.search_candidates_reranked = 0
        self.estimated_retrieval_tokens = 0
        self.ephemeral_candidates = 0
        self.ephemeral_updates = 0

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return [
            {
                "name": "create_tool",
                "description": "Create a tool, optionally save it and/or run it.",
                "parameters": {
                    "type": "object",
                    "required": ["spec"],
                    "properties": {
                        "spec": {"type": "object"},
                        "save": {"type": "boolean", "default": True},
                        "run": {"type": "boolean", "default": False},
                        "input": {},
                    },
                },
            },
            {
                "name": "find_tool",
                "description": "Search saved tools, optionally running a selected match.",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "strategy": {"enum": ["lexical", "semantic", "hybrid"]},
                        "limit": {"type": "integer"},
                        "filters": {"type": "object"},
                        "run": {"type": "boolean", "default": False},
                        "selected_tool_id": {"type": "string"},
                        "input": {},
                    },
                },
            },
            {
                "name": "update_tool",
                "description": "Update, run, delete, restore, or edit metadata for a saved tool.",
                "parameters": {
                    "type": "object",
                    "required": ["tool_id", "action"],
                    "properties": {
                        "tool_id": {"type": "string"},
                        "action": {
                            "enum": [
                                "update", "run", "delete", "delete_version",
                                "restore", "metadata",
                            ]
                        },
                        "version": {"type": "integer"},
                        "changes": {"type": "object"},
                        "save": {"type": "boolean", "default": True},
                        "run": {"type": "boolean", "default": False},
                        "input": {},
                    },
                },
            },
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.TOOL_NAMES:
            raise ValueError(f"unknown meta-tool: {name}")
        return getattr(self, name)(**arguments)

    def _validate(self, spec: ToolSpec) -> None:
        if not spec.name.strip() or not spec.description.strip() or not spec.source.strip():
            raise ValueError("name, description, and source are required")
        if spec.runtime == Runtime.PYTHON:
            compile(spec.source, "<generated-tool>", "exec")
        elif spec.runtime == Runtime.SHELL:
            with tempfile.NamedTemporaryFile("w", suffix=".sh") as file:
                file.write(spec.source)
                file.flush()
                result = subprocess.run(["sh", "-n", file.name], capture_output=True, text=True)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
        if spec.entry_command and any("\x00" in part for part in spec.entry_command):
            raise ValueError("entry command contains null bytes")

    def _execute(
        self,
        spec: ToolSpec,
        tool_input: Any,
        tool_id: str | None,
        version: int | None,
        ephemeral: bool,
    ) -> dict[str, Any]:
        result = self.executor.execute(spec, tool_input)
        self.registry.record_execution(result, tool_id, version, ephemeral)
        return result.to_dict()

    def create_tool(
        self,
        spec: dict[str, Any],
        save: bool = True,
        run: bool = False,
        input: Any = None,
    ) -> dict[str, Any]:
        self.tool_calls += 1
        parsed = ToolSpec.from_dict(spec)
        self._validate(parsed)
        saved = self.registry.save(parsed) if save else None
        if not save:
            self.ephemeral_candidates += 1
        response: dict[str, Any] = {
            "storage": "saved" if save else "ephemeral",
            "tool": saved.to_dict(include_source=False) if saved else {
                "spec": {key: value for key, value in parsed.to_dict().items() if key != "source"}
            },
        }
        if run:
            response["execution"] = self._execute(
                parsed,
                input,
                saved.tool_id if saved else None,
                saved.version if saved else None,
                ephemeral=not save,
            )
        return response

    def find_tool(
        self,
        query: str,
        strategy: str = "hybrid",
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        run: bool = False,
        selected_tool_id: str | None = None,
        input: Any = None,
    ) -> dict[str, Any]:
        self.tool_calls += 1
        self.search_calls += 1
        filters = filters or {}
        started = time.perf_counter()
        results = self.registry.search(
            query=query,
            strategy=strategy,
            limit=limit,
            runtime=filters.get("runtime"),
            tags=filters.get("tags"),
            input_schema=filters.get("input_schema"),
            min_reliability=float(filters.get("min_reliability", 0)),
            max_latency_ms=filters.get("max_latency_ms"),
        )
        elapsed = (time.perf_counter() - started) * 1000
        summaries = [result.to_dict() for result in results]
        encoded = json.dumps(summaries, separators=(",", ":"))
        self.search_latency_ms += elapsed
        self.search_results_returned += len(results)
        self.search_candidates_reranked += len(results)
        self.estimated_retrieval_tokens += max(1, len(encoded) // 4)
        response: dict[str, Any] = {
            "strategy": strategy,
            "query": query,
            "search_latency_ms": elapsed,
            "matches": summaries,
            "estimated_context_tokens": max(1, len(encoded) // 4),
        }
        if run:
            if not results:
                response["execution"] = {"status": "error", "error_category": "no_match"}
                return response
            selected = (
                next((item for item in results if item.tool_id == selected_tool_id), None)
                if selected_tool_id
                else results[0]
            )
            if selected is None:
                response["execution"] = {
                    "status": "error",
                    "error_category": "selected_tool_not_in_results",
                }
                return response
            tool = self.registry.get(selected.tool_id, selected.version)
            response["selected_tool_id"] = selected.tool_id
            response["execution"] = self._execute(
                tool.spec, input, tool.tool_id, tool.version, ephemeral=False
            )
        return response

    def update_tool(
        self,
        tool_id: str,
        action: str,
        version: int | None = None,
        changes: dict[str, Any] | None = None,
        save: bool = True,
        run: bool = False,
        input: Any = None,
    ) -> dict[str, Any]:
        self.tool_calls += 1
        operation = UpdateAction(action)
        if operation == UpdateAction.DELETE:
            self.registry.delete(tool_id)
            return {"status": "deleted", "tool_id": tool_id}
        if operation == UpdateAction.DELETE_VERSION:
            if version is None:
                raise ValueError("version is required for delete_version")
            self.registry.delete(tool_id, version)
            return {"status": "deleted", "tool_id": tool_id, "version": version}
        if operation == UpdateAction.RESTORE:
            if version is None:
                raise ValueError("version is required for restore")
            restored = self.registry.restore(tool_id, version)
            response: dict[str, Any] = {
                "status": "restored",
                "tool": restored.to_dict(include_source=False),
            }
            if run:
                response["execution"] = self._execute(
                    restored.spec, input, restored.tool_id, restored.version, False
                )
            return response

        existing = self.registry.get(tool_id, version)
        if operation == UpdateAction.RUN:
            return {
                "tool": existing.to_dict(include_source=False),
                "execution": self._execute(
                    existing.spec, input, existing.tool_id, existing.version, False
                ),
            }
        changes = changes or {}
        spec_data = existing.spec.to_dict()
        if operation == UpdateAction.METADATA:
            allowed = {"name", "description", "input_schema", "output_schema", "tags"}
            unexpected = set(changes) - allowed
            if unexpected:
                raise ValueError(f"metadata action cannot change: {sorted(unexpected)}")
        spec_data.update(changes)
        updated = ToolSpec.from_dict(spec_data)
        self._validate(updated)
        saved = self.registry.save(updated, tool_id=tool_id) if save else None
        if not save:
            self.ephemeral_updates += 1
        response = {
            "storage": "saved" if save else "ephemeral",
            "tool": saved.to_dict(include_source=False) if saved else {
                "tool_id": tool_id,
                "base_version": existing.version,
                "spec": {key: value for key, value in updated.to_dict().items() if key != "source"},
            },
        }
        if run:
            response["execution"] = self._execute(
                updated,
                input,
                saved.tool_id if saved else tool_id,
                saved.version if saved else existing.version,
                ephemeral=not save,
            )
        return response

    def metrics(self) -> dict[str, Any]:
        return {
            **self.registry.metrics(),
            "meta_tool_calls": self.tool_calls,
            "search_calls": self.search_calls,
            "search_latency_ms": self.search_latency_ms,
            "search_results_returned": self.search_results_returned,
            "search_candidates_reranked": self.search_candidates_reranked,
            "average_search_latency_ms": (
                self.search_latency_ms / self.search_calls if self.search_calls else 0
            ),
            "estimated_retrieval_prompt_tokens": self.estimated_retrieval_tokens,
            "ephemeral_candidates": self.ephemeral_candidates,
            "ephemeral_updates": self.ephemeral_updates,
        }
