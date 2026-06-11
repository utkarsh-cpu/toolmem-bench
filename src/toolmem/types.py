from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Runtime(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SHELL = "shell"
    CUSTOM = "custom"


class ToolState(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"


class UpdateAction(StrEnum):
    UPDATE = "update"
    RUN = "run"
    DELETE = "delete"
    DELETE_VERSION = "delete_version"
    RESTORE = "restore"
    METADATA = "metadata"


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    source: str
    runtime: Runtime
    entry_command: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolSpec":
        data = dict(value)
        data["runtime"] = Runtime(data["runtime"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["runtime"] = self.runtime.value
        return result


@dataclass(slots=True)
class ToolVersion:
    tool_id: str
    version: int
    spec: ToolSpec
    source_hash: str
    state: ToolState
    created_at: str
    restored_from: int | None = None

    def to_dict(self, include_source: bool = True) -> dict[str, Any]:
        spec = self.spec.to_dict()
        if not include_source:
            spec.pop("source", None)
        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "spec": spec,
            "source_hash": self.source_hash,
            "state": self.state.value,
            "created_at": self.created_at,
            "restored_from": self.restored_from,
        }


@dataclass(slots=True)
class ExecutionResult:
    status: str
    payload: Any = None
    files: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0
    cpu_seconds: float | None = None
    max_rss_bytes: int | None = None
    error_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    tool_id: str
    version: int
    name: str
    description: str
    runtime: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    tags: list[str]
    score: float
    lexical_score: float
    semantic_score: float
    success_rate: float
    average_duration_ms: float
    usage_count: int
    last_used_at: str | None
    source_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelResponse:
    content: str = ""
    tool_call: dict[str, Any] | None = None
    final_answer: Any = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeResult:
    task_id: str
    final_answer: Any
    passed: bool
    score: dict[str, float]
    metrics: dict[str, Any]
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
