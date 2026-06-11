from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .types import ExecutionResult, Runtime, ToolSpec


class ToolExecutor(ABC):
    @abstractmethod
    def execute(self, spec: ToolSpec, tool_input: Any) -> ExecutionResult:
        raise NotImplementedError


def default_command(spec: ToolSpec, filename: str) -> list[str]:
    if spec.entry_command:
        return [part.replace("{source}", filename) for part in spec.entry_command]
    return {
        Runtime.PYTHON: ["python", filename],
        Runtime.JAVASCRIPT: ["node", filename],
        Runtime.SHELL: ["sh", filename],
    }.get(spec.runtime, [])


def source_filename(runtime: Runtime) -> str:
    return {
        Runtime.PYTHON: "tool.py",
        Runtime.JAVASCRIPT: "tool.js",
        Runtime.SHELL: "tool.sh",
        Runtime.CUSTOM: "tool",
    }[runtime]


def parse_payload(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return text


class LocalExecutor(ToolExecutor):
    """Test executor. Do not use with untrusted model output."""

    def __init__(self, timeout_seconds: int = 30, output_limit_bytes: int = 1_000_000):
        self.timeout_seconds = timeout_seconds
        self.output_limit_bytes = output_limit_bytes

    def execute(self, spec: ToolSpec, tool_input: Any) -> ExecutionResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="toolmem-") as directory:
            filename = source_filename(spec.runtime)
            path = Path(directory) / filename
            path.write_text(spec.source)
            if spec.runtime == Runtime.SHELL:
                path.chmod(0o700)
            command = default_command(spec, filename)
            if spec.runtime == Runtime.PYTHON and not spec.entry_command:
                command[0] = sys.executable
            if not command:
                return ExecutionResult(status="error", error_category="invalid_command")
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            try:
                process = subprocess.run(
                    command,
                    cwd=directory,
                    input=json.dumps(tool_input),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env={**os.environ, "TOOLMEM_INPUT": json.dumps(tool_input)},
                )
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                stdout = process.stdout[: self.output_limit_bytes]
                stderr = process.stderr[: self.output_limit_bytes]
                status = "success" if process.returncode == 0 else "error"
                return ExecutionResult(
                    status=status,
                    payload=parse_payload(stdout),
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
                    max_rss_bytes=int(after.ru_maxrss * 1024),
                    error_category=None if status == "success" else "nonzero_exit",
                )
            except subprocess.TimeoutExpired as error:
                return ExecutionResult(
                    status="error",
                    stdout=(error.stdout or "")[: self.output_limit_bytes],
                    stderr=(error.stderr or "")[: self.output_limit_bytes],
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_category="timeout",
                )
            except FileNotFoundError as error:
                return ExecutionResult(
                    status="error",
                    stderr=str(error),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_category="runtime_unavailable",
                )


class DockerExecutor(ToolExecutor):
    def __init__(
        self,
        timeout_seconds: int = 60,
        cpus: float = 2,
        memory: str = "2g",
        network: bool = True,
        output_limit_bytes: int = 1_000_000,
    ):
        self.timeout_seconds = timeout_seconds
        self.cpus = cpus
        self.memory = memory
        self.network = network
        self.output_limit_bytes = output_limit_bytes

    def execute(self, spec: ToolSpec, tool_input: Any) -> ExecutionResult:
        if not shutil.which("docker"):
            return ExecutionResult(
                status="error",
                stderr="Docker executable is unavailable",
                error_category="docker_unavailable",
            )
        image = {
            Runtime.PYTHON: "python:3.13-alpine",
            Runtime.JAVASCRIPT: "node:22-alpine",
            Runtime.SHELL: "alpine:3.21",
            Runtime.CUSTOM: "alpine:3.21",
        }[spec.runtime]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="toolmem-docker-") as directory:
            filename = source_filename(spec.runtime)
            (Path(directory) / filename).write_text(spec.source)
            command = default_command(spec, f"/workspace/{filename}")
            docker_command = [
                "docker", "run", "--rm", "--read-only",
                "--cpus", str(self.cpus), "--memory", self.memory,
                "--pids-limit", "128", "--security-opt", "no-new-privileges",
                "-v", f"{directory}:/workspace:ro", "-w", "/workspace",
                "-i",
            ]
            if not self.network:
                docker_command += ["--network", "none"]
            docker_command += [image, *command]
            try:
                process = subprocess.run(
                    docker_command,
                    input=json.dumps(tool_input),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                stdout = process.stdout[: self.output_limit_bytes]
                stderr = process.stderr[: self.output_limit_bytes]
                status = "success" if process.returncode == 0 else "error"
                return ExecutionResult(
                    status=status,
                    payload=parse_payload(stdout),
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=process.returncode,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_category=None if status == "success" else "container_error",
                )
            except subprocess.TimeoutExpired as error:
                return ExecutionResult(
                    status="error",
                    stdout=(error.stdout or "")[: self.output_limit_bytes],
                    stderr=(error.stderr or "")[: self.output_limit_bytes],
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_category="timeout",
                )
