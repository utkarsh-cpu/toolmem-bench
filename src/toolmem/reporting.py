from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .types import EpisodeResult


def aggregate(results: list[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {"episodes": 0}
    score_keys = results[0].score
    return {
        "episodes": len(results),
        "passed": sum(result.passed for result in results),
        "success_rate": mean(float(result.passed) for result in results),
        "scores": {
            key: mean(result.score[key] for result in results)
            for key in score_keys
        },
        "average_meta_tool_calls": mean(
            result.metrics.get("meta_tool_calls", 0) for result in results
        ),
        "average_executions": mean(
            result.metrics.get("total_executions", 0) for result in results
        ),
        "final_memory": {
            key: results[-1].metrics.get(key)
            for key in (
                "active_saved_tools",
                "total_versions",
                "total_storage_bytes",
                "exact_duplicate_rate",
                "near_duplicate_rate",
                "saved_tools_never_used",
                "useful_tool_density",
                "successful_reuse_rate",
                "storage_bytes_per_successful_reuse",
            )
        },
    }


def write_reports(
    results: list[EpisodeResult],
    output_dir: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "episodes.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(result.to_dict(), default=str) + "\n" for result in results)
    )
    aggregate_path = output / "aggregate.json"
    summary = {"run": run_metadata or {}, **aggregate(results)}
    aggregate_path.write_text(json.dumps(summary, indent=2, default=str))
    csv_path = output / "episodes.csv"
    rows = [
        {
            "task_id": result.task_id,
            "passed": result.passed,
            "composite": result.score["composite"],
            "correctness": result.score["correctness"],
            "lifecycle": result.score["lifecycle"],
            "retrieval": result.score["retrieval"],
            "execution": result.score["execution"],
            "memory": result.score["memory"],
            "meta_tool_calls": result.metrics.get("meta_tool_calls", 0),
            "executions": result.metrics.get("total_executions", 0),
            "active_saved_tools": result.metrics.get("active_saved_tools", 0),
            "storage_bytes": result.metrics.get("total_storage_bytes", 0),
        }
        for result in results
    ]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)
    return {"jsonl": jsonl_path, "aggregate": aggregate_path, "csv": csv_path}
