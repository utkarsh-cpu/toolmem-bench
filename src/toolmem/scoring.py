from __future__ import annotations

from typing import Any


WEIGHTS = {
    "correctness": 0.50,
    "lifecycle": 0.15,
    "retrieval": 0.10,
    "execution": 0.10,
    "memory": 0.10,
    "compliance": 0.05,
}


def score_episode(
    passed: bool,
    lifecycle_targets: list[str],
    trace: list[dict[str, Any]],
    metrics: dict[str, Any],
    constraints_ok: bool = True,
    persistent: bool = False,
) -> dict[str, float]:
    repair_task = "update" in lifecycle_targets
    calls = [entry for entry in trace if entry.get("type") == "tool_call"]
    names = [entry["name"] for entry in calls]
    observed: set[str] = set()
    if "create_tool" in names:
        observed.add("create")
    if "find_tool" in names:
        observed.add("find")
    if "update_tool" in names:
        observed.add("update")
    for entry in calls:
        arguments = entry.get("arguments", {})
        if arguments.get("save") is False:
            observed.add("ephemeral")
        if arguments.get("save") is True:
            observed.add("save")
        if arguments.get("run"):
            observed.add("run")
        if arguments.get("action") in {"delete", "delete_version"}:
            observed.add("delete")
    lifecycle = (
        len(set(lifecycle_targets) & observed) / len(set(lifecycle_targets))
        if lifecycle_targets
        else (1.0 if not calls else 0.5)
    )
    search_calls = metrics.get("search_calls", 0)
    retrieval = 1.0
    if "find" in lifecycle_targets:
        retrieval = 1.0 if search_calls and passed else 0.0
    elif search_calls:
        retrieval = 0.7
    execution_count = metrics.get("total_executions", 0)
    if repair_task:
        execution = (1.0 if passed else 0.2) / (
            1 + max(0, execution_count - 4) * 0.15
        )
    else:
        execution = (1.0 if passed else 0.2) / (
            1 + max(0, execution_count - 1) * 0.15
        )
    active = metrics.get("active_saved_tools", 0)
    unused = metrics.get("saved_tools_never_used", 0)
    duplicate_rate = metrics.get("exact_duplicate_rate", 0)
    if persistent:
        memory = max(0.0, 1 - 0.08 * unused - 0.35 * duplicate_rate)
    else:
        memory = max(
            0.0,
            1
            - 0.08 * unused
            - 0.35 * duplicate_rate
            - 0.02 * max(0, active - 5),
        )
    components = {
        "correctness": float(passed),
        "lifecycle": lifecycle,
        "retrieval": retrieval,
        "execution": min(1.0, execution),
        "memory": memory,
        "compliance": float(constraints_ok),
    }
    components["composite"] = sum(components[key] * WEIGHTS[key] for key in WEIGHTS)
    return {key: round(value, 6) for key, value in components.items()}
