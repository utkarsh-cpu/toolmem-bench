from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .executor import DockerExecutor, LocalExecutor
from .harness import BenchmarkHarness
from .models import DeterministicFakeModel, OpenAICompatibleAdapter, OpenRouterAdapter
from .registry import ToolRegistry
from .reporting import aggregate, write_reports
from .tasks import BenchmarkTask, load_suite, starter_suite
from .types import ModelResponse


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="toolmem", description="Agentic tool-memory benchmark")
    subcommands = root.add_subparsers(dest="command")

    subcommands.add_parser("ui", help="Open the guided terminal interface")

    list_tasks = subcommands.add_parser("list-tasks", help="List starter benchmark tasks")
    list_tasks.add_argument("--suite")

    run = subcommands.add_parser("run", help="Run one task or the full suite")
    run.add_argument("--task")
    run.add_argument("--suite")
    run.add_argument("--output", default="benchmark-results")
    run.add_argument("--memory", choices=["fresh", "persistent"], default="fresh")
    run.add_argument("--executor", choices=["docker", "local"], default="docker")
    run.add_argument("--offline", action="store_true")
    run.add_argument(
        "--model-provider",
        choices=["openrouter", "openai-compatible", "fake"],
        default="fake",
    )
    run.add_argument("--endpoint", default="")
    run.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    run.add_argument("--api-key", default="")
    run.add_argument("--referer", default=os.getenv("OPENROUTER_HTTP_REFERER", ""))
    run.add_argument(
        "--app-title",
        default=os.getenv("OPENROUTER_APP_TITLE", "ToolMem Bench"),
    )
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--repetitions", type=int, default=1)

    compare = subcommands.add_parser(
        "compare-retrieval", help="Compare lexical, semantic, and hybrid search"
    )
    compare.add_argument("--query", required=True)
    compare.add_argument("--limit", type=int, default=5)
    compare.add_argument("--registry")

    inspect = subcommands.add_parser("memory", help="Inspect a saved tool registry")
    inspect.add_argument("registry")
    return root


def suite_from_args(path: str | None) -> list[BenchmarkTask]:
    return load_suite(path) if path else starter_suite()


def fake_model_for(task: BenchmarkTask) -> DeterministicFakeModel:
    # This is a harness smoke-test adapter, not a benchmark baseline.
    expected = task.grader.get("expected")
    answer = json.dumps(expected) if isinstance(expected, (dict, list, bool)) else str(expected)
    return DeterministicFakeModel([ModelResponse(content=answer, final_answer=answer)])


def make_executor(name: str, offline: bool):
    if name == "local":
        return LocalExecutor()
    return DockerExecutor(network=not offline)


def make_model(args: argparse.Namespace):
    if args.model_provider == "openrouter":
        model = args.model or os.getenv("OPENROUTER_MODEL", "")
        if not model:
            raise SystemExit("--model is required for OpenRouter")
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise SystemExit("--api-key or OPENROUTER_API_KEY is required for OpenRouter")
        return OpenRouterAdapter(
            model=model,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            referer=args.referer,
            app_title=args.app_title,
            endpoint=args.endpoint or OpenRouterAdapter.DEFAULT_ENDPOINT,
        )
    if args.model_provider == "openai-compatible":
        if not args.model:
            raise SystemExit("--model is required for an OpenAI-compatible provider")
        return OpenAICompatibleAdapter(
            args.endpoint or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            args.model,
            args.api_key or os.getenv("OPENAI_API_KEY", ""),
            args.temperature,
            args.max_tokens,
        )
    return None


async def run_command(args: argparse.Namespace) -> int:
    tasks = suite_from_args(args.suite)
    if args.task:
        tasks = [task for task in tasks if task.task_id == args.task]
        if not tasks:
            raise SystemExit(f"unknown task: {args.task}")
    all_results = []
    for repetition in range(args.repetitions):
        run_root = Path(args.output) / f"run-{repetition + 1}"
        if args.model_provider != "fake":
            model = make_model(args)
            harness = BenchmarkHarness(
                model,
                run_root,
                executor=make_executor(args.executor, args.offline),
                persistent=args.memory == "persistent",
            )
            results = []
            for index, task in enumerate(tasks, 1):
                show_progress(index, len(tasks), task.task_id)
                results.append(await harness.run_task(task))
            harness.close()
        else:
            results = []
            # A fresh scripted fake is created per task so smoke tests remain deterministic.
            for index, task in enumerate(tasks, 1):
                show_progress(index, len(tasks), task.task_id)
                harness = BenchmarkHarness(
                    fake_model_for(task),
                    run_root,
                    executor=make_executor(args.executor, args.offline),
                    persistent=args.memory == "persistent",
                )
                results.append(await harness.run_task(task))
                harness.close()
        all_results.extend(results)
    paths = write_reports(
        all_results,
        args.output,
        {
            "model_provider": args.model_provider,
            "model": args.model,
            "memory": args.memory,
            "executor": args.executor,
            "network": not args.offline,
            "repetitions": args.repetitions,
        },
    )
    summary = aggregate(all_results)
    if sys.stdout.isatty():
        render_summary(summary, paths["aggregate"])
    else:
        print(json.dumps(summary, indent=2))
        print(f"Reports: {paths['aggregate']}")
    return 0


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
DIM = "\033[2m"


def heading(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{text}{RESET}")


def show_progress(index: int, total: int, task_id: str) -> None:
    if sys.stdout.isatty():
        print(f"{CYAN}[{index:>2}/{total}]{RESET} Running {task_id}...")


def render_summary(summary: dict[str, Any], report_path: Path) -> None:
    heading("Benchmark complete")
    passed = summary.get("passed", 0)
    episodes = summary.get("episodes", 0)
    success = summary.get("success_rate", 0) * 100
    print(f"{GREEN}{BOLD}{passed}/{episodes} tasks passed{RESET}  {success:.1f}% success")
    print()
    print(f"{BOLD}Scores{RESET}")
    for name, value in summary.get("scores", {}).items():
        label = name.replace("_", " ").title()
        print(f"  {label:<14} {value:>7.3f}")
    memory = summary.get("final_memory", {})
    print(f"\n{BOLD}Final tool memory{RESET}")
    print(f"  Saved tools     {memory.get('active_saved_tools', 0)}")
    print(f"  Tool versions   {memory.get('total_versions', 0)}")
    print(f"  Storage bytes   {memory.get('total_storage_bytes', 0)}")
    print(f"  Duplicate rate  {memory.get('exact_duplicate_rate', 0) * 100:.1f}%")
    print(f"\n{DIM}Detailed report: {report_path}{RESET}")


def choose(label: str, options: list[tuple[str, str]], default: int = 1) -> str:
    print(f"\n{BOLD}{label}{RESET}")
    for index, (_, description) in enumerate(options, 1):
        marker = " (default)" if index == default else ""
        print(f"  {index}. {description}{marker}")
    while True:
        value = input(f"> [{default}] ").strip()
        if not value:
            return options[default - 1][0]
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        print("Enter one of the displayed numbers.")


def ask(label: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    prompt = f"{label}{suffix}: "
    value = getpass.getpass(prompt) if secret else input(prompt)
    return value.strip() or default


def interactive_args() -> argparse.Namespace | None:
    print(f"{BOLD}{CYAN}ToolMem Bench{RESET}")
    print(f"{DIM}Evaluate how models create, reuse, update, and manage tool memory.{RESET}")
    action = choose(
        "What would you like to do?",
        [
            ("run", "Run a benchmark"),
            ("list", "Browse benchmark tasks"),
            ("search", "Compare tool-search strategies"),
            ("memory", "Inspect a saved tool memory"),
            ("quit", "Exit"),
        ],
    )
    if action == "quit":
        return None
    if action == "list":
        return argparse.Namespace(command="list-tasks", suite=None)
    if action == "search":
        return argparse.Namespace(
            command="compare-retrieval",
            query=ask("Search query", "sum a numbers array"),
            limit=int(ask("Number of results", "5")),
            registry=ask("Registry path (leave blank for examples)", "") or None,
        )
    if action == "memory":
        return argparse.Namespace(command="memory", registry=ask("Registry path"))

    provider = choose(
        "Model provider",
        [
            ("openrouter", "OpenRouter"),
            ("openai-compatible", "OpenAI-compatible endpoint"),
            ("fake", "Deterministic smoke-test model"),
        ],
    )
    model = ""
    api_key = ""
    endpoint = ""
    referer = ""
    app_title = "ToolMem Bench"
    if provider == "openrouter":
        model = ask("OpenRouter model ID", os.getenv("OPENROUTER_MODEL", ""))
        api_key = os.getenv("OPENROUTER_API_KEY", "") or ask(
            "OpenRouter API key", secret=True
        )
        referer = ask(
            "App URL for OpenRouter attribution (optional)",
            os.getenv("OPENROUTER_HTTP_REFERER", ""),
        )
        app_title = ask(
            "App title",
            os.getenv("OPENROUTER_APP_TITLE", "ToolMem Bench"),
        )
    elif provider == "openai-compatible":
        endpoint = ask(
            "API base URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        model = ask("Model", os.getenv("OPENAI_MODEL", ""))
        api_key = os.getenv("OPENAI_API_KEY", "") or ask("API key", secret=True)
    scope = choose(
        "Tasks",
        [("suite", "Run all 20 tasks"), ("single", "Run one task")],
    )
    task = None
    if scope == "single":
        print("\nAvailable task IDs:")
        print("  " + ", ".join(task.task_id for task in starter_suite()))
        task = ask("Task ID", "no-tool-needed")
    memory = choose(
        "Tool memory mode",
        [("fresh", "Fresh memory for every task"), ("persistent", "Reuse memory across tasks")],
    )
    executor = choose(
        "Tool execution",
        [
            ("docker", "Docker sandbox (recommended for model-generated code)"),
            ("local", "Local process (trusted testing only)"),
        ],
    )
    network = choose(
        "Generated-tool network access",
        [("online", "Allow network access"), ("offline", "Disable network access")],
    )
    return argparse.Namespace(
        command="run",
        task=task,
        suite=None,
        output=ask("Results directory", "benchmark-results"),
        memory=memory,
        executor=executor,
        offline=network == "offline",
        model_provider=provider,
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        referer=referer,
        app_title=app_title,
        temperature=float(ask("Temperature", "0")),
        max_tokens=int(ask("Maximum response tokens", "4096")),
        repetitions=int(ask("Repetitions", "1")),
    )


def compare_retrieval(args: argparse.Namespace) -> int:
    temporary = None
    if args.registry:
        registry = ToolRegistry(args.registry)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="toolmem-search-")
        registry = ToolRegistry(temporary.name)
        for task in starter_suite():
            for spec in task.seeded_tools:
                registry.save(spec)
    report: dict[str, Any] = {}
    for strategy in ("lexical", "semantic", "hybrid"):
        report[strategy] = [
            result.to_dict()
            for result in registry.search(args.query, strategy=strategy, limit=args.limit)
        ]
    print(json.dumps(report, indent=2))
    registry.close()
    if temporary:
        temporary.cleanup()
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.command in {None, "ui"}:
        if not sys.stdin.isatty():
            parser().print_help()
            return 2
        try:
            args = interactive_args()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0
        if args is None:
            return 0
    if args.command == "list-tasks":
        for task in suite_from_args(args.suite):
            print(f"{task.task_id}\t{','.join(task.tags)}\t{task.prompt}")
        return 0
    if args.command == "run":
        return asyncio.run(run_command(args))
    if args.command == "compare-retrieval":
        return compare_retrieval(args)
    if args.command == "memory":
        registry = ToolRegistry(args.registry)
        print(json.dumps(registry.metrics(), indent=2))
        registry.close()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
