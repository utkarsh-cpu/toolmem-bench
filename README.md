# ToolMem Bench

ToolMem Bench is a research benchmark for measuring whether an agent can build and
manage its own reusable tool library. The model receives exactly three meta-tools:

- `create_tool`: create, optionally save, and optionally execute a tool.
- `find_tool`: search saved memory, optionally execute a selected result.
- `update_tool`: update, run, delete, restore, or edit a saved tool.

The model controls persistence. The benchmark never deduplicates or evicts valid
tools on its behalf; inefficient storage is measured and penalized.

## Quick start

Python 3.11 or later is required. Docker is recommended for untrusted model output.

```bash
uv sync
uv run toolmem
uv run toolmem list-tasks
uv run toolmem run --task no-tool-needed --executor local
```

Running `toolmem` without a command opens a guided terminal interface.

The `fake` provider is a deterministic harness smoke test and directly returns the
known expected answer. It is not a model baseline. To run a real model:

```bash
export OPENAI_API_KEY=...
uv run toolmem run \
  --model-provider openai-compatible \
  --model your-model \
  --endpoint https://api.openai.com/v1 \
  --executor docker \
  --memory persistent
```

OpenRouter is available as a first-class provider:

```bash
export OPENROUTER_API_KEY=...
uv run toolmem run \
  --model-provider openrouter \
  --model openai/gpt-5.2 \
  --executor docker \
  --memory persistent
```

Optional `--referer` and `--app-title` values add OpenRouter app attribution headers.
The interactive UI prompts for these values and reads `OPENROUTER_API_KEY`,
`OPENROUTER_MODEL`, `OPENROUTER_HTTP_REFERER`, and `OPENROUTER_APP_TITLE`.

Use `--offline` to disable networking in generated-tool containers. Local execution
exists for tests and trusted development only.

## Commands

```bash
# Show the 20 starter tasks
uv run toolmem list-tasks

# Open the guided UI
uv run toolmem ui

# Run a complete fresh-memory suite
uv run toolmem run --model-provider openai-compatible --model MODEL

# Retain model-saved tools across tasks in a run
uv run toolmem run --memory persistent --model-provider openai-compatible --model MODEL

# Compare retrieval strategies
uv run toolmem compare-retrieval --query "sum a numbers array"

# Inspect memory metrics for a registry
uv run toolmem memory benchmark-results/run-1/persistent-memory
```

Each run writes:

- `episodes.jsonl`: full episode traces and metrics.
- `aggregate.json`: run configuration and aggregate scores.
- `episodes.csv`: compact comparison table.

## Tool input/output convention

Generated tools receive one JSON value on standard input and should print a JSON
result. Python, JavaScript, and shell have default entry commands. A tool can provide
an explicit `entry_command`, using `{source}` as the generated source-file placeholder.

Saved updates are immutable versions. Ephemeral creations and updates may execute
without changing persistent memory. Deletion is an action of `update_tool`, preserving
the exactly-three-tools constraint.

## Retrieval

The registry supports:

- SQLite FTS5 lexical search.
- Deterministic local semantic vectors with no external embedding dependency.
- Hybrid reranking using lexical relevance, semantic similarity, schema compatibility,
  execution reliability, and latency.

Search returns compact summaries and never exposes source code. Source is loaded only
when a result is executed or updated.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The test suite uses the local executor and must never receive untrusted model code.
