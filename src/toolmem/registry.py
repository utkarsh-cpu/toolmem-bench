from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .types import ExecutionResult, Runtime, SearchResult, ToolSpec, ToolState, ToolVersion

TOKEN_RE = re.compile(r"[a-z0-9_]+")
VECTOR_SIZE = 128


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def embed(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    for token, count in Counter(TOKEN_RE.findall(text.lower())).items():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % VECTOR_SIZE
        sign = 1 if digest[0] & 1 else -1
        vector[index] += sign * (1 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class ToolRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self.db_path = self.root / "registry.sqlite3"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tools (
                tool_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                current_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS versions (
                tool_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                description TEXT NOT NULL,
                runtime TEXT NOT NULL,
                entry_command TEXT NOT NULL,
                dependencies TEXT NOT NULL,
                input_schema TEXT NOT NULL,
                output_schema TEXT NOT NULL,
                tags TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                source_bytes INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                restored_from INTEGER,
                PRIMARY KEY (tool_id, version),
                FOREIGN KEY (tool_id) REFERENCES tools(tool_id)
            );
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                tool_id TEXT,
                version INTEGER,
                ephemeral INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                cpu_seconds REAL,
                max_rss_bytes INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                tool_id TEXT,
                version INTEGER,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS versions_fts USING fts5(
                tool_id UNINDEXED, version UNINDEXED, name, description, tags
            );
            """
        )
        self.connection.commit()

    def _write_artifact(self, digest: str, source: str) -> None:
        path = self.artifacts / digest
        if not path.exists():
            path.write_text(source)

    def _event(
        self, kind: str, tool_id: str | None, version: int | None, data: dict[str, Any]
    ) -> None:
        self.connection.execute(
            "INSERT INTO events(kind, tool_id, version, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (kind, tool_id, version, json.dumps(data, sort_keys=True), utcnow()),
        )

    def save(self, spec: ToolSpec, tool_id: str | None = None, restored_from: int | None = None) -> ToolVersion:
        tool_id = tool_id or str(uuid.uuid4())
        row = self.connection.execute(
            "SELECT current_version, state FROM tools WHERE tool_id = ?", (tool_id,)
        ).fetchone()
        version = int(row["current_version"]) + 1 if row else 1
        digest = source_hash(spec.source)
        self._write_artifact(digest, spec.source)
        text = " ".join([spec.name, spec.description, *spec.tags, json.dumps(spec.input_schema)])
        created = utcnow()
        if row:
            self.connection.execute(
                "UPDATE tools SET name = ?, state = 'active', current_version = ?, deleted_at = NULL WHERE tool_id = ?",
                (spec.name, version, tool_id),
            )
        else:
            self.connection.execute(
                "INSERT INTO tools(tool_id, name, state, current_version, created_at) VALUES (?, ?, 'active', ?, ?)",
                (tool_id, spec.name, version, created),
            )
        self.connection.execute(
            """
            INSERT INTO versions(
                tool_id, version, description, runtime, entry_command, dependencies,
                input_schema, output_schema, tags, source_hash, source_bytes,
                embedding, state, created_at, restored_from
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                tool_id,
                version,
                spec.description,
                spec.runtime.value,
                json.dumps(spec.entry_command),
                json.dumps(spec.dependencies),
                json.dumps(spec.input_schema, sort_keys=True),
                json.dumps(spec.output_schema, sort_keys=True),
                json.dumps(spec.tags),
                digest,
                len(spec.source.encode()),
                json.dumps(embed(text)),
                created,
                restored_from,
            ),
        )
        self.connection.execute(
            "INSERT INTO versions_fts(tool_id, version, name, description, tags) VALUES (?, ?, ?, ?, ?)",
            (tool_id, version, spec.name, spec.description, " ".join(spec.tags)),
        )
        self._event("save", tool_id, version, {"source_hash": digest})
        self.connection.commit()
        return self.get(tool_id, version)

    def get(self, tool_id: str, version: int | None = None, include_deleted: bool = False) -> ToolVersion:
        tool = self.connection.execute(
            "SELECT * FROM tools WHERE tool_id = ?", (tool_id,)
        ).fetchone()
        if not tool or (tool["state"] == "deleted" and not include_deleted):
            raise KeyError(f"tool not found: {tool_id}")
        target = version or int(tool["current_version"])
        row = self.connection.execute(
            "SELECT * FROM versions WHERE tool_id = ? AND version = ?", (tool_id, target)
        ).fetchone()
        if not row or (row["state"] == "deleted" and not include_deleted):
            raise KeyError(f"tool version not found: {tool_id}@{target}")
        spec = ToolSpec(
            name=tool["name"],
            description=row["description"],
            source=(self.artifacts / row["source_hash"]).read_text(),
            runtime=Runtime(row["runtime"]),
            entry_command=json.loads(row["entry_command"]),
            dependencies=json.loads(row["dependencies"]),
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]),
            tags=json.loads(row["tags"]),
        )
        return ToolVersion(
            tool_id=tool_id,
            version=target,
            spec=spec,
            source_hash=row["source_hash"],
            state=ToolState(row["state"]),
            created_at=row["created_at"],
            restored_from=row["restored_from"],
        )

    def delete(self, tool_id: str, version: int | None = None) -> None:
        if version is None:
            changed = self.connection.execute(
                "UPDATE tools SET state = 'deleted', deleted_at = ? WHERE tool_id = ?",
                (utcnow(), tool_id),
            ).rowcount
            if not changed:
                raise KeyError(f"tool not found: {tool_id}")
            self._event("delete_tool", tool_id, None, {})
        else:
            changed = self.connection.execute(
                "UPDATE versions SET state = 'deleted' WHERE tool_id = ? AND version = ?",
                (tool_id, version),
            ).rowcount
            if not changed:
                raise KeyError(f"tool version not found: {tool_id}@{version}")
            current = self.connection.execute(
                "SELECT current_version FROM tools WHERE tool_id = ?", (tool_id,)
            ).fetchone()
            if current and int(current["current_version"]) == version:
                replacement = self.connection.execute(
                    "SELECT MAX(version) AS version FROM versions WHERE tool_id = ? AND state = 'active'",
                    (tool_id,),
                ).fetchone()["version"]
                if replacement is None:
                    self.connection.execute(
                        "UPDATE tools SET state = 'deleted', deleted_at = ? WHERE tool_id = ?",
                        (utcnow(), tool_id),
                    )
                else:
                    self.connection.execute(
                        "UPDATE tools SET current_version = ? WHERE tool_id = ?",
                        (replacement, tool_id),
                    )
            self._event("delete_version", tool_id, version, {})
        self.connection.commit()

    def restore(self, tool_id: str, version: int) -> ToolVersion:
        original = self.get(tool_id, version, include_deleted=True)
        return self.save(original.spec, tool_id=tool_id, restored_from=version)

    def record_execution(
        self,
        result: ExecutionResult,
        tool_id: str | None,
        version: int | None,
        ephemeral: bool,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO executions(
                execution_id, tool_id, version, ephemeral, status, duration_ms,
                cpu_seconds, max_rss_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                tool_id,
                version,
                int(ephemeral),
                result.status,
                result.duration_ms,
                result.cpu_seconds,
                result.max_rss_bytes,
                utcnow(),
            ),
        )
        self.connection.commit()

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        limit: int = 5,
        runtime: str | None = None,
        tags: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
        min_reliability: float = 0,
        max_latency_ms: float | None = None,
    ) -> list[SearchResult]:
        if strategy not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("strategy must be lexical, semantic, or hybrid")
        rows = self.connection.execute(
            """
            SELECT t.tool_id, t.name, t.current_version, v.*
            FROM tools t JOIN versions v
              ON v.tool_id = t.tool_id AND v.version = t.current_version
            WHERE t.state = 'active' AND v.state = 'active'
            """
        ).fetchall()
        lexical: dict[tuple[str, int], float] = {}
        if query.strip():
            fts_query = " OR ".join(TOKEN_RE.findall(query.lower())) or '""'
            try:
                hits = self.connection.execute(
                    """
                    SELECT tool_id, CAST(version AS INTEGER) version, bm25(versions_fts) rank
                    FROM versions_fts WHERE versions_fts MATCH ?
                    """,
                    (fts_query,),
                ).fetchall()
                if hits:
                    best = min(float(hit["rank"]) for hit in hits)
                    worst = max(float(hit["rank"]) for hit in hits)
                    span = worst - best or 1.0
                    lexical = {
                        (hit["tool_id"], int(hit["version"])): 1 - (float(hit["rank"]) - best) / span
                        for hit in hits
                    }
            except sqlite3.OperationalError:
                lexical = {}
        query_vector = embed(query)
        requested_keys = set((input_schema or {}).get("properties", {}))
        results: list[SearchResult] = []
        for row in rows:
            row_tags = json.loads(row["tags"])
            if runtime and row["runtime"] != runtime:
                continue
            if tags and not set(tags).issubset(row_tags):
                continue
            key = (row["tool_id"], int(row["version"]))
            lex = lexical.get(key, 0.0)
            sem = max(0.0, cosine(query_vector, json.loads(row["embedding"])))
            stats = self.connection.execute(
                """
                SELECT COUNT(*) count,
                       AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) success_rate,
                       AVG(duration_ms) avg_ms,
                       MAX(created_at) last_used
                FROM executions WHERE tool_id = ? AND version = ?
                """,
                key,
            ).fetchone()
            count = int(stats["count"])
            success = float(stats["success_rate"] or 0)
            avg_ms = float(stats["avg_ms"] or 0)
            if success < min_reliability or (max_latency_ms is not None and avg_ms > max_latency_ms):
                continue
            schema = json.loads(row["input_schema"])
            available_keys = set(schema.get("properties", {}))
            schema_score = (
                len(requested_keys & available_keys) / len(requested_keys)
                if requested_keys
                else 1.0
            )
            if strategy == "lexical":
                score = lex
            elif strategy == "semantic":
                score = sem
            else:
                reliability = success if count else 0.5
                latency = 1 / (1 + avg_ms / 1000) if count else 0.5
                score = 0.35 * lex + 0.35 * sem + 0.15 * schema_score + 0.1 * reliability + 0.05 * latency
            results.append(
                SearchResult(
                    tool_id=row["tool_id"],
                    version=int(row["version"]),
                    name=row["name"],
                    description=row["description"],
                    runtime=row["runtime"],
                    input_schema=schema,
                    output_schema=json.loads(row["output_schema"]),
                    tags=row_tags,
                    score=round(score, 6),
                    lexical_score=round(lex, 6),
                    semantic_score=round(sem, 6),
                    success_rate=round(success, 4),
                    average_duration_ms=round(avg_ms, 3),
                    usage_count=count,
                    last_used_at=stats["last_used"],
                    source_bytes=int(row["source_bytes"]),
                )
            )
        return sorted(results, key=lambda item: (-item.score, item.name))[:limit]

    def metrics(self) -> dict[str, Any]:
        active = self.connection.execute(
            "SELECT COUNT(*) count FROM tools WHERE state = 'active'"
        ).fetchone()["count"]
        deleted = self.connection.execute(
            "SELECT COUNT(*) count FROM tools WHERE state = 'deleted'"
        ).fetchone()["count"]
        versions = self.connection.execute("SELECT COUNT(*) count FROM versions").fetchone()["count"]
        deleted_versions = self.connection.execute(
            "SELECT COUNT(*) count FROM versions WHERE state = 'deleted'"
        ).fetchone()["count"]
        source = self.connection.execute(
            "SELECT COALESCE(SUM(source_bytes), 0) bytes FROM versions"
        ).fetchone()["bytes"]
        unique_source = self.connection.execute(
            "SELECT COALESCE(SUM(source_bytes), 0) bytes FROM (SELECT source_hash, MAX(source_bytes) source_bytes FROM versions GROUP BY source_hash)"
        ).fetchone()["bytes"]
        duplicate_versions = self.connection.execute(
            "SELECT COALESCE(SUM(count - 1), 0) duplicates FROM (SELECT COUNT(*) count FROM versions GROUP BY source_hash)"
        ).fetchone()["duplicates"]
        executions = self.connection.execute(
            "SELECT COUNT(*) count, SUM(ephemeral) ephemeral FROM executions"
        ).fetchone()
        unused = self.connection.execute(
            """
            SELECT COUNT(*) count FROM tools t
            WHERE t.state = 'active' AND NOT EXISTS (
                SELECT 1 FROM executions e WHERE e.tool_id = t.tool_id
            )
            """
        ).fetchone()["count"]
        successful_reused = self.connection.execute(
            """
            SELECT COUNT(*) count FROM (
                SELECT tool_id FROM executions
                WHERE tool_id IS NOT NULL AND status = 'success'
                GROUP BY tool_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()["count"]
        successful_executions = self.connection.execute(
            "SELECT COUNT(*) count FROM executions WHERE status = 'success'"
        ).fetchone()["count"]
        reused_executions = self.connection.execute(
            """
            SELECT COALESCE(SUM(count - 1), 0) count FROM (
                SELECT COUNT(*) count FROM executions
                WHERE tool_id IS NOT NULL GROUP BY tool_id
            )
            """
        ).fetchone()["count"]
        successful_reused_executions = self.connection.execute(
            """
            SELECT COALESCE(SUM(successes_after_first), 0) count FROM (
                SELECT MAX(
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) - 1,
                    0
                ) successes_after_first
                FROM executions WHERE tool_id IS NOT NULL GROUP BY tool_id
            )
            """
        ).fetchone()["count"]
        first_success = self.connection.execute(
            """
            SELECT AVG(status = 'success') rate FROM executions e
            WHERE created_at = (
                SELECT MIN(e2.created_at) FROM executions e2
                WHERE e2.tool_id = e.tool_id
            ) AND tool_id IS NOT NULL
            """
        ).fetchone()["rate"]
        execution_stats = self.connection.execute(
            """
            SELECT AVG(duration_ms) avg_ms,
                   AVG(cpu_seconds) avg_cpu,
                   MAX(max_rss_bytes) max_rss
            FROM executions
            """
        ).fetchone()
        version_rows = self.connection.execute(
            "SELECT tool_id, version, source_hash, embedding FROM versions"
        ).fetchall()
        near_duplicate_pairs = 0
        comparable_pairs = 0
        for index, left in enumerate(version_rows):
            left_vector = json.loads(left["embedding"])
            for right in version_rows[index + 1 :]:
                comparable_pairs += 1
                if left["source_hash"] != right["source_hash"] and cosine(
                    left_vector, json.loads(right["embedding"])
                ) >= 0.92:
                    near_duplicate_pairs += 1
        db_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        embedding_bytes = self.connection.execute(
            "SELECT COALESCE(SUM(LENGTH(embedding)), 0) bytes FROM versions"
        ).fetchone()["bytes"]
        useful_density = successful_reused / active if active else 0
        return {
            "active_saved_tools": active,
            "total_versions": versions,
            "deleted_tools": deleted,
            "deleted_versions": deleted_versions,
            "ephemeral_executions": int(executions["ephemeral"] or 0),
            "total_executions": int(executions["count"]),
            "logical_source_bytes": int(source),
            "unique_source_bytes": int(unique_source),
            "metadata_database_bytes": db_bytes,
            "embedding_bytes": int(embedding_bytes),
            "total_storage_bytes": db_bytes + int(unique_source),
            "exact_duplicate_versions": int(duplicate_versions),
            "exact_duplicate_rate": int(duplicate_versions) / versions if versions else 0,
            "saved_tools_never_used": int(unused),
            "successfully_reused_tools": int(successful_reused),
            "successful_executions": int(successful_executions),
            "reused_executions": int(reused_executions),
            "successful_reused_executions": int(successful_reused_executions),
            "successful_reuse_rate": (
                int(successful_reused_executions) / int(reused_executions)
                if reused_executions
                else 0
            ),
            "first_execution_success_rate": float(first_success or 0),
            "useful_tool_density": useful_density,
            "storage_bytes_per_successful_reuse": (
                (db_bytes + int(unique_source)) / int(successful_reused_executions)
                if successful_reused_executions
                else None
            ),
            "near_duplicate_pairs": near_duplicate_pairs,
            "near_duplicate_rate": (
                near_duplicate_pairs / comparable_pairs if comparable_pairs else 0
            ),
            "average_execution_duration_ms": float(execution_stats["avg_ms"] or 0),
            "average_cpu_seconds": float(execution_stats["avg_cpu"] or 0),
            "peak_rss_bytes": int(execution_stats["max_rss"] or 0),
            "versions_per_tool": versions / (active + deleted) if active + deleted else 0,
        }
