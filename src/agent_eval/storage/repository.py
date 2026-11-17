from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from agent_eval.domain import (
    Attribution,
    AuditRecord,
    ErrorCategory,
    RunTrace,
    WatchlistEntry,
)

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    strategy TEXT,
    problem_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    test_result TEXT NOT NULL,
    started_at TEXT,
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_name, experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_problem ON runs(problem_id);
CREATE TABLE IF NOT EXISTS attributions (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    method TEXT NOT NULL,
    confidence REAL NOT NULL,
    needs_review INTEGER NOT NULL,
    attribution_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audits (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    manual_category TEXT NOT NULL,
    predicted_category TEXT NOT NULL,
    agrees INTEGER NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, reviewer)
);
CREATE TABLE IF NOT EXISTS watchlist (
    problem_id TEXT PRIMARY KEY,
    failure_count INTEGER NOT NULL,
    watchlist_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class EvaluationRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(_DDL)

    def save_run(self, trace: RunTrace) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, agent_name, strategy, problem_id, experiment_id,
                    attempt_index, test_result, started_at, trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.run_id,
                    trace.agent_name,
                    trace.strategy,
                    trace.problem_id,
                    trace.experiment_id,
                    trace.attempt_index,
                    trace.test_result.value,
                    trace.started_at.isoformat() if trace.started_at else None,
                    trace.model_dump_json(),
                ),
            )

    def get_run(self, run_id: str) -> RunTrace | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT trace_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return RunTrace.model_validate_json(row["trace_json"]) if row else None

    def list_runs(
        self,
        *,
        experiment_id: str | None = None,
        agent_names: Iterable[str] | None = None,
        problem_ids: Iterable[str] | None = None,
    ) -> list[RunTrace]:
        clauses: list[str] = []
        parameters: list[object] = []
        if experiment_id:
            clauses.append("experiment_id=?")
            parameters.append(experiment_id)
        agents = list(agent_names or [])
        if agents:
            clauses.append(f"agent_name IN ({','.join('?' for _ in agents)})")
            parameters.extend(agents)
        problems = list(problem_ids or [])
        if problems:
            clauses.append(f"problem_id IN ({','.join('?' for _ in problems)})")
            parameters.extend(problems)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT trace_json FROM runs {where} "
                "ORDER BY experiment_id, agent_name, strategy, problem_id, attempt_index, run_id",
                parameters,
            ).fetchall()
        return [RunTrace.model_validate_json(row["trace_json"]) for row in rows]

    def save_attribution(self, attribution: Attribution) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO attributions (
                    run_id, category, method, confidence, needs_review, attribution_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    attribution.run_id,
                    attribution.category.value,
                    attribution.method,
                    attribution.confidence,
                    int(attribution.needs_review),
                    attribution.model_dump_json(),
                ),
            )

    def delete_attribution(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM attributions WHERE run_id=?", (run_id,))

    def get_attribution(self, run_id: str) -> Attribution | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attribution_json FROM attributions WHERE run_id=?", (run_id,)
            ).fetchone()
        return Attribution.model_validate_json(row["attribution_json"]) if row else None

    def list_attributions(self, run_ids: Iterable[str] | None = None) -> list[Attribution]:
        ids = list(run_ids or [])
        where = f"WHERE run_id IN ({','.join('?' for _ in ids)})" if ids else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT attribution_json FROM attributions {where} ORDER BY run_id", ids
            ).fetchall()
        return [Attribution.model_validate_json(row["attribution_json"]) for row in rows]

    def record_audit(
        self,
        run_id: str,
        reviewer: str,
        manual_category: ErrorCategory,
        note: str = "",
    ) -> AuditRecord:
        attribution = self.get_attribution(run_id)
        if attribution is None:
            raise ValueError(f"run 尚无自动归因，不能抽检: {run_id}")
        record = AuditRecord(
            run_id=run_id,
            reviewer=reviewer,
            manual_category=manual_category,
            predicted_category=attribution.category,
            agrees=manual_category == attribution.category,
            note=note,
            created_at=datetime.now(timezone.utc),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO audits (
                    run_id, reviewer, manual_category, predicted_category,
                    agrees, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.reviewer,
                    record.manual_category.value,
                    record.predicted_category.value,
                    int(record.agrees),
                    record.note,
                    record.created_at.isoformat(),
                ),
            )
        return record

    def list_audits(self) -> list[AuditRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audits ORDER BY created_at, run_id"
            ).fetchall()
        return [
            AuditRecord(
                run_id=row["run_id"],
                reviewer=row["reviewer"],
                manual_category=ErrorCategory(row["manual_category"]),
                predicted_category=ErrorCategory(row["predicted_category"]),
                agrees=bool(row["agrees"]),
                note=row["note"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def audit_agreement(self) -> dict[str, int | float | None]:
        audits = self.list_audits()
        agreements = sum(record.agrees for record in audits)
        return {
            "sample_count": len(audits),
            "agreement_count": agreements,
            "agreement_rate": agreements / len(audits) if audits else None,
        }

    def refresh_watchlist(self, min_failures: int = 2) -> list[WatchlistEntry]:
        runs = self.list_runs()
        failed = [run for run in runs if not run.passed]
        attributions = {item.run_id: item for item in self.list_attributions()}
        grouped: dict[str, list[RunTrace]] = defaultdict(list)
        for run in failed:
            grouped[run.problem_id].append(run)
        now = datetime.now(timezone.utc)
        entries: list[WatchlistEntry] = []
        for problem_id, items in grouped.items():
            if len(items) < min_failures:
                continue
            categories = Counter(
                attributions[item.run_id].category
                if item.run_id in attributions
                else item.error_category
                for item in items
            )
            present = [category for category in categories if category is not None]
            ordered_categories = sorted(
                present, key=lambda category: (-categories[category], category.value)
            )
            run_ids = sorted(item.run_id for item in items)
            entries.append(
                WatchlistEntry(
                    problem_id=problem_id,
                    failure_count=len(items),
                    categories=ordered_categories,
                    run_ids=run_ids,
                    reason=f"历史失败 {len(items)} 次，达到优先复测阈值 {min_failures}",
                    updated_at=now,
                )
            )
        entries.sort(key=lambda item: (-item.failure_count, item.problem_id))
        with self._connect() as connection:
            connection.execute("DELETE FROM watchlist")
            for entry in entries:
                connection.execute(
                    """
                    INSERT INTO watchlist (problem_id, failure_count, watchlist_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        entry.problem_id,
                        entry.failure_count,
                        entry.model_dump_json(),
                        entry.updated_at.isoformat(),
                    ),
                )
        return entries

    def list_watchlist(self) -> list[WatchlistEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT watchlist_json FROM watchlist ORDER BY failure_count DESC, problem_id"
            ).fetchall()
        return [WatchlistEntry.model_validate_json(row["watchlist_json"]) for row in rows]
