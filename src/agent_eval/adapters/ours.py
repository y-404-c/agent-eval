from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from agent_eval.adapters.base import TraceAdapter
from agent_eval.domain import RunTrace, Step, TestOutcome, TestStatus


class OursAdapter(TraceAdapter):
    """读取项目一原生 RunTrace，不依赖项目一安装或其内部枚举。"""

    name = "ours"

    def load(
        self,
        source: Path | str,
        *,
        experiment_id: str = "default",
        attempt_offset: int = 0,
        dataset_digest: str = "",
        **_: Any,
    ) -> Iterable[RunTrace]:
        path = Path(source)
        for payload in self._payloads(path):
            yield self._convert(
                payload,
                source=path,
                experiment_id=experiment_id,
                attempt_offset=attempt_offset,
                dataset_digest=dataset_digest,
            )

    def _payloads(self, path: Path) -> Iterator[dict[str, Any]]:
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(path) as connection:
                rows = connection.execute(
                    "SELECT trace_json FROM runs ORDER BY started_at, run_id"
                ).fetchall()
            for row in rows:
                yield json.loads(row[0])
            return
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        yield _json_object(line, f"{path}:{line_number}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError(f"轨迹数组项必须是对象: {path}")
                yield item
        elif isinstance(payload, dict):
            yield payload
        else:
            raise ValueError(f"不支持的轨迹 JSON 结构: {path}")

    def _convert(
        self,
        raw: dict[str, Any],
        *,
        source: Path,
        experiment_id: str,
        attempt_offset: int,
        dataset_digest: str,
    ) -> RunTrace:
        attempts = raw.get("attempts") or []
        if not isinstance(attempts, list):
            raise ValueError(f"项目一 attempts 必须是数组: {raw.get('run_id')}")

        steps: list[Step] = []
        history: list[TestOutcome] = []
        final_patch = ""
        changed_files: list[str] = []
        for attempt_index, attempt in enumerate(attempts, 1):
            for item in attempt.get("steps", []):
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        action=str(item.get("action", "think")),
                        args=item.get("args") or {},
                        observation=str(item.get("observation", "")),
                        tokens=int(item.get("tokens", 0) or 0),
                        latency_ms=int(item.get("latency_ms", 0) or 0),
                        thought=str(item.get("thought", "")),
                        ok=bool(item.get("ok", True)),
                        error=item.get("error"),
                    )
                )
            report = attempt.get("test_report")
            if report:
                history.append(_test_outcome(report, attempt_index, attempt.get("passed")))
            if str(attempt.get("patch_text", "")).strip():
                final_patch = str(attempt["patch_text"])
            if attempt.get("changed_files"):
                changed_files = [str(item) for item in attempt["changed_files"]]

        passed = bool(raw.get("passed", False))
        timed_out = bool(history and history[-1].status == TestStatus.TIMEOUT)
        status = TestStatus.PASS if passed else (
            TestStatus.TIMEOUT if timed_out else TestStatus.FAIL
        )
        meta = dict(raw.get("meta") or {})
        if "final_stop_reason" in meta and "stop_reason" not in meta:
            meta["stop_reason"] = meta["final_stop_reason"]
        meta.update(
            {
                "model": raw.get("model", ""),
                "upstream_config_hash": raw.get("config_hash", ""),
                "prompt_version": raw.get("prompt_version", ""),
                "upstream_error_category": str(raw.get("error_category", "")),
            }
        )
        return RunTrace(
            run_id=str(raw["run_id"]),
            agent_name=self.name,
            strategy=str(raw["strategy"]) if raw.get("strategy") is not None else None,
            problem_id=str(raw["problem_id"]),
            steps=steps,
            final_patch=final_patch,
            test_result=status,
            error_category=None,
            experiment_id=experiment_id,
            attempt_index=int(raw.get("repeat_index", 0)) + attempt_offset,
            retry_count=max(0, len(attempts) - 1),
            test_history=history,
            test_output=history[-1].output if history else "",
            changed_files=changed_files or _patch_files(final_patch),
            started_at=raw.get("started_at"),
            source_uri=f"{source.as_posix()}#{raw['run_id']}",
            metadata=meta,
        )


def _test_outcome(
    report: dict[str, Any], attempt: int, attempt_passed: Any
) -> TestOutcome:
    timed_out = bool(report.get("timed_out", False))
    report_ok = (
        not timed_out
        and int(report.get("returncode", -1)) == 0
        and int(report.get("failed", 0)) == 0
        and int(report.get("errors", 0)) == 0
        and int(report.get("passed", 0)) > 0
    )
    passed = report_ok if attempt_passed is None else bool(attempt_passed)
    return TestOutcome(
        attempt=attempt,
        status=TestStatus.PASS if passed else (
            TestStatus.TIMEOUT if timed_out else TestStatus.FAIL
        ),
        output=str(report.get("output_tail", "")),
        returncode=int(report.get("returncode", -1)),
        passed=int(report.get("passed", 0)),
        failed=int(report.get("failed", 0)),
        errors=int(report.get("errors", 0)),
        failed_tests=[str(item) for item in report.get("failed_tests", [])],
        duration_ms=int(report.get("duration_ms", 0)),
    )


def _patch_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        value = line[4:].strip()
        if value == "/dev/null":
            continue
        if value.startswith(("a/", "b/")):
            value = value[2:]
        files.append(value)
    return list(dict.fromkeys(files))


def _json_object(line: str, location: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"JSONL 每行必须是对象: {location}")
    return value
