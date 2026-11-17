from __future__ import annotations

import json

from conftest import unified_patch

from agent_eval.domain import (
    TestStatus,
)
from agent_eval.evaluation import EvaluationService, ObjectiveGrader
from agent_eval.storage import EvaluationRepository


def test_ingest_regrades_logged_success_and_closed_loop(problem, dataset, tmp_path) -> None:
    raw = {
        "run_id": "logged-pass",
        "strategy": "S0",
        "problem_id": "sample",
        "repeat_index": 0,
        "passed": True,
        "attempts": [
            {
                "attempt": 1,
                "steps": [],
                "patch_text": "",
                "test_report": {
                    "returncode": 0,
                    "passed": 1,
                    "failed": 0,
                    "errors": 0,
                    "output_tail": "1 passed",
                },
                "passed": True,
            }
        ],
        "meta": {},
    }
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    repository = EvaluationRepository(tmp_path / "eval.db")
    service = EvaluationService(dataset, repository, ObjectiveGrader(timeout_s=10))

    traces = service.ingest(agent_name="ours", source=source, experiment_id="objectivity")

    assert traces[0].test_result == TestStatus.FAIL
    assert traces[0].metadata["logged_test_result"] == "pass"
    assert repository.get_run("logged-pass").test_result == TestStatus.FAIL
    assert repository.get_attribution("logged-pass") is not None

    attribution = repository.get_attribution("logged-pass")
    repository.save_run(
        traces[0].model_copy(update={"run_id": "second-failure"})
    )
    repository.save_attribution(attribution.model_copy(update={"run_id": "second-failure"}))
    entries = repository.refresh_watchlist(min_failures=2)
    assert [entry.problem_id for entry in entries] == ["sample"]

    audit = repository.record_audit(
        "logged-pass", "reviewer", attribution.category, "checked trace"
    )
    assert audit.agrees is True
    assert repository.audit_agreement()["agreement_rate"] == 1.0

    service.regrade("logged-pass")
    regraded = service.regrade("logged-pass")
    assert len(regraded.test_history) == len(traces[0].test_history) == 2
    assert [outcome.attempt for outcome in regraded.test_history] == [1, 2]
    assert regraded.metadata["logged_test_result"] == "pass"
    assert regraded.metadata["objective_regrades"] == 1


def test_ingest_real_patch_passes(problem, dataset, tmp_path) -> None:
    patch = unified_patch(problem, "def add(a, b):\n    return a + b\n")
    raw = {
        "run_id": "actual-pass",
        "strategy": "S0",
        "problem_id": "sample",
        "repeat_index": 0,
        "passed": False,
        "attempts": [
            {
                "attempt": 1,
                "steps": [],
                "patch_text": patch,
                "changed_files": [problem.entry_path],
                "test_report": {"returncode": 1, "passed": 0, "failed": 1},
                "passed": False,
            }
        ],
        "meta": {},
    }
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    repository = EvaluationRepository(tmp_path / "eval.db")
    service = EvaluationService(dataset, repository, ObjectiveGrader(timeout_s=10))

    trace = service.ingest(agent_name="ours", source=source, experiment_id="pass")[0]

    assert trace.passed
    assert trace.test_history[-1].status == TestStatus.PASS
    assert repository.get_attribution(trace.run_id) is None
