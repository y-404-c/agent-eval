from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from agent_eval.api.app import create_app
from agent_eval.config import Settings
from agent_eval.domain import Attribution, ErrorCategory, RunTrace, TestStatus
from agent_eval.reporting import ReportGenerator
from agent_eval.storage import EvaluationRepository


def _run(run_id: str, agent: str, status: TestStatus) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        agent_name=agent,
        strategy="baseline",
        problem_id="sample",
        experiment_id="comparison",
        test_result=status,
        test_output="" if status == TestStatus.PASS else "SyntaxError: invalid syntax",
        metadata={"dataset_digest": "test-digest"},
    )


def test_report_is_generated_from_stored_runs(dataset, tmp_path) -> None:
    repository = EvaluationRepository(tmp_path / "eval.db")
    repository.save_run(_run("ours-pass", "ours", TestStatus.PASS))
    repository.save_run(_run("mini-fail", "mini-swe-agent", TestStatus.FAIL))
    repository.save_attribution(
        Attribution(
            run_id="mini-fail",
            category=ErrorCategory.SYNTAX_ERROR,
            method="rule",
            confidence=0.98,
            rationale="真实测试输出包含 SyntaxError",
            created_at=datetime.now(timezone.utc),
        )
    )

    output = ReportGenerator(dataset, repository, tmp_path / "reports").generate(
        experiment_id="comparison", k=1
    )

    assert (output / "report.md").exists()
    assert (output / "metrics.csv").exists()
    assert (output / "pass_at_1.png").stat().st_size > 0
    assert (output / "error_distribution.png").stat().st_size > 0
    assert (output / "token_success.png").stat().st_size > 0
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert len(payload["metrics"]) == 2
    assert payload["bad_cases"][0]["run_id"] == "mini-fail"
    assert (output / payload["bad_cases"][0]["trace_link"]).exists()
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "100.00%" in markdown
    assert "syntax_error" in markdown
    assert "未使用 LLM 打分" in markdown


def test_api_health_replay_and_audit(dataset, tmp_path) -> None:
    settings = Settings(
        db_path=tmp_path / "api.db",
        reports_dir=tmp_path / "reports",
    )
    app = create_app(settings, dataset_path=dataset.path)
    repository = app.state.container.repository
    repository.save_run(_run("api-fail", "ours", TestStatus.FAIL))
    repository.save_attribution(
        Attribution(
            run_id="api-fail",
            category=ErrorCategory.SYNTAX_ERROR,
            method="rule",
            confidence=1,
            rationale="test",
            created_at=datetime.now(timezone.utc),
        )
    )
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/runs/api-fail").json()["trace"]["run_id"] == "api-fail"
    assert client.get("/runs/missing").status_code == 404
    response = client.post(
        "/runs/api-fail/audit",
        json={"reviewer": "human", "manual_category": "syntax_error"},
    )
    assert response.status_code == 200
    assert client.get("/audits/agreement").json()["sample_count"] == 1


def test_api_regrade_is_repeatable(dataset, tmp_path) -> None:
    settings = Settings(
        db_path=tmp_path / "api.db",
        reports_dir=tmp_path / "reports",
    )
    app = create_app(settings, dataset_path=dataset.path)
    app.state.container.repository.save_run(_run("api-fail", "ours", TestStatus.FAIL))
    client = TestClient(app)

    first = client.post("/runs/api-fail/regrade")
    second = client.post("/runs/api-fail/regrade")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["trace"]["test_result"] == "fail"
    assert len(second.json()["trace"]["test_history"]) == 1
    assert second.json()["attribution"] is not None
    assert client.post("/runs/missing/regrade").status_code == 404
