from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from agent_eval.api.deps import Container, get_container
from agent_eval.api.schemas import AuditRequest, ReportRequest
from agent_eval.config import Settings
from agent_eval.domain import ErrorCategory
from agent_eval.evaluation import EvaluationService, ObjectiveGrader
from agent_eval.reporting import ReportGenerator
from agent_eval.storage import EvaluationRepository

ContainerDep = Annotated[Container, Depends(get_container)]


def create_app(
    settings: Settings | None = None,
    *,
    dataset_path: Path | str | None = None,
) -> FastAPI:
    settings = settings or Settings()
    reports_dir = Path(settings.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    container = Container(
        settings=settings,
        repository=EvaluationRepository(settings.db_path),
        dataset_path=Path(dataset_path or "data/quixbugs.v1.jsonl"),
    )
    app = FastAPI(
        title="Agent Evaluation Harness",
        version="0.1.0",
        description="统一 trace、官方测试客观复测、失败归因与横向报告",
    )
    app.state.container = container
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs")
    def list_runs(
        container: ContainerDep,
        experiment_id: str | None = None,
        agent_name: str | None = Query(default=None),
    ) -> list[dict]:
        runs = container.repository.list_runs(
            experiment_id=experiment_id,
            agent_names=[agent_name] if agent_name else None,
        )
        return [
            {
                "run_id": run.run_id,
                "agent_name": run.agent_name,
                "strategy": run.strategy,
                "problem_id": run.problem_id,
                "experiment_id": run.experiment_id,
                "attempt_index": run.attempt_index,
                "test_result": run.test_result.value,
                "total_tokens": run.total_tokens,
            }
            for run in runs
        ]

    @app.get("/runs/{run_id}")
    def replay(run_id: str, container: ContainerDep) -> dict:
        trace = container.repository.get_run(run_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="run_id 不存在")
        attribution = container.repository.get_attribution(run_id)
        return {
            "trace": trace.model_dump(mode="json"),
            "attribution": attribution.model_dump(mode="json") if attribution else None,
        }

    @app.post("/runs/{run_id}/regrade")
    def regrade(run_id: str, container: ContainerDep) -> dict:
        settings = container.settings
        service = EvaluationService(
            container.dataset(),
            container.repository,
            ObjectiveGrader(
                python_executable=settings.python_executable,
                timeout_s=settings.test_timeout_s,
                max_output_chars=settings.max_output_chars,
            ),
        )
        try:
            trace = service.regrade(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        attribution = container.repository.get_attribution(run_id)
        return {
            "trace": trace.model_dump(mode="json"),
            "attribution": attribution.model_dump(mode="json") if attribution else None,
        }

    @app.get("/watchlist")
    def watchlist(container: ContainerDep) -> list[dict]:
        return [
            item.model_dump(mode="json") for item in container.repository.list_watchlist()
        ]

    @app.post("/watchlist/refresh")
    def refresh_watchlist(
        container: ContainerDep,
        min_failures: int = Query(default=2, ge=1),
    ) -> list[dict]:
        entries = container.repository.refresh_watchlist(min_failures)
        return [item.model_dump(mode="json") for item in entries]

    @app.post("/runs/{run_id}/audit")
    def audit(
        run_id: str,
        request: AuditRequest,
        container: ContainerDep,
    ) -> dict:
        try:
            record = container.repository.record_audit(
                run_id,
                request.reviewer,
                ErrorCategory(request.manual_category),
                request.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return record.model_dump(mode="json")

    @app.get("/audits/agreement")
    def agreement(container: ContainerDep) -> dict:
        return container.repository.audit_agreement()

    @app.post("/reports")
    def generate_report(
        request: ReportRequest,
        container: ContainerDep,
    ) -> dict:
        try:
            path = ReportGenerator(
                container.dataset(), container.repository, reports_dir
            ).generate(
                experiment_id=request.experiment_id,
                agent_names=request.agent_names,
                strategy=request.strategy,
                k=request.k,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "report_dir": str(path),
            "report_markdown": str(path / "report.md"),
            "metrics_json": str(path / "metrics.json"),
        }

    return app


app = create_app()
