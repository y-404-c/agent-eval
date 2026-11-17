from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_eval.adapters import supported_agents
from agent_eval.attribution import LLMAssistedAttributor, RuleBasedAttributor
from agent_eval.config import Settings
from agent_eval.dataset import DatasetManager, LeakageChecker
from agent_eval.domain import ErrorCategory
from agent_eval.evaluation import ObjectiveGrader
from agent_eval.evaluation.service import EvaluationService
from agent_eval.reporting import ReportGenerator
from agent_eval.storage import EvaluationRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ae", description="Agent 客观评测与失败归因框架")
    parser.add_argument("--db", default="var/agent_eval.db")
    parser.add_argument("--dataset", default="data/quixbugs.v1.jsonl")
    parser.add_argument("--reports-dir", default="reports")
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="版本化评测集管理")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    importer = dataset_commands.add_parser("import-quixbugs")
    importer.add_argument("--root", required=True)
    importer.add_argument("--version", required=True)
    importer.add_argument("--source", default="https://github.com/jkoppel/QuixBugs")
    dataset_commands.add_parser("verify")
    leakage = dataset_commands.add_parser("check-leakage")
    leakage.add_argument("--few-shot", required=True)

    ingest = commands.add_parser("ingest", help="适配 trace 并用官方测试重新评分")
    ingest.add_argument("--agent", required=True, choices=supported_agents())
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--experiment", required=True)
    ingest.add_argument("--problem-id", help="外部轨迹缺少 problem_id 时显式指定")
    ingest.add_argument("--attempt-index", type=int)
    ingest.add_argument("--attempt-offset", type=int, default=0)

    attribute = commands.add_parser("attribute", help="重新归因一个实验")
    attribute.add_argument("--experiment", required=True)
    attribute.add_argument("--llm", action="store_true", help="仅对低置信规则结果请求 LLM 建议")

    report = commands.add_parser("report", help="从真实 run 自动生成对比报告")
    report.add_argument("--experiment", required=True)
    report.add_argument("--agent", action="append", dest="agents")
    report.add_argument("--strategy")
    report.add_argument("--k", type=int, default=3)
    report.add_argument("--output")

    watchlist = commands.add_parser("watchlist", help="bad case 重点观测集")
    watch_commands = watchlist.add_subparsers(dest="watch_command", required=True)
    refresh = watch_commands.add_parser("refresh")
    refresh.add_argument("--min-failures", type=int, default=2)
    watch_commands.add_parser("list")
    export = watch_commands.add_parser("export")
    export.add_argument("--output", required=True)

    audit = commands.add_parser("audit", help="人工抽检与一致率")
    audit_commands = audit.add_subparsers(dest="audit_command", required=True)
    record = audit_commands.add_parser("record")
    record.add_argument("--run-id", required=True)
    record.add_argument("--reviewer", required=True)
    record.add_argument("--category", required=True, choices=[item.value for item in ErrorCategory])
    record.add_argument("--note", default="")
    audit_commands.add_parser("agreement")

    serve = commands.add_parser("serve", help="启动 FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    demo = commands.add_parser("demo", help="启动 Gradio trace 回放与报告 demo")
    demo.add_argument("--server-name", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument("--share", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(
        db_path=Path(args.db), reports_dir=Path(args.reports_dir)
    )
    repository = EvaluationRepository(settings.db_path)
    manager = DatasetManager()
    try:
        if args.command == "dataset":
            return _dataset(args, manager)
        dataset = manager.load(args.dataset)
        if args.command == "ingest":
            service = EvaluationService(dataset, repository, _grader(settings))
            options = {
                key: value
                for key, value in {
                    "problem_id": args.problem_id,
                    "attempt_index": args.attempt_index,
                    "attempt_offset": args.attempt_offset,
                }.items()
                if value is not None
            }
            traces = service.ingest(
                agent_name=args.agent,
                source=args.source,
                experiment_id=args.experiment,
                **options,
            )
            _print(
                {
                    "ingested": len(traces),
                    "passed": sum(trace.passed for trace in traces),
                    "failed": sum(not trace.passed for trace in traces),
                    "dataset_digest": dataset.digest,
                }
            )
            return 0
        if args.command == "attribute":
            attributor = _attributor(settings) if args.llm else RuleBasedAttributor()
            service = EvaluationService(
                dataset, repository, _grader(settings), attributor=attributor
            )
            results = service.attribute_experiment(args.experiment)
            if isinstance(attributor, LLMAssistedAttributor):
                attributor.close()
            _print({"attributed_failures": len(results)})
            return 0
        if args.command == "report":
            path = ReportGenerator(dataset, repository, settings.reports_dir).generate(
                experiment_id=args.experiment,
                agent_names=args.agents,
                strategy=args.strategy,
                k=args.k,
                output_dir=args.output,
            )
            _print({"report": str(path / "report.md"), "metrics": str(path / "metrics.json")})
            return 0
        if args.command == "watchlist":
            return _watchlist(args, repository, dataset)
        if args.command == "audit":
            return _audit(args, repository)
        if args.command == "serve":
            import uvicorn

            from agent_eval.api.app import create_app

            uvicorn.run(
                create_app(settings, dataset_path=args.dataset),
                host=args.host,
                port=args.port,
            )
            return 0
        if args.command == "demo":
            from agent_eval.demo import create_demo

            create_demo(settings).launch(
                server_name=args.server_name,
                server_port=args.port,
                share=args.share,
            )
            return 0
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error("未知命令")
    return 2


def _dataset(args, manager: DatasetManager) -> int:
    if args.dataset_command == "import-quixbugs":
        dataset = manager.import_quixbugs(
            args.root, args.dataset, version=args.version, source=args.source
        )
        _print({"problems": len(dataset), "digest": dataset.digest, "path": str(dataset.path)})
        return 0
    dataset = manager.load(args.dataset)
    if args.dataset_command == "verify":
        _print(
            {
                "status": "ok",
                "problems": len(dataset),
                "digest": dataset.digest,
                "versions": sorted({problem.version for problem in dataset}),
            }
        )
        return 0
    hits = LeakageChecker().check_file(dataset, args.few_shot)
    _print({"leaked_problem_ids": hits, "ok": not hits})
    return 1 if hits else 0


def _watchlist(args, repository: EvaluationRepository, dataset) -> int:
    if args.watch_command == "refresh":
        entries = repository.refresh_watchlist(args.min_failures)
    else:
        entries = repository.list_watchlist()
    if args.watch_command == "export":
        priority = [entry.problem_id for entry in entries]
        remaining = [problem_id for problem_id in dataset.problems if problem_id not in priority]
        payload = {
            "dataset_digest": dataset.digest,
            "problem_ids": [*priority, *remaining],
            "priority_problem_ids": priority,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _print({"output": str(output), "priority_count": len(priority)})
        return 0
    _print([entry.model_dump(mode="json") for entry in entries])
    return 0


def _audit(args, repository: EvaluationRepository) -> int:
    if args.audit_command == "record":
        record = repository.record_audit(
            args.run_id, args.reviewer, ErrorCategory(args.category), args.note
        )
        _print(record.model_dump(mode="json"))
        return 0
    _print(repository.audit_agreement())
    return 0


def _grader(settings: Settings) -> ObjectiveGrader:
    return ObjectiveGrader(
        python_executable=settings.python_executable,
        timeout_s=settings.test_timeout_s,
        max_output_chars=settings.max_output_chars,
    )


def _attributor(settings: Settings) -> LLMAssistedAttributor:
    if not settings.llm_base_url or not settings.llm_model:
        raise ValueError("启用 --llm 前需配置 AGENT_EVAL_LLM_BASE_URL 与 AGENT_EVAL_LLM_MODEL")
    return LLMAssistedAttributor(
        settings.llm_base_url, settings.llm_model, settings.llm_api_key
    )


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
