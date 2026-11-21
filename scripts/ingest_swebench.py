"""把 SWE-bench 的真实 Agent 轨迹接入项目二：适配 -> 官方判定复测 -> 落库 -> 归因 -> 报告。

用法（在 agent-eval 根目录）:
    .venv/Scripts/python.exe scripts/ingest_swebench.py

数据全部来自真实运行：轨迹是 mini-swe-agent 在 SWE-bench 镜像里的执行记录，
判定是 swebench harness 在容器里跑 FAIL_TO_PASS/PASS_TO_PASS 的结果。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_eval.attribution import RuleBasedAttributor
from agent_eval.dataset import DatasetManager
from agent_eval.domain import ErrorCategory
from agent_eval.evaluation.harness_grader import HarnessVerdictGrader
from agent_eval.evaluation.service import EvaluationService
from agent_eval.reporting import ReportGenerator
from agent_eval.storage import EvaluationRepository

DATA = ROOT / "data" / "swebench"
DB = ROOT / "var" / "swebench_eval.db"
REPORTS = ROOT / "reports"

EXPERIMENT = "swebench-lite-harness-v1"
RUNS = [
    ("pytest-dev__pytest-6116", "traces/pytest-6116.traj.json", 0),
    ("django__django-14534", "traces/django-14534.traj.json", 0),
]


def main() -> int:
    dataset = DatasetManager().load(DATA / "dataset.jsonl")
    grader = HarnessVerdictGrader(DATA / "verdicts" / "official.json")
    repository = EvaluationRepository(DB)
    service = EvaluationService(dataset, repository, grader, RuleBasedAttributor())

    for problem_id, relative, attempt_index in RUNS:
        traces = service.ingest(
            agent_name="mini-swe-agent",
            source=DATA / relative,
            experiment_id=EXPERIMENT,
            problem_id=problem_id,
            attempt_index=attempt_index,
        )
        for trace in traces:
            attribution = repository.get_attribution(trace.run_id)
            category = attribution.category.value if attribution else "-"
            print(
                f"INGEST {problem_id}: status={trace.test_result.value} "
                f"category={category} steps={len(trace.steps)} patch={len(trace.final_patch)}"
            )

    attributed = service.attribute_experiment(EXPERIMENT)
    print(f"ATTRIBUTED {len(attributed)} failures")
    for item in attributed:
        print(f"  {item.run_id}: {item.category.value} conf={item.confidence:.2f}")

    watchlist = repository.refresh_watchlist(min_failures=1)
    print(f"WATCHLIST {len(watchlist)} entries")

    path = ReportGenerator(dataset, repository, REPORTS).generate(
        experiment_id=EXPERIMENT,
        output_dir=REPORTS / "swebench-harness-v1",
    )
    print(f"REPORT {path / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
