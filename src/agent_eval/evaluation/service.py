from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Protocol

from agent_eval.adapters import get_adapter
from agent_eval.attribution import RuleBasedAttributor
from agent_eval.dataset import VersionedDataset
from agent_eval.domain import Attribution, RunTrace
from agent_eval.evaluation.grader import ObjectiveGrader
from agent_eval.evaluation.patch import changed_files
from agent_eval.storage import EvaluationRepository


class Attributor(Protocol):
    def attribute(self, trace: RunTrace, problem) -> Attribution | None: ...


class EvaluationService:
    """同一入口切换 agent_name：适配 -> 官方测试复测 -> 落库 -> 归因。"""

    def __init__(
        self,
        dataset: VersionedDataset,
        repository: EvaluationRepository,
        grader: ObjectiveGrader,
        attributor: Attributor | None = None,
    ) -> None:
        self.dataset = dataset
        self.repository = repository
        self.grader = grader
        self.attributor = attributor or RuleBasedAttributor()

    def ingest(
        self,
        *,
        agent_name: str,
        source: Path | str,
        experiment_id: str,
        **adapter_options: Any,
    ) -> list[RunTrace]:
        adapter = get_adapter(agent_name)
        traces = adapter.load(
            source,
            experiment_id=experiment_id,
            dataset_digest=self.dataset.digest,
            **adapter_options,
        )
        results: list[RunTrace] = []
        for original in traces:
            if original.metadata.get("model") == "scripted":
                raise ValueError(
                    f"拒绝导入 scripted 模拟数据 run_id={original.run_id}："
                    "它只用于项目一离线自检，不代表真实 Agent 结果。"
                    "请使用正式评测产出的 var/runs.db。"
                )
            if original.problem_id not in self.dataset.problems:
                warnings.warn(
                    f"run {original.run_id} 的题目 {original.problem_id} 不在当前评测集中，"
                    "可能与项目一当时使用的数据集版本不一致。"
                )
            trace = self.regrade_trace(original)
            self.repository.save_run(trace)
            self.attribute(trace)
            results.append(trace)
        return results

    def regrade(self, run_id: str) -> RunTrace:
        original = self.repository.get_run(run_id)
        if original is None:
            raise KeyError(f"未知 run_id: {run_id}")
        trace = self.regrade_trace(original)
        self.repository.save_run(trace)
        self.attribute(trace)
        return trace

    def regrade_trace(self, original: RunTrace) -> RunTrace:
        problem = self.dataset.get(original.problem_id)
        metadata = dict(original.metadata)
        previous_regrades = int(metadata.get("objective_regrades", 0))
        upstream_history = (
            original.test_history[:-previous_regrades]
            if previous_regrades
            else list(original.test_history)
        )
        outcome = self.grader.grade(
            problem,
            original.final_patch,
            attempt=len(upstream_history) + 1,
        )
        metadata.update(
            {
                "dataset_digest": self.dataset.digest,
                "objective_grader": "official-pytest-v1",
                "objective_regrades": 1,
                "logged_test_result": metadata.get(
                    "logged_test_result", original.test_result.value
                ),
            }
        )
        return original.model_copy(
            update={
                "test_result": outcome.status,
                "test_output": outcome.output,
                "test_history": [*upstream_history, outcome],
                "changed_files": changed_files(original.final_patch)
                or original.changed_files,
                "error_category": None,
                "metadata": metadata,
            }
        )

    def attribute(self, trace: RunTrace) -> Attribution | None:
        if trace.passed:
            self.repository.delete_attribution(trace.run_id)
            return None
        problem = self.dataset.get(trace.problem_id)
        attribution = self.attributor.attribute(trace, problem)
        if attribution is not None:
            self.repository.save_attribution(attribution)
        return attribution

    def attribute_experiment(self, experiment_id: str) -> list[Attribution]:
        results: list[Attribution] = []
        for trace in self.repository.list_runs(experiment_id=experiment_id):
            attribution = self.attribute(trace)
            if attribution is not None:
                results.append(attribution)
        return results

    def replay(self, run_id: str) -> tuple[RunTrace, Attribution | None]:
        trace = self.repository.get_run(run_id)
        if trace is None:
            raise KeyError(f"未知 run_id: {run_id}")
        return trace, self.repository.get_attribution(run_id)
