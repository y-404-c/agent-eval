from __future__ import annotations

import posixpath
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from agent_eval.domain import Attribution, Problem, RunTrace, Step
from agent_eval.metrics.models import MetricsResult, WilsonInterval

_Z_95 = 1.959963984540054
_READ_ACTIONS = {
    "ast",
    "ast_slice",
    "cat",
    "open_file",
    "read",
    "read_code",
    "read_file",
    "read_snippet",
    "view_file",
}
_PATH_KEYS = (
    "path",
    "file_path",
    "filepath",
    "file",
    "filename",
    "file_name",
    "relative_path",
    "target_path",
    "source_path",
)


class MetricsCalculator:
    """从统一轨迹计算指标，不使用测试历史或 LLM 为通过情况打分。"""

    @classmethod
    def calculate(
        cls,
        traces: Iterable[RunTrace],
        problems: Iterable[Problem],
        k: int = 3,
        attributions: Iterable[Attribution] | None = None,
    ) -> list[MetricsResult]:
        """按 Agent 和策略分组计算指标。

        每条 trace 是一次独立生成。pass@k 取每题按 attempt_index、run_id 排序后的
        前 k 条 trace，只要至少一条最终测试通过即为通过；trace 内部重试不计为新生成。
        第 N 个 repeat 对应该排序中的第 N 条 trace，仅在拥有该 repeat 的题目上计算。
        """
        if k < 1:
            raise ValueError("k 必须大于等于 1")

        items = list(traces)
        if not items:
            return []

        gold_files = {
            problem.problem_id: tuple(
                normalized
                for path in problem.gold_files
                if (normalized := _normalize_path(path)) not in {"", "."}
            )
            for problem in problems
        }
        attribution_by_run = {
            attribution.run_id: attribution for attribution in (attributions or [])
        }
        grouped: dict[tuple[str, str], list[RunTrace]] = defaultdict(list)
        for trace in items:
            strategy = (trace.strategy or "").strip() or "-"
            grouped[(trace.agent_name, strategy)].append(trace)

        return [
            _calculate_group(
                agent_name,
                strategy,
                grouped[(agent_name, strategy)],
                gold_files,
                attribution_by_run,
                k,
            )
            for agent_name, strategy in sorted(grouped)
        ]


def _calculate_group(
    agent_name: str,
    strategy: str,
    traces: list[RunTrace],
    gold_files: dict[str, tuple[str, ...]],
    attributions: dict[str, Attribution],
    k: int,
) -> MetricsResult:
    runs_by_problem: dict[str, list[RunTrace]] = defaultdict(list)
    for trace in traces:
        runs_by_problem[trace.problem_id].append(trace)
    ordered_runs = [
        sorted(runs, key=lambda trace: (trace.attempt_index, trace.run_id))
        for runs in runs_by_problem.values()
    ]

    problem_count = len(ordered_runs)
    run_count = len(traces)
    repeat_count = max(len(runs) for runs in ordered_runs)
    pass_at_1 = sum(runs[0].passed for runs in ordered_runs) / problem_count
    pass_at_k = (
        sum(any(trace.passed for trace in runs[:k]) for runs in ordered_runs) / problem_count
    )

    repeat_pass_rates = []
    for repeat in range(repeat_count):
        repeat_runs = [runs[repeat] for runs in ordered_runs if len(runs) > repeat]
        repeat_pass_rates.append(
            sum(trace.passed for trace in repeat_runs) / len(repeat_runs)
        )
    mean_repeat_pass_rate = statistics.fmean(repeat_pass_rates)
    repeat_variance = (
        statistics.variance(repeat_pass_rates) if len(repeat_pass_rates) > 1 else None
    )

    read_paths = [_read_paths(trace) for trace in traces]
    localization_top1 = sum(
        _contains_gold(paths[:1], gold_files.get(trace.problem_id, ()))
        for trace, paths in zip(traces, read_paths, strict=True)
    ) / run_count
    localization_top3 = sum(
        _contains_gold(paths[:3], gold_files.get(trace.problem_id, ()))
        for trace, paths in zip(traces, read_paths, strict=True)
    ) / run_count

    total_tokens = sum(trace.total_tokens for trace in traces)
    latencies = [
        step.latency_ms
        for trace in traces
        for step in trace.steps
        if step.latency_ms > 0
    ]
    retried = [trace for trace in traces if trace.retry_count > 0]
    retry_success_rate = (
        sum(trace.passed for trace in retried) / len(retried) if retried else None
    )
    error_counts = _error_counts(traces, attributions)
    failure_count = sum(error_counts.values())

    return MetricsResult(
        agent_name=agent_name,
        strategy=strategy,
        run_count=run_count,
        problem_count=problem_count,
        repeat_count=repeat_count,
        k=k,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        repeat_pass_rates=repeat_pass_rates,
        mean_repeat_pass_rate=mean_repeat_pass_rate,
        repeat_pass_rate_variance=repeat_variance,
        repeat_pass_rate_ci95=_wilson_interval(
            sum(trace.passed for trace in traces), run_count
        ),
        localization_top1=localization_top1,
        localization_top3=localization_top3,
        avg_tokens_per_run=total_tokens / run_count,
        avg_tokens_per_problem=total_tokens / problem_count,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        retry_success_rate=retry_success_rate,
        failure_count=failure_count,
        error_counts=error_counts,
        error_distribution={
            category: count / failure_count for category, count in error_counts.items()
        },
    )


def _read_paths(trace: RunTrace) -> list[str | None]:
    paths: list[str | None] = []
    for step in trace.steps:
        if step.action.strip().lower() in _READ_ACTIONS:
            paths.append(_read_path(step))
    return paths


def _read_path(step: Step) -> str | None:
    for key in _PATH_KEYS:
        value: Any = step.args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _contains_gold(paths: list[str | None], gold_files: tuple[str, ...]) -> bool:
    return any(
        path is not None and _path_matches(path, gold)
        for path in paths
        for gold in gold_files
    )


def _path_matches(candidate: str, gold: str) -> bool:
    normalized = _normalize_path(candidate)
    if normalized in {"", "."}:
        return False
    return (
        normalized == gold
        or normalized.endswith(f"/{gold}")
        or gold.endswith(f"/{normalized}")
    )


def _normalize_path(path: str) -> str:
    return posixpath.normpath(path.strip().replace("\\", "/"))


def _percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _wilson_interval(successes: int, total: int) -> WilsonInterval:
    proportion = successes / total
    z_squared = _Z_95**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        _Z_95
        * (proportion * (1 - proportion) / total + z_squared / (4 * total**2)) ** 0.5
        / denominator
    )
    return WilsonInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
    )


def _error_counts(
    traces: list[RunTrace], attributions: dict[str, Attribution]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for trace in traces:
        if trace.passed:
            continue
        attribution = attributions.get(trace.run_id)
        category = attribution.category if attribution else trace.error_category
        counts[category.value if category else "unknown"] += 1
    return dict(sorted(counts.items()))
