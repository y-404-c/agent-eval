from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent_eval.attribution import FailureClusterer
from agent_eval.dataset import VersionedDataset
from agent_eval.domain import Attribution, RunTrace
from agent_eval.metrics import MetricsCalculator, MetricsResult
from agent_eval.storage import EvaluationRepository


class ReportGenerator:
    """报告中的每个数字和图都来自 SQLite 中已复测的真实 run。"""

    def __init__(
        self,
        dataset: VersionedDataset,
        repository: EvaluationRepository,
        reports_root: Path | str,
    ) -> None:
        self.dataset = dataset
        self.repository = repository
        self.reports_root = Path(reports_root)
        self.reports_root.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        experiment_id: str,
        agent_names: Iterable[str] | None = None,
        strategy: str | None = None,
        k: int = 3,
        output_dir: Path | str | None = None,
    ) -> Path:
        runs = self.repository.list_runs(
            experiment_id=experiment_id, agent_names=agent_names
        )
        if strategy is not None:
            runs = [run for run in runs if (run.strategy or "-") == strategy]
        if not runs:
            raise ValueError("没有可用于报告的已复测 run")
        runs, common_ids, warnings = self._common_coverage(runs)
        if not runs:
            raise ValueError("不同 Agent 没有共同题目，无法公平对比")

        run_ids = [run.run_id for run in runs]
        attributions = self.repository.list_attributions(run_ids)
        metrics = MetricsCalculator.calculate(
            runs,
            self.dataset,
            k=k,
            attributions=attributions,
        )
        clusters = FailureClusterer(n_clusters=3).cluster(runs, attributions)[:3]
        report_dir = Path(output_dir) if output_dir else self._new_report_dir(experiment_id)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "traces").mkdir(exist_ok=True)

        bad_cases = self._export_bad_cases(runs, attributions, report_dir)
        generated_at = datetime.now(timezone.utc)
        grader_sources = sorted(
            {run.metadata.get("objective_grader", "unknown") for run in runs}
        )
        if grader_sources == ["swebench-harness-v1"]:
            objective_description = (
                "主指标为 SWE-bench 官方 harness 在对应镜像容器内运行 "
                "FAIL_TO_PASS/PASS_TO_PASS 的判定结果（resolved）；未使用 LLM 打分。"
            )
        else:
            objective_description = (
                "主指标为最终 patch 在原始源码副本上运行官方 pytest 的结果；未使用 LLM 打分。"
            )
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "generated_at": generated_at.isoformat(),
            "dataset_digest": self.dataset.digest,
            "dataset_version": sorted({item.version for item in self.dataset}),
            "common_problem_ids": sorted(common_ids),
            "warnings": warnings,
            "metrics": [item.model_dump(mode="json") for item in metrics],
            "failure_clusters": clusters,
            "bad_cases": bad_cases,
            "manual_audit": self.repository.audit_agreement(run_ids),
            "objective_metric": objective_description,
            "objective_grader_sources": grader_sources,
        }
        (report_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_metrics_csv(metrics, report_dir / "metrics.csv")
        self._plot_pass(metrics, report_dir / "pass_at_1.png")
        self._plot_errors(metrics, report_dir / "error_distribution.png")
        self._plot_tokens(runs, report_dir / "token_success.png")
        (report_dir / "report.md").write_text(
            self._markdown(payload, metrics, clusters, bad_cases), encoding="utf-8"
        )
        pointer = {
            "report_dir": str(report_dir.resolve()),
            "report": str((report_dir / "report.md").resolve()),
            "generated_at": generated_at.isoformat(),
        }
        (self.reports_root / "latest.json").write_text(
            json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report_dir

    def _common_coverage(
        self, runs: list[RunTrace]
    ) -> tuple[list[RunTrace], set[str], list[str]]:
        by_group: dict[str, set[str]] = defaultdict(set)
        for run in runs:
            group = f"{run.agent_name}/{run.strategy or '-'}"
            by_group[group].add(run.problem_id)
        common = set.intersection(*by_group.values()) if by_group else set()
        filtered = [run for run in runs if run.problem_id in common]
        warnings: list[str] = []
        for group, problems in sorted(by_group.items()):
            excluded = len(problems - common)
            if excluded:
                warnings.append(f"{group} 有 {excluded} 题未纳入横向对比（非共同题）。")
        digests = {
            str(run.metadata.get("dataset_digest", ""))
            for run in filtered
            if run.metadata.get("dataset_digest")
        }
        if digests != {self.dataset.digest}:
            warnings.append("部分 run 的数据集指纹与当前评测集不同，请核对版本。")
        return filtered, common, warnings

    def _export_bad_cases(
        self,
        runs: list[RunTrace],
        attributions: list[Attribution],
        report_dir: Path,
    ) -> list[dict[str, Any]]:
        attribution_by_run = {item.run_id: item for item in attributions}
        grouped: dict[str, list[RunTrace]] = defaultdict(list)
        for run in runs:
            if not run.passed:
                grouped[run.agent_name].append(run)
        cases: list[dict[str, Any]] = []
        for agent, failures in sorted(grouped.items()):
            for run in sorted(failures, key=lambda item: item.run_id)[:2]:
                filename = f"{_safe(run.run_id)}.json"
                (report_dir / "traces" / filename).write_text(
                    run.model_dump_json(indent=2), encoding="utf-8"
                )
                attribution = attribution_by_run.get(run.run_id)
                cases.append(
                    {
                        "agent_name": agent,
                        "run_id": run.run_id,
                        "problem_id": run.problem_id,
                        "category": attribution.category.value if attribution else "unknown",
                        "rationale": attribution.rationale if attribution else "尚未归因",
                        "trace_link": f"traces/{filename}",
                    }
                )
        return cases

    def _write_metrics_csv(self, metrics: list[MetricsResult], path: Path) -> None:
        rows = [
            item.model_dump(exclude={"repeat_pass_rates", "repeat_pass_rate_ci95"})
            for item in metrics
        ]
        pd.DataFrame(rows).to_csv(path, index=False)

    def _plot_pass(self, metrics: list[MetricsResult], path: Path) -> None:
        labels = [_label(item) for item in metrics]
        values = [item.pass_at_1 for item in metrics]
        figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
        axis.bar(labels, values, color="#3568b8")
        axis.set_ylim(0, 1)
        axis.set_ylabel("pass@1")
        axis.set_title("Objective test pass@1")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    def _plot_errors(self, metrics: list[MetricsResult], path: Path) -> None:
        labels = [_label(item) for item in metrics]
        categories = sorted({key for item in metrics for key in item.error_counts})
        figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.5))
        bottom = [0.0] * len(metrics)
        for category in categories:
            values = [item.error_counts.get(category, 0) for item in metrics]
            axis.bar(labels, values, bottom=bottom, label=category)
            bottom = [left + right for left, right in zip(bottom, values, strict=True)]
        axis.set_ylabel("failed runs")
        axis.set_title("Failure attribution distribution")
        axis.tick_params(axis="x", rotation=25)
        if categories:
            axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    def _plot_tokens(self, runs: list[RunTrace], path: Path) -> None:
        agents = sorted({run.agent_name for run in runs})
        colors = {agent: plt.cm.tab10(index % 10) for index, agent in enumerate(agents)}
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for agent in agents:
            selected = [run for run in runs if run.agent_name == agent]
            axis.scatter(
                [run.total_tokens for run in selected],
                [int(run.passed) for run in selected],
                alpha=0.7,
                label=agent,
                color=colors[agent],
            )
        axis.set_xlabel("tokens per run")
        axis.set_ylabel("official tests passed (0/1)")
        axis.set_yticks([0, 1])
        axis.set_title("Token cost vs objective success")
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    def _markdown(
        self,
        payload: dict[str, Any],
        metrics: list[MetricsResult],
        clusters: list[dict[str, Any]],
        bad_cases: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"# Agent 横向评测报告：{payload['experiment_id']}",
            "",
            f"生成时间：{payload['generated_at']}  ",
            f"数据集版本：{', '.join(payload['dataset_version'])}  ",
            f"数据集 SHA-256：`{payload['dataset_digest']}`  ",
            f"共同题目数：{len(payload['common_problem_ids'])}",
            "",
            f"> {payload['objective_metric']}",
            "> pass@k 的 k 是独立生成次数，Agent 内部重试不计入 k。"
            "延迟为步骤耗时，0/缺失值不参与分位数。",
            "",
        ]
        lines.extend(f"> 警告：{warning}" for warning in payload["warnings"])
        if payload["warnings"]:
            lines.append("")
        lines.extend(
            [
                "## 指标对比",
                "",
                "| Agent | 策略 | runs | repeats | pass@1 | pass@k | repeat 均值 | "
                "方差 | Top-1 | Top-3 | token/run | P50 ms | P95 ms | 重试成功率 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in metrics:
            lines.append(
                f"| {item.agent_name} | {item.strategy} | {item.run_count} | {item.repeat_count} "
                f"| {_pct(item.pass_at_1)} | {_pct(item.pass_at_k)} "
                f"| {_pct(item.mean_repeat_pass_rate)} "
                f"| {_variance(item.repeat_pass_rate_variance)} "
                f"| {_pct(item.localization_top1)} | {_pct(item.localization_top3)} "
                f"| {item.avg_tokens_per_run:.1f} | {_number(item.latency_p50_ms)} "
                f"| {_number(item.latency_p95_ms)} | {_optional_pct(item.retry_success_rate)} |"
            )
        lines.extend(
            [
                "",
                "![pass@1](pass_at_1.png)",
                "",
                "## 失败归因分布",
                "",
                "![错误类别分布](error_distribution.png)",
                "",
                "| Agent | 策略 | 类别 | 失败数 | 失败内占比 |",
                "|---|---|---|---:|---:|",
            ]
        )
        for item in metrics:
            for category, count in item.error_counts.items():
                lines.append(
                    f"| {item.agent_name} | {item.strategy} | {category} | {count} "
                    f"| {_pct(item.error_distribution[category])} |"
                )
        lines.extend(["", "## 自动对比结论", ""])
        lines.extend(self._conclusions(metrics))
        lines.extend(["", "## 失效模式聚类（TF-IDF + KMeans）", ""])
        if not clusters:
            lines.append("没有失败 run，未生成聚类。")
        for cluster in clusters:
            lines.append(
                f"- 簇 {cluster['cluster_id']}：{cluster['size']} 条；主要类别 "
                f"`{cluster['category'] or 'unknown'}`；代表 run "
                f"`{cluster['representative_run_id']}`；"
                f"特征词：{', '.join(cluster['top_terms'][:5])}。"
            )
        lines.extend(["", "## 典型 bad case", ""])
        if not bad_cases:
            lines.append("没有失败 run。")
        for case in bad_cases:
            lines.append(
                f"- `{case['agent_name']}` / `{case['problem_id']}` / `{case['category']}`："
                f"{case['rationale']} [trace]({case['trace_link']})"
            )
        audit = payload["manual_audit"]
        lines.extend(
            [
                "",
                "## 归因结果人工抽检",
                "",
                f"- 已抽检：{audit['sample_count']} 条",
                f"- 一致率：{_optional_pct(audit['agreement_rate'])}",
                "- 抽检方式：逐条核对失败 trace 的断言报错与工具调用记录。",
                "- 未抽检时不展示或推测一致率；归因只做失败分类，不干预客观 pass/fail。",
                "",
                "## Token 与成功率",
                "",
                "![token 成功率散点](token_success.png)",
                "",
            ]
        )
        return "\n".join(lines)

    def _conclusions(self, metrics: list[MetricsResult]) -> list[str]:
        if len(metrics) < 2:
            return ["当前只有一个 Agent/策略分组，无法给出横向差异结论。"]
        conclusions: list[str] = []
        for left, right in combinations(metrics, 2):
            categories = set(left.error_distribution) | set(right.error_distribution)
            if categories:
                category = max(
                    categories,
                    key=lambda key: abs(
                        left.error_distribution.get(key, 0)
                        - right.error_distribution.get(key, 0)
                    ),
                )
                delta = (
                    left.error_distribution.get(category, 0)
                    - right.error_distribution.get(category, 0)
                ) * 100
                category_text = f"失败占比差最大的类别是 `{category}`（{delta:+.2f} 个百分点）"
            else:
                category_text = "两组均无失败，没有错误分布差异"
            pass_delta = (left.pass_at_1 - right.pass_at_1) * 100
            line = (
                f"- `{_label(left)}` 相对 `{_label(right)}` 的 pass@1 差值为 "
                f"{pass_delta:+.2f} 个百分点；{category_text}"
            )
            # pass@1 打平时，正确率没有区分度，但过程指标可能有——把差异显式讲出来，
            # 否则报告只剩一堆"0.00 个百分点"，掩盖了定位/成本/延迟上的真实差别。
            if abs(pass_delta) < 1e-9:
                loc_delta = (left.localization_top1 - right.localization_top1) * 100
                details: list[str] = []
                if abs(loc_delta) >= 1e-9:
                    details.append(f"定位 Top-1 差 {loc_delta:+.2f} 个百分点")
                # token=0 表示原日志未记录，不参与对比（未知≠零消耗）
                if left.avg_tokens_per_run > 0 and right.avg_tokens_per_run > 0:
                    tok_delta = left.avg_tokens_per_run - right.avg_tokens_per_run
                    details.append(f"平均 token/题 差 {tok_delta:+.1f}")
                if left.latency_p50_ms and right.latency_p50_ms:
                    lat_delta = left.latency_p50_ms - right.latency_p50_ms
                    details.append(f"P50 步骤延迟差 {lat_delta:+.1f} ms")
                if details:
                    line += "；但两者过程不同：" + "，".join(details)
                line += "（pass@1 已触及天花板，差异需看过程指标）"
            conclusions.append(line + "。")
        return conclusions

    def _new_report_dir(self, experiment_id: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return self.reports_root / f"{_safe(experiment_id)}_{timestamp}"


def _label(item: MetricsResult) -> str:
    return f"{item.agent_name}/{item.strategy}"


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "run"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _optional_pct(value: float | None) -> str:
    return _pct(value) if value is not None else "N/A"


def _number(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


def _variance(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "N/A"
