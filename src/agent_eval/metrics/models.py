from __future__ import annotations

from pydantic import BaseModel, Field


class WilsonInterval(BaseModel):
    """二项分布通过率的 95% Wilson 区间。"""

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)


class MetricsResult(BaseModel):
    """一个 Agent 与策略组合在一组独立生成轨迹上的客观指标。"""

    agent_name: str
    strategy: str
    run_count: int = Field(ge=0)
    problem_count: int = Field(ge=0)
    repeat_count: int = Field(ge=0)
    k: int = Field(ge=1)
    pass_at_1: float = Field(ge=0, le=1)
    pass_at_k: float = Field(ge=0, le=1)
    repeat_pass_rates: list[float]
    mean_repeat_pass_rate: float = Field(ge=0, le=1)
    repeat_pass_rate_variance: float | None = Field(default=None, ge=0)
    repeat_pass_rate_ci95: WilsonInterval
    localization_top1: float = Field(ge=0, le=1)
    localization_top3: float = Field(ge=0, le=1)
    avg_tokens_per_run: float = Field(ge=0)
    avg_tokens_per_problem: float = Field(ge=0)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    retry_success_rate: float | None = Field(default=None, ge=0, le=1)
    failure_count: int = Field(ge=0)
    error_counts: dict[str, int]
    error_distribution: dict[str, float]


GroupMetrics = MetricsResult
