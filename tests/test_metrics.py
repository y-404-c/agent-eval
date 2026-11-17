from datetime import date, datetime, timezone

import pytest

from agent_eval.domain import (
    Attribution,
    ErrorCategory,
    Problem,
    RunTrace,
    Step,
    TestOutcome,
)
from agent_eval.domain import (
    TestStatus as TraceStatus,
)
from agent_eval.metrics import MetricsCalculator


def make_problem(problem_id: str, gold_files: list[str] | None = None) -> Problem:
    gold_files = gold_files or [f"src/{problem_id}.py"]
    return Problem(
        problem_id=problem_id,
        source="test",
        version="1",
        added_at=date(2026, 1, 1),
        entry_path=gold_files[0],
        test_path=f"tests/test_{problem_id}.py",
        source_code="",
        test_code="",
        gold_files=gold_files,
    )


def make_trace(
    run_id: str,
    problem_id: str = "p1",
    status: TraceStatus = TraceStatus.FAIL,
    *,
    agent_name: str = "agent-a",
    strategy: str | None = None,
    attempt_index: int = 0,
    retry_count: int = 0,
    steps: list[Step] | None = None,
    error_category: ErrorCategory | None = None,
) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        agent_name=agent_name,
        strategy=strategy,
        problem_id=problem_id,
        test_result=status,
        attempt_index=attempt_index,
        retry_count=retry_count,
        steps=steps or [],
        error_category=error_category,
    )


def step(
    step_id: int,
    *,
    action: str = "think",
    args: dict | None = None,
    tokens: int = 0,
    latency_ms: int = 0,
) -> Step:
    return Step(
        step_id=step_id,
        action=action,
        args=args or {},
        tokens=tokens,
        latency_ms=latency_ms,
    )


def attribution(run_id: str, category: ErrorCategory) -> Attribution:
    return Attribution(
        run_id=run_id,
        category=category,
        method="manual",
        confidence=1,
        rationale="test",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_pass_at_k_repeat_statistics_tokens_latency_and_retry() -> None:
    problems = [make_problem("p1"), make_problem("p2")]
    traces = [
        make_trace(
            "p1-r2",
            attempt_index=2,
            steps=[step(1, tokens=30, latency_ms=300)],
            error_category=ErrorCategory.SYNTAX_ERROR,
        ),
        make_trace(
            "p2-r0",
            problem_id="p2",
            status=TraceStatus.PASS,
            steps=[step(1, tokens=40, latency_ms=400)],
        ),
        make_trace(
            "p1-r0",
            attempt_index=0,
            retry_count=1,
            steps=[
                step(1, tokens=6, latency_ms=0),
                step(2, tokens=4, latency_ms=100),
            ],
            error_category=ErrorCategory.INCOMPLETE_FIX,
        ),
        make_trace(
            "p1-r1",
            status=TraceStatus.PASS,
            attempt_index=1,
            retry_count=2,
            steps=[step(1, tokens=20, latency_ms=200)],
        ),
    ]

    result = MetricsCalculator.calculate(
        traces,
        problems,
        k=2,
        attributions=[attribution("p1-r0", ErrorCategory.REGRESSION)],
    )[0]

    assert (result.run_count, result.problem_count, result.repeat_count) == (4, 2, 3)
    assert result.pass_at_1 == pytest.approx(0.5)
    assert result.pass_at_k == pytest.approx(1.0)
    assert result.repeat_pass_rates == pytest.approx([0.5, 1.0, 0.0])
    assert result.mean_repeat_pass_rate == pytest.approx(0.5)
    assert result.repeat_pass_rate_variance == pytest.approx(0.25)
    assert result.repeat_pass_rate_ci95.lower == pytest.approx(0.1500, abs=1e-4)
    assert result.repeat_pass_rate_ci95.upper == pytest.approx(0.8500, abs=1e-4)
    assert result.avg_tokens_per_run == pytest.approx(25)
    assert result.avg_tokens_per_problem == pytest.approx(50)
    assert result.latency_p50_ms == pytest.approx(250)
    assert result.latency_p95_ms == pytest.approx(385)
    assert result.retry_success_rate == pytest.approx(0.5)
    assert result.failure_count == 2
    assert result.error_counts == {"regression": 1, "syntax_error": 1}
    assert result.error_distribution == {"regression": 0.5, "syntax_error": 0.5}


def test_pass_at_k_does_not_count_internal_retries_as_generations() -> None:
    trace = make_trace("only-run", retry_count=3)
    trace.test_history = [TestOutcome(attempt=1, status=TraceStatus.PASS)]

    result = MetricsCalculator.calculate([trace], [make_problem("p1")], k=3)[0]

    assert result.repeat_count == 1
    assert result.pass_at_1 == 0
    assert result.pass_at_k == 0
    assert result.retry_success_rate == 0


def test_localization_uses_ordered_reads_and_all_runs_as_denominator() -> None:
    problem = make_problem("p1", ["src/foo.py"])
    traces = [
        make_trace(
            "wrong-then-gold",
            steps=[
                step(1, action="read", args={"path": "src/wrong.py"}),
                step(2, action="read_file", args={"path": "/workspace/src/foo.py"}),
            ],
        ),
        make_trace(
            "windows-gold",
            attempt_index=1,
            steps=[step(1, action="ast_slice", args={"file_path": r".\src\foo.py"})],
        ),
        make_trace(
            "missing-path-then-gold",
            attempt_index=2,
            steps=[
                step(1, action="read_file"),
                step(2, action="read", args={"file": "foo.py"}),
            ],
        ),
        make_trace(
            "no-reads",
            attempt_index=3,
            steps=[step(1, action="edit_file", args={"path": "src/foo.py"})],
        ),
    ]

    result = MetricsCalculator.calculate(traces, [problem])[0]

    assert result.localization_top1 == pytest.approx(0.25)
    assert result.localization_top3 == pytest.approx(0.75)


def test_error_distribution_prefers_attribution_and_ignores_passed_runs() -> None:
    traces = [
        make_trace("syntax", error_category=ErrorCategory.SYNTAX_ERROR),
        make_trace("assertion", error_category=ErrorCategory.ASSERTION_MISMATCH),
        make_trace("uncategorized"),
        make_trace("passed", status=TraceStatus.PASS),
    ]

    result = MetricsCalculator.calculate(
        traces,
        [make_problem("p1")],
        attributions=[attribution("syntax", ErrorCategory.WRONG_LOCATION)],
    )[0]

    assert result.failure_count == 3
    assert result.error_counts == {
        "assertion_mismatch": 1,
        "unknown": 1,
        "wrong_location": 1,
    }
    assert result.error_distribution == pytest.approx(
        {"assertion_mismatch": 1 / 3, "unknown": 1 / 3, "wrong_location": 1 / 3}
    )


def test_grouping_normalizes_blank_strategy() -> None:
    traces = [
        make_trace("a1", strategy=None),
        make_trace("a2", strategy="  "),
        make_trace("b1", agent_name="agent-b", strategy=" S1 "),
    ]

    results = MetricsCalculator.calculate(traces, [make_problem("p1")])

    assert [(item.agent_name, item.strategy) for item in results] == [
        ("agent-a", "-"),
        ("agent-b", "S1"),
    ]
    assert results[0].run_count == 2


def test_empty_input_and_missing_optional_samples_are_safe() -> None:
    assert MetricsCalculator.calculate([], []) == []

    result = MetricsCalculator.calculate([make_trace("empty")], [])[0]

    assert result.pass_at_1 == 0
    assert result.pass_at_k == 0
    assert result.repeat_pass_rates == [0]
    assert result.repeat_pass_rate_variance is None
    assert result.localization_top1 == 0
    assert result.localization_top3 == 0
    assert result.avg_tokens_per_run == 0
    assert result.avg_tokens_per_problem == 0
    assert result.latency_p50_ms is None
    assert result.latency_p95_ms is None
    assert result.retry_success_rate is None
    assert result.error_counts == {"unknown": 1}
    assert result.error_distribution == {"unknown": 1}

    with pytest.raises(ValueError, match="k"):
        MetricsCalculator.calculate([], [], k=0)
