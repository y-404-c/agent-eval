from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from agent_eval.attribution import FailureClusterer, LLMAssistedAttributor, RuleBasedAttributor
from agent_eval.domain import (
    ErrorCategory,
    Problem,
    RunTrace,
    Step,
    TestOutcome,
)
from agent_eval.domain import TestStatus as DomainTestStatus


def make_problem(gold_files: list[str] | None = None) -> Problem:
    return Problem(
        problem_id="p1",
        source="unit",
        version="1",
        added_at=date(2024, 1, 1),
        entry_path="src/target.py",
        test_path="tests/test_target.py",
        source_code="def target():\n    return 0\n",
        test_code="def test_target():\n    assert target() == 1\n",
        gold_files=gold_files or ["src/target.py"],
    )


def make_trace(run_id: str = "r1", **kwargs: Any) -> RunTrace:
    values: dict[str, Any] = {
        "run_id": run_id,
        "agent_name": "test-agent",
        "problem_id": "p1",
        "test_result": DomainTestStatus.FAIL,
    }
    values.update(kwargs)
    return RunTrace(**values)


def outcome(
    attempt: int,
    failed_tests: list[str],
    *,
    output: str = "",
    status: DomainTestStatus = DomainTestStatus.FAIL,
) -> TestOutcome:
    return TestOutcome(
        attempt=attempt,
        status=status,
        output=output,
        failed=len(failed_tests),
        failed_tests=failed_tests,
    )


def test_pass_has_no_attribution() -> None:
    trace = make_trace(test_result=DomainTestStatus.PASS, test_output="SyntaxError: stale output")

    assert RuleBasedAttributor().attribute(trace, make_problem()) is None


@pytest.mark.parametrize(
    ("trace", "expected_fragment"),
    [
        (make_trace(test_result=DomainTestStatus.TIMEOUT), "timeout"),
        (make_trace(test_output="Fatal Python error: Out of memory"), "Out of memory"),
        (
            make_trace(
                test_history=[
                    outcome(
                        1,
                        ["test_target"],
                        output="ModuleNotFoundError: No module named 'numpy'",
                    )
                ]
            ),
            "ModuleNotFoundError",
        ),
    ],
)
def test_environment_rules(trace: RunTrace, expected_fragment: str) -> None:
    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.ENVIRONMENT
    assert expected_fragment in " ".join(result.evidence)


def test_environment_has_highest_priority() -> None:
    trace = make_trace(
        test_output="SyntaxError: invalid syntax\nOut of memory",
        steps=[Step(step_id=number, action="edit", args={"path": "x"}) for number in range(1, 4)],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.ENVIRONMENT


def test_syntax_error_precedes_repeated_actions() -> None:
    trace = make_trace(
        test_history=[outcome(1, [], output='File "x.py", line 1\nSyntaxError: invalid syntax')],
        steps=[Step(step_id=number, action="read", args={}) for number in range(1, 4)],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.SYNTAX_ERROR


def test_loop_detects_three_consecutive_actions_with_normalized_args() -> None:
    trace = make_trace(
        steps=[
            Step(step_id=1, action=" edit_file ", args={"path": " src/a.py ", "line": 1}),
            Step(step_id=2, action="EDIT_FILE", args={"line": 1, "path": "src/a.py"}),
            Step(step_id=3, action="edit_file", args={"path": "src/a.py", "line": 1}),
        ]
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.LOOP_EXHAUSTED
    assert "step_id=1, 2, 3" in " ".join(result.evidence)


def test_loop_requires_consecutive_repetition() -> None:
    trace = make_trace(
        steps=[
            Step(step_id=1, action="read", args={"path": "a.py"}),
            Step(step_id=2, action="read", args={"path": "a.py"}),
            Step(step_id=3, action="test", args={}),
            Step(step_id=4, action="read", args={"path": "a.py"}),
        ]
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.INCOMPLETE_FIX
    assert result.needs_review


@pytest.mark.parametrize("reason", ["loop", "max_steps", "max-steps"])
def test_loop_honours_executor_stop_reason(reason: str) -> None:
    trace = make_trace(metadata={"stop_reason": reason})

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.LOOP_EXHAUSTED


@pytest.mark.parametrize("final_patch", ["", "not a unified diff"])
def test_tool_failure_without_valid_patch_precedes_wrong_location(final_patch: str) -> None:
    trace = make_trace(
        final_patch=final_patch,
        steps=[Step(step_id=1, action="edit_file", ok=False, error="permission denied")],
        changed_files=["src/other.py"],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.TOOL_FAILURE
    assert "permission denied" in " ".join(result.evidence)


def test_patch_application_error_is_tool_failure_even_with_diff() -> None:
    trace = make_trace(
        final_patch="diff --git a/src/target.py b/src/target.py",
        changed_files=["src/target.py"],
        steps=[Step(step_id=1, action="apply_patch", ok=False, error="patch does not apply")],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.TOOL_FAILURE


def test_wrong_location_normalizes_paths_and_precedes_regression() -> None:
    trace = make_trace(
        changed_files=[".\\src\\other.py"],
        test_history=[
            outcome(1, ["test_old"]),
            outcome(2, ["test_new"], output="AssertionError: new failure"),
        ],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.WRONG_LOCATION
    assert "src/other.py" in " ".join(result.evidence)


def test_regression_requires_resolved_and_brand_new_failures() -> None:
    trace = make_trace(
        changed_files=["src/target.py"],
        test_history=[
            outcome(1, ["test_old", "test_shared"]),
            outcome(2, ["test_shared", "test_new"], output="AssertionError: 1 != 2"),
        ],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.REGRESSION
    assert "test_old" in result.evidence[0]
    assert "test_new" in result.evidence[1]


def test_additional_failure_without_resolved_failure_is_not_regression() -> None:
    trace = make_trace(
        changed_files=["src/target.py"],
        test_history=[outcome(1, ["test_old"]), outcome(2, ["test_old", "test_new"])],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.INCOMPLETE_FIX


def test_assertion_mismatch_requires_gold_file_hit() -> None:
    trace = make_trace(
        changed_files=["src/target.py"],
        test_history=[outcome(1, ["test_target"], output="E assert 0 == 1")],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.ASSERTION_MISMATCH
    assert not result.needs_review


def test_other_remaining_failure_is_low_confidence_incomplete_fix() -> None:
    trace = make_trace(
        changed_files=["src/target.py"],
        final_patch="diff --git a/src/target.py b/src/target.py",
        test_history=[outcome(1, ["test_target"], output="TypeError: unsupported operand")],
    )

    result = RuleBasedAttributor().attribute(trace, make_problem())

    assert result is not None
    assert result.category == ErrorCategory.INCOMPLETE_FIX
    assert result.method == "rule"
    assert result.confidence < 0.5
    assert result.needs_review
    assert "test_target" in " ".join(result.evidence)


def test_llm_is_not_called_for_confident_rule_result() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"不应请求 LLM: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(unexpected_request))
    attributor = LLMAssistedAttributor(
        "http://llm.local/v1", "test-model", rules=RuleBasedAttributor(), client=client
    )
    trace = make_trace(test_output="SyntaxError: invalid syntax")

    result = attributor.attribute(trace, make_problem())

    assert result is not None
    assert result.method == "rule"
    assert result.category == ErrorCategory.SYNTAX_ERROR


def test_llm_assists_only_review_case_and_returns_restricted_category() -> None:
    requests: list[httpx.Request] = []

    def llm_response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"category": "incomplete_fix", "rationale": "修复逻辑仍不完整"}
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(llm_response))
    attributor = LLMAssistedAttributor(
        "http://llm.local/v1", "test-model", rules=RuleBasedAttributor(), client=client
    )

    result = attributor.attribute(make_trace(), make_problem())

    assert len(requests) == 1
    assert result is not None
    assert result.category == ErrorCategory.INCOMPLETE_FIX
    assert result.method == "llm_assisted"
    assert result.needs_review
    assert "人工抽检" in result.rationale


def test_llm_network_failure_falls_back_to_rule_without_fabrication() -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(transport=httpx.MockTransport(offline))
    attributor = LLMAssistedAttributor(
        "http://llm.local/v1", "test-model", rules=RuleBasedAttributor(), client=client
    )

    result = attributor.attribute(make_trace(), make_problem())

    assert result is not None
    assert result.method == "rule"
    assert result.category == ErrorCategory.INCOMPLETE_FIX
    assert result.needs_review


def test_clusterer_handles_empty_pass_and_single_failure() -> None:
    clusterer = FailureClusterer()
    passed = make_trace("passed", test_result=DomainTestStatus.PASS)
    failed = make_trace(
        "failed",
        steps=[Step(step_id=1, action="edit_file")],
        test_output="TypeError at /tmp/repo/src/target.py:12",
    )
    attribution = RuleBasedAttributor().attribute(failed, make_problem())
    assert attribution is not None

    assert clusterer.cluster([]) == []
    assert clusterer.cluster([passed]) == []
    result = clusterer.cluster([failed], [attribution])

    assert len(result) == 1
    assert result[0]["run_ids"] == ["failed"]
    assert result[0]["size"] == 1
    assert result[0]["category"] == ErrorCategory.INCOMPLETE_FIX.value
    assert result[0]["representative_run_id"] == "failed"
    assert "action_edit_file" in result[0]["top_terms"]


def test_clusterer_is_deterministic_and_small_sample_safe() -> None:
    traces = [
        make_trace(
            f"r{index}",
            steps=[Step(step_id=1, action=action)],
            test_output=error,
            error_category=category,
        )
        for index, (action, error, category) in enumerate(
            [
                ("apply_patch", "patch does not apply", ErrorCategory.TOOL_FAILURE),
                ("apply_patch", "malformed patch", ErrorCategory.TOOL_FAILURE),
                ("edit_file", "AssertionError value mismatch", ErrorCategory.ASSERTION_MISMATCH),
                ("edit_file", "AssertionError wrong result", ErrorCategory.ASSERTION_MISMATCH),
            ]
        )
    ]
    clusterer = FailureClusterer(n_clusters=3, random_state=7)

    first = clusterer.cluster(traces)
    second = clusterer.cluster(reversed(traces))

    assert first == second
    assert len(first) <= len(traces)
    assert sorted(run_id for item in first for run_id in item["run_ids"]) == [
        "r0",
        "r1",
        "r2",
        "r3",
    ]
    for item in first:
        assert item["size"] == len(item["run_ids"])
        assert item["representative_run_id"] in item["run_ids"]
        assert item["category"] in {
            ErrorCategory.TOOL_FAILURE.value,
            ErrorCategory.ASSERTION_MISMATCH.value,
        }
        assert item["top_terms"]


def test_clusterer_never_requests_more_clusters_than_samples() -> None:
    traces = [
        make_trace("a", steps=[Step(step_id=1, action="read")], test_output="SyntaxError"),
        make_trace("b", steps=[Step(step_id=1, action="edit")], test_output="AssertionError"),
    ]

    result = FailureClusterer(n_clusters=3).cluster(traces)

    assert len(result) <= 2
    assert sum(item["size"] for item in result) == 2
