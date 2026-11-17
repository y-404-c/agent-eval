from __future__ import annotations

import json
import posixpath
import re
from datetime import datetime, timezone
from typing import Any

from agent_eval.domain import (
    Attribution,
    ErrorCategory,
    Problem,
    RunTrace,
    Step,
    TestOutcome,
    TestStatus,
)

_ENVIRONMENT_RE = re.compile(
    r"(?:\btimeout\b|\btimed out\b|\btime limit exceeded\b|"
    r"\bout of memory\b|\boom(?:killed)?\b|\bmemoryerror\b|"
    r"\bmodulenotfounderror\b|no module named|"
    r"dependency (?:is )?missing|missing dependency|package (?:is )?not installed)",
    re.IGNORECASE,
)
_SYNTAX_RE = re.compile(
    r"(?:\bSyntaxError\b|\bIndentationError\b|\bTabError\b|"
    r"invalid syntax|unexpected indent|unindent does not match|"
    r"expected an indented block)",
    re.IGNORECASE,
)
_PATCH_ERROR_RE = re.compile(
    r"(?:failed to apply (?:the )?patch|patch (?:application|apply) failed|"
    r"patch does not apply|cannot apply (?:the )?patch|corrupt patch|"
    r"malformed patch|hunk(?:s)? failed|apply_patch (?:failed|error))",
    re.IGNORECASE,
)
_ASSERTION_RE = re.compile(
    r"(?:\bAssertionError\b|\bassertion failed\b|^\s*E\s+assert\b)",
    re.IGNORECASE | re.MULTILINE,
)
_LOOP_STOP_REASONS = {
    "loop",
    "loop_exhausted",
    "max_steps",
    "max_iterations",
    "iteration_limit",
    "step_limit",
}


def _normalize_action(action: str) -> str:
    return " ".join(action.strip().casefold().split())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def _normalized_args(args: dict[str, Any]) -> str:
    return json.dumps(
        _normalize_value(args),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path.strip().replace("\\", "/"))
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def _path_matches(candidate: str, gold: str) -> bool:
    return (
        candidate == gold
        or candidate.endswith(f"/{gold}")
        or gold.endswith(f"/{candidate}")
    )


def _matching_line(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    if end == -1:
        end = len(text)
    return text[start:end].strip()[:400] or match.group(0)


def _short(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def _looks_like_unified_diff(patch: str) -> bool:
    if re.search(r"^diff --git ", patch, re.MULTILINE):
        return True
    return all(
        re.search(pattern, patch, re.MULTILINE) for pattern in (r"^--- ", r"^\+\+\+ ", r"^@@ ")
    )


class RuleBasedAttributor:
    """按固定优先级归因；测试通过时不产生失败归因。"""

    repeat_threshold: int = 3

    def attribute(self, trace: RunTrace, problem: Problem) -> Attribution | None:
        if trace.passed:
            return None

        latest = trace.test_history[-1] if trace.test_history else None
        test_text = self._current_test_text(trace, latest)

        environment = self._environment(trace, latest, test_text)
        if environment is not None:
            return self._result(
                trace,
                ErrorCategory.ENVIRONMENT,
                0.99,
                environment[0],
                environment[1],
            )

        syntax = _matching_line(test_text, _SYNTAX_RE)
        if syntax is not None:
            return self._result(
                trace,
                ErrorCategory.SYNTAX_ERROR,
                0.98,
                "最终测试输出包含 Python 语法或缩进错误。",
                [f"syntax_error 规则：最终测试报错为 {syntax!r}。"],
            )

        loop = self._loop_exhausted(trace)
        if loop is not None:
            return self._result(trace, ErrorCategory.LOOP_EXHAUSTED, 0.96, loop[0], loop[1])

        patch_issue = self._patch_application_issue(trace, test_text)
        failed_steps = [step for step in trace.steps if not step.ok or step.error]
        has_patch = _looks_like_unified_diff(trace.final_patch) and patch_issue is None
        if patch_issue is not None or (failed_steps and not has_patch):
            evidence: list[str] = []
            if patch_issue is not None:
                evidence.append(f"tool_failure 规则：检测到 patch 应用错误：{patch_issue}")
            if failed_steps:
                evidence.extend(self._step_failure_evidence(step) for step in failed_steps[:3])
            if not trace.final_patch.strip():
                patch_evidence = "为空，没有有效修复 patch。"
            elif patch_issue is not None:
                patch_evidence = "非空，但 patch 应用失败，因此不视为有效修复。"
            else:
                patch_evidence = "非空，但不是可识别的 unified diff，不视为有效修复。"
            evidence.append("tool_failure 规则：final_patch " + patch_evidence)
            return self._result(
                trace,
                ErrorCategory.TOOL_FAILURE,
                0.94,
                "工具失败阻断了有效 patch，或生成的 patch 无法应用。",
                evidence,
            )

        changed = {_normalize_path(path) for path in trace.changed_files if path.strip()}
        gold = {_normalize_path(path) for path in problem.gold_files if path.strip()}
        hits = sorted(
            path
            for path in changed
            if any(_path_matches(path, gold_path) for gold_path in gold)
        )
        if changed and not hits:
            return self._result(
                trace,
                ErrorCategory.WRONG_LOCATION,
                0.94,
                "轨迹记录了代码改动，但改动文件完全没有命中该题 gold 文件。",
                [
                    "wrong_location 规则：changed_files=" + ", ".join(sorted(changed)) + "。",
                    "wrong_location 规则：problem.gold_files="
                    + ", ".join(sorted(gold))
                    + "；交集为空。",
                ],
            )

        regression = self._regression(trace.test_history)
        if regression is not None:
            resolved, introduced = regression
            return self._result(
                trace,
                ErrorCategory.REGRESSION,
                0.95,
                "最终测试中既有历史失败消失，又出现了此前从未失败的新测试。",
                [
                    "regression 规则：旧失败已消失：" + ", ".join(resolved) + "。",
                    "regression 规则：全新失败：" + ", ".join(introduced) + "。",
                ],
            )

        assertion = _matching_line(test_text, _ASSERTION_RE)
        if assertion is not None and hits:
            return self._result(
                trace,
                ErrorCategory.ASSERTION_MISMATCH,
                0.88,
                "修复位置命中 gold，但最终测试仍出现断言不一致。",
                [
                    "assertion_mismatch 规则：最终测试断言报错为 " + repr(assertion) + "。",
                    "assertion_mismatch 规则：改动命中 gold 文件：" + ", ".join(hits) + "。",
                ],
            )

        return self._incomplete_fix(trace, latest, changed, gold, hits)

    def _current_test_text(self, trace: RunTrace, latest: TestOutcome | None) -> str:
        parts = [trace.test_output]
        if latest is not None:
            parts.append(latest.output)
        return "\n".join(part for part in parts if part)

    def _environment(
        self,
        trace: RunTrace,
        latest: TestOutcome | None,
        test_text: str,
    ) -> tuple[str, list[str]] | None:
        if trace.test_result == TestStatus.TIMEOUT or (
            latest is not None and latest.status == TestStatus.TIMEOUT
        ):
            attempt = latest.attempt if latest is not None else "最终"
            return (
                "运行以 timeout 状态结束，优先归为不可纠的环境超时。",
                [f"environment 规则：test_result/TestOutcome({attempt}) 状态为 timeout。"],
            )
        matched = _matching_line(test_text, _ENVIRONMENT_RE)
        if matched is None:
            return None
        return (
            "最终测试输出包含超时、OOM 或依赖缺失信号。",
            [f"environment 规则：最终测试报错为 {matched!r}。"],
        )

    def _loop_exhausted(self, trace: RunTrace) -> tuple[str, list[str]] | None:
        stop_reason = str(trace.metadata.get("stop_reason", "")).strip().casefold()
        stop_reason = stop_reason.replace("-", "_").replace(" ", "_")
        if stop_reason in _LOOP_STOP_REASONS:
            return (
                f"执行器以 {stop_reason} 停止，动作循环或步数预算已经耗尽。",
                [f"loop_exhausted 规则：metadata.stop_reason={stop_reason!r}。"],
            )

        previous_signature: tuple[str, str] | None = None
        run_start = 0
        for index, step in enumerate(trace.steps):
            signature = (_normalize_action(step.action), _normalized_args(step.args))
            if signature != previous_signature:
                previous_signature = signature
                run_start = index
                continue
            run_length = index - run_start + 1
            if run_length >= self.repeat_threshold:
                repeated = trace.steps[run_start : index + 1]
                step_ids = ", ".join(str(item.step_id) for item in repeated)
                return (
                    f"同一 action 与规范化 args 连续重复 {run_length} 次。",
                    [
                        "loop_exhausted 规则：连续重复步骤 step_id=" + step_ids + "。",
                        "loop_exhausted 规则：action="
                        + repr(signature[0])
                        + "，规范化 args="
                        + signature[1]
                        + "。",
                    ],
                )
        return None

    def _patch_application_issue(self, trace: RunTrace, test_text: str) -> str | None:
        matched = _matching_line(test_text, _PATCH_ERROR_RE)
        if matched is not None:
            return repr(matched)
        for step in trace.steps:
            if step.ok and not step.error:
                continue
            text = " ".join(part for part in (step.error, step.observation) if part)
            if "patch" in _normalize_action(step.action) or _PATCH_ERROR_RE.search(text):
                return f"step_id={step.step_id}, action={step.action!r}, error={_short(text)!r}"
        return None

    def _step_failure_evidence(self, step: Step) -> str:
        detail = step.error or step.observation or "ok=false"
        return (
            f"tool_failure 规则：step_id={step.step_id}, action={step.action!r}, "
            f"失败信息={_short(detail)!r}。"
        )

    def _regression(self, history: list[TestOutcome]) -> tuple[list[str], list[str]] | None:
        if len(history) < 2 or not history[-1].failed_tests:
            return None
        previous = {
            name for outcome in history[:-1] for name in outcome.failed_tests if name.strip()
        }
        current = {name for name in history[-1].failed_tests if name.strip()}
        resolved = sorted(previous - current)
        introduced = sorted(current - previous)
        if resolved and introduced:
            return resolved, introduced
        return None

    def _incomplete_fix(
        self,
        trace: RunTrace,
        latest: TestOutcome | None,
        changed: set[str],
        gold: set[str],
        hits: set[str],
    ) -> Attribution:
        evidence = [
            "incomplete_fix 兜底规则：更高优先级规则均未命中，最终 run 仍未通过。",
            f"incomplete_fix 规则：test_result={trace.test_result.value!r}。",
        ]
        if latest is not None:
            evidence.append(
                f"incomplete_fix 规则：attempt={latest.attempt}, passed={latest.passed}, "
                f"failed={latest.failed}, errors={latest.errors}。"
            )
            if latest.failed_tests:
                evidence.append(
                    "incomplete_fix 规则：剩余失败测试=" + ", ".join(latest.failed_tests) + "。"
                )
        evidence.append(
            "incomplete_fix 规则：changed_files/gold_files 命中="
            + (", ".join(hits) if hits else "无")
            + f"；changed={len(changed)}, gold={len(gold)}。"
        )
        return self._result(
            trace,
            ErrorCategory.INCOMPLETE_FIX,
            0.4,
            "仍有无法由硬规则进一步区分的失败，按不完整修复低置信度处理并送人工复核。",
            evidence,
            needs_review=True,
        )

    def _result(
        self,
        trace: RunTrace,
        category: ErrorCategory,
        confidence: float,
        rationale: str,
        evidence: list[str],
        needs_review: bool = False,
    ) -> Attribution:
        return Attribution(
            run_id=trace.run_id,
            category=category,
            method="rule",
            confidence=confidence,
            rationale=rationale,
            evidence=evidence,
            needs_review=needs_review,
            created_at=datetime.now(timezone.utc),
        )
