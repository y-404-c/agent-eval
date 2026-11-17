from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from agent_eval.adapters.base import TraceAdapter
from agent_eval.adapters.ours import _patch_files
from agent_eval.domain import RunTrace, Step, TestOutcome, TestStatus

_COMMAND_RE = re.compile(r"```(?:bash|sh|shell|console)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_REPEAT_RE = re.compile(r"(?:repeat|rep|attempt|run)[-_](\d+)(?:\D|$)", re.IGNORECASE)
_COUNT_RES = {
    "passed": re.compile(r"(\d+) passed\b"),
    "failed": re.compile(r"(\d+) failed\b"),
    "errors": re.compile(r"(\d+) errors?\b"),
}


class MiniSweAgentAdapter(TraceAdapter):
    """适配 mini-SWE-agent 的 info+messages JSON，也兼容显式 steps 轨迹。"""

    name = "mini-swe-agent"

    def load(
        self,
        source: Path | str,
        *,
        experiment_id: str = "default",
        problem_id: str | None = None,
        attempt_index: int | None = None,
        attempt_offset: int = 0,
        dataset_digest: str = "",
        **_: Any,
    ) -> Iterable[RunTrace]:
        path = Path(source)
        files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"未找到 mini-SWE-agent JSON 轨迹: {path}")
        for file_path in files:
            for payload in self._payloads(file_path):
                yield self._convert(
                    payload,
                    file_path,
                    experiment_id=experiment_id,
                    problem_id_override=problem_id,
                    attempt_index=attempt_index,
                    attempt_offset=attempt_offset,
                    dataset_digest=dataset_digest,
                )

    def _payloads(self, path: Path) -> Iterator[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"mini-SWE-agent 轨迹必须是 JSON 对象: {path}")
            yield value

    def _convert(
        self,
        raw: dict[str, Any],
        source: Path,
        *,
        experiment_id: str,
        problem_id_override: str | None,
        attempt_index: int | None,
        attempt_offset: int,
        dataset_digest: str,
    ) -> RunTrace:
        info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
        problem_id = problem_id_override or _problem_id(raw, info)
        if not problem_id:
            raise ValueError(f"无法从轨迹提取 problem_id，请显式传入: {source}")

        if isinstance(raw.get("steps"), list):
            steps = [_structured_step(item, index) for index, item in enumerate(raw["steps"], 1)]
        else:
            messages = raw.get("messages") or raw.get("trajectory") or []
            steps = self._message_steps(messages)
        history: list[TestOutcome] = []
        for step in steps:
            if step.action != "run_test":
                continue
            outcome = _parse_test_output(step.observation, len(history) + 1)
            if outcome is not None:
                history.append(outcome)

        final_patch = str(
            raw.get("final_patch") or info.get("submission") or info.get("patch") or ""
        )
        if not _is_diff(final_patch):
            final_patch = _last_observed_diff(steps)
        run_id = str(raw.get("run_id") or info.get("run_id") or "").strip()
        if not run_id:
            path_hash = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
            run_id = f"mini-{source.stem}-{path_hash}"
        repeat = attempt_index
        if repeat is None:
            repeat = raw.get("attempt_index", info.get("repeat_index"))
        if repeat is None:
            match = _REPEAT_RE.search(source.stem)
            repeat = int(match.group(1)) if match else 0

        explicit_status = raw.get("test_result") or info.get("test_result")
        if explicit_status in {item.value for item in TestStatus}:
            status = TestStatus(explicit_status)
        elif history:
            status = history[-1].status
        else:
            # submitted 只表示 Agent 结束，绝不把它当成测试通过。
            status = TestStatus.FAIL
        metadata = {
            "exit_status": info.get("exit_status", raw.get("exit_status", "")),
            "mini_version": info.get("mini_version", ""),
            "model": info.get("model", raw.get("model", "")),
            "stop_reason": raw.get("stop_reason", info.get("stop_reason", "")),
        }
        if dataset_digest:
            metadata["dataset_digest"] = dataset_digest
        return RunTrace(
            run_id=run_id,
            agent_name=self.name,
            strategy=raw.get("strategy"),
            problem_id=problem_id,
            steps=steps,
            final_patch=final_patch,
            test_result=status,
            experiment_id=experiment_id,
            attempt_index=int(repeat) + attempt_offset,
            retry_count=int(raw.get("retry_count", info.get("retry_count", 0)) or 0),
            reported_total_tokens=_reported_tokens(raw, info),
            test_history=history,
            test_output=history[-1].output if history else "",
            changed_files=_patch_files(final_patch),
            started_at=raw.get("started_at") or info.get("started_at"),
            source_uri=source.as_posix(),
            metadata=metadata,
        )

    def _message_steps(self, messages: Any) -> list[Step]:
        if not isinstance(messages, list):
            return []
        steps: list[Step] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = _content_text(message.get("content", ""))
            commands = _commands(message, content)
            observation = _next_observation(messages, index)
            if not commands:
                if content.strip():
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            action="think",
                            observation=observation,
                            thought=content,
                            tokens=_tokens(message),
                            latency_ms=_latency(message),
                        )
                    )
                continue
            for command_name, command in commands:
                action = _action(command_name, command)
                args: dict[str, Any] = {"command": command} if command else {}
                if command_name:
                    args["tool"] = command_name
                path = _command_path(command)
                if path and action in {"read", "grep", "edit"}:
                    args["path"] = path
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        action=action,
                        args=args,
                        observation=observation,
                        thought=_COMMAND_RE.sub("", content).strip(),
                        tokens=_tokens(message),
                        latency_ms=_latency(message),
                        ok=not _command_failed(observation),
                        error=observation[-500:] if _command_failed(observation) else None,
                    )
                )
        return steps


def _structured_step(item: Any, index: int) -> Step:
    if not isinstance(item, dict):
        raise ValueError("steps 中的每一步必须是对象")
    return Step(
        step_id=int(item.get("step_id", item.get("step", index))),
        action=str(item.get("action", "think")),
        args=item.get("args") or {},
        observation=str(item.get("observation", "")),
        tokens=int(item.get("tokens", 0) or 0),
        latency_ms=int(item.get("latency_ms", 0) or 0),
        thought=str(item.get("thought", "")),
        ok=bool(item.get("ok", True)),
        error=item.get("error"),
    )


def _problem_id(raw: dict[str, Any], info: dict[str, Any]) -> str | None:
    candidates = [raw.get("problem_id"), info.get("problem_id"), info.get("instance_id")]
    instance = info.get("instance")
    if isinstance(instance, dict):
        candidates.extend(instance.get(key) for key in ("problem_id", "instance_id", "id"))
    return next((str(value) for value in candidates if value), None)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return str(content or "")


def _commands(message: dict[str, Any], content: str) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = str(function.get("name", ""))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"command": arguments}
        if isinstance(arguments, dict):
            command = str(arguments.get("command", arguments.get("cmd", "")))
            if not command:
                command = json.dumps(arguments, ensure_ascii=False)
        else:
            command = str(arguments)
        commands.append((name, command))
    extra = message.get("extra")
    if isinstance(extra, dict):
        for action in extra.get("actions") or []:
            if isinstance(action, dict):
                commands.append(
                    (
                        str(action.get("name", action.get("action", ""))),
                        str(action.get("command", action.get("cmd", ""))),
                    )
                )
    if not commands:
        commands.extend(("bash", match.strip()) for match in _COMMAND_RE.findall(content))
    return commands


def _next_observation(messages: list[Any], assistant_index: int) -> str:
    for message in messages[assistant_index + 1 :]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            break
        if role in {"user", "tool", "observation"}:
            return _content_text(message.get("content", message.get("observation", "")))
    return ""


def _action(tool: str, command: str) -> str:
    text = f"{tool} {command}".casefold()
    if any(token in text for token in ("pytest", "unittest", "run_test", "tox ", "nox ")):
        return "run_test"
    if any(token in text for token in ("apply_patch", "edit", "git apply", "sed -i")):
        return "edit"
    if "grep" in text or "rg " in text or "findstr" in text:
        return "grep"
    if re.search(r"(?:^|\s)(?:cat|head|tail|less|view|read)(?:\s|$)", text):
        return "read"
    return "think" if not command else "bash"


def _command_path(command: str) -> str | None:
    match = re.search(
        r"(?:cat|head|tail|grep|rg|sed)\s+(?:-[A-Za-z0-9]+\s+)*(?:\"[^\"]+\"\s+)?([^\s|;]+)",
        command,
    )
    return match.group(1).strip("'\"") if match else None


def _reported_tokens(raw: dict[str, Any], info: dict[str, Any]) -> int | None:
    stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}
    direct = raw.get("total_tokens", info.get("total_tokens", stats.get("total_tokens")))
    if direct is not None:
        return int(direct)
    input_tokens = stats.get("input_tokens")
    output_tokens = stats.get("output_tokens")
    if input_tokens is not None or output_tokens is not None:
        return int(input_tokens or 0) + int(output_tokens or 0)
    return None


def _tokens(message: dict[str, Any]) -> int:
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    total = message.get("tokens", usage.get("total_tokens", 0)) or 0
    if not total:
        total = int(usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("completion_tokens", 0) or 0
        )
    return int(total)


def _latency(message: dict[str, Any]) -> int:
    return int(message.get("latency_ms", message.get("duration_ms", 0)) or 0)


def _parse_test_output(output: str, attempt: int) -> TestOutcome | None:
    if not output or not any(pattern.search(output) for pattern in _COUNT_RES.values()):
        return None
    counts = {
        name: int(match.group(1)) if (match := pattern.search(output)) else 0
        for name, pattern in _COUNT_RES.items()
    }
    timed_out = bool(re.search(r"timed out|timeout", output, re.IGNORECASE))
    passed = counts["passed"] > 0 and counts["failed"] == 0 and counts["errors"] == 0
    return TestOutcome(
        attempt=attempt,
        status=TestStatus.TIMEOUT if timed_out else (
            TestStatus.PASS if passed else TestStatus.FAIL
        ),
        output=output[-12000:],
        returncode=0 if passed else -1,
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["errors"],
        failed_tests=re.findall(r"^FAILED\s+(\S+)", output, re.MULTILINE),
    )


def _is_diff(value: str) -> bool:
    return bool(value and re.search(r"^---\s+.*\n\+\+\+\s+", value, re.MULTILINE))


def _last_observed_diff(steps: list[Step]) -> str:
    for step in reversed(steps):
        if _is_diff(step.observation):
            start = step.observation.find("diff --git ")
            if start < 0:
                start = step.observation.find("--- ")
            return step.observation[start:]
    return ""


def _command_failed(observation: str) -> bool:
    lowered = observation.casefold()
    return any(
        marker in lowered
        for marker in ("command not found", "traceback (most recent call last)", "patch failed")
    )
