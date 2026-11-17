from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from agent_eval.domain import Problem, TestOutcome, TestStatus
from agent_eval.evaluation.patch import PatchError, apply_unified_diff, changed_files

_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+) passed\b"),
    "failed": re.compile(r"(\d+) failed\b"),
    "errors": re.compile(r"(\d+) errors?\b"),
}


class ObjectiveGrader:
    """在临时目录应用 patch 并运行官方 pytest；不使用 LLM 或 patch 相似度评分。"""

    def __init__(
        self,
        *,
        python_executable: str = "python",
        timeout_s: float = 20.0,
        max_output_chars: int = 12000,
    ) -> None:
        self.python_executable = python_executable
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars

    def grade(self, problem: Problem, patch: str, *, attempt: int = 1) -> TestOutcome:
        started = time.perf_counter()
        files = {problem.entry_path: problem.source_code}
        protected = {problem.test_path, *problem.support_files, "conftest.py"}
        try:
            targets = set(changed_files(patch))
            illegal = sorted(targets & protected)
            if illegal:
                raise PatchError("禁止修改官方测试或测试辅助文件: " + ", ".join(illegal))
            patched = apply_unified_diff(patch, files) if patch.strip() else files
        except PatchError as exc:
            return TestOutcome(
                attempt=attempt,
                status=TestStatus.FAIL,
                output=f"patch application failed: {exc}",
                returncode=-1,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        with tempfile.TemporaryDirectory(prefix="agent-eval-") as temporary:
            root = Path(temporary)
            for relative_path, content in patched.items():
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            for relative_path, content in problem.support_files.items():
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            test_target = root / problem.test_path
            test_target.parent.mkdir(parents=True, exist_ok=True)
            test_target.write_text(problem.test_code, encoding="utf-8")
            (root / "python_programs" / "__init__.py").write_text("", encoding="utf-8")
            command = [
                self.python_executable,
                "-m",
                "pytest",
                problem.test_path,
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_s,
                    check=False,
                )
                output = (completed.stdout or "") + (completed.stderr or "")
                return self._parse(
                    output,
                    attempt=attempt,
                    returncode=completed.returncode,
                    timed_out=False,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            except subprocess.TimeoutExpired as exc:
                output = _decode(exc.stdout) + _decode(exc.stderr) + "\nTIMEOUT"
                return self._parse(
                    output,
                    attempt=attempt,
                    returncode=-1,
                    timed_out=True,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            except OSError as exc:
                return TestOutcome(
                    attempt=attempt,
                    status=TestStatus.FAIL,
                    output=f"environment error: {exc}",
                    returncode=-1,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

    def _parse(
        self,
        output: str,
        *,
        attempt: int,
        returncode: int,
        timed_out: bool,
        duration_ms: int,
    ) -> TestOutcome:
        counts = {
            name: int(match.group(1)) if (match := pattern.search(output)) else 0
            for name, pattern in _COUNT_PATTERNS.items()
        }
        passed = (
            not timed_out
            and returncode == 0
            and counts["passed"] > 0
            and counts["failed"] == 0
            and counts["errors"] == 0
        )
        failed_tests = re.findall(r"^(?:FAILED|ERROR)\s+(\S+)", output, re.MULTILINE)
        return TestOutcome(
            attempt=attempt,
            status=TestStatus.TIMEOUT if timed_out else (
                TestStatus.PASS if passed else TestStatus.FAIL
            ),
            output=output[-self.max_output_chars :],
            returncode=returncode,
            passed=counts["passed"],
            failed=counts["failed"],
            errors=counts["errors"],
            failed_tests=list(dict.fromkeys(failed_tests))[:50],
            duration_ms=duration_ms,
        )


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
