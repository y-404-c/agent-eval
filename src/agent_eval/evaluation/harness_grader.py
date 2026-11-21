"""SWE-bench 官方 harness 判定适配器。

SWE-bench 是多文件真实仓库，测试必须在对应镜像的 /testbed 里跑，
不能像 QuixBugs 那样把单文件源码塞进临时目录用本地 pytest 重评分。
本 grader 直接读取 SWE-bench harness 已经在容器里跑出的客观判定
（FAIL_TO_PASS / PASS_TO_PASS / resolved），转成项目二的 TestOutcome。
判定来源可追溯到 logs/run_evaluation/<run_id>/.../run_instance.log，
不使用 LLM、不使用 patch 相似度，符合 ObjectiveGrader 的契约。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_eval.domain import Problem, TestOutcome, TestStatus


class HarnessVerdictGrader:
    """用 SWE-bench 官方 harness 判定替代本地 pytest 重评分。"""

    source_label = "swebench-harness-v1"

    def __init__(self, verdict_path: Path | str) -> None:
        path = Path(verdict_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.verdicts: dict[str, dict[str, Any]] = payload
        if not self.verdicts:
            raise ValueError(f"SWE-bench 判定文件为空: {path}")

    def grade(self, problem: Problem, patch: str, *, attempt: int = 1) -> TestOutcome:
        verdict = self.verdicts.get(problem.problem_id)
        if verdict is None:
            raise KeyError(f"缺少 {problem.problem_id} 的官方 harness 判定")

        f2p = verdict["fail_to_pass"]
        p2p = verdict["pass_to_pass"]
        p2f = verdict.get("pass_to_fail", {"success": [], "failure": []})
        failed_tests = list(f2p.get("failure", [])) + list(p2f.get("failure", []))
        passed_count = len(f2p.get("success", [])) + len(p2p.get("success", []))
        failed_count = len(failed_tests)
        status = TestStatus.PASS if verdict["resolved"] else TestStatus.FAIL

        lines = [
            f"source: swebench-harness run_id={verdict.get('source_run_id', '')}",
            f"patch_successfully_applied={verdict.get('patch_successfully_applied')}",
            f"FAIL_TO_PASS success={len(f2p.get('success', []))} failure={len(f2p.get('failure', []))}",
            f"PASS_TO_PASS success={len(p2p.get('success', []))} failure={len(p2p.get('failure', []))}",
        ]
        for name in failed_tests[:20]:
            lines.append(f"FAILED {name}")
        if failed_count > 20:
            lines.append(f"... and {failed_count - 20} more failures")

        return TestOutcome(
            attempt=attempt,
            status=status,
            output="\n".join(lines),
            returncode=0 if status == TestStatus.PASS else 1,
            passed=passed_count,
            failed=failed_count,
            errors=0,
            failed_tests=failed_tests,
            duration_ms=0,
        )
