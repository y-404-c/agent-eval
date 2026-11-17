from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from agent_eval.domain import Problem


class LeakageChecker:
    """检测 few-shot 文本是否引用当前评测题；命中即拒绝继续实验。"""

    def check(self, problems: Iterable[Problem], fewshot_text: str) -> list[str]:
        hits: list[str] = []
        for problem in problems:
            identifiers = {
                problem.problem_id,
                Path(problem.entry_path).stem,
                Path(problem.entry_path).name,
            }
            for identifier in identifiers:
                if identifier and re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
                    fewshot_text,
                    re.IGNORECASE,
                ):
                    hits.append(problem.problem_id)
                    break
        return sorted(set(hits))

    def check_file(self, problems: Iterable[Problem], path: Path | str) -> list[str]:
        fewshot_path = Path(path)
        return self.check(problems, fewshot_path.read_text(encoding="utf-8"))

    def assert_no_leakage(self, problems: Iterable[Problem], fewshot_text: str) -> None:
        hits = self.check(problems, fewshot_text)
        if hits:
            raise ValueError("few-shot 与评测集泄露: " + ", ".join(hits))
