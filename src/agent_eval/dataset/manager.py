from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from agent_eval.domain import Problem

_DEFAULT_CONFTEST = '''import pytest


def pytest_addoption(parser):
    parser.addoption("--correct", action="store_true", default=False)
    parser.addoption("--runslow", action="store_true", default=False)


def pytest_configure(config):
    pytest.use_correct = bool(config.getoption("--correct"))
    pytest.run_slow = bool(config.getoption("--runslow"))
'''


@dataclass(frozen=True)
class VersionedDataset:
    path: Path
    problems: dict[str, Problem]
    digest: str

    def get(self, problem_id: str) -> Problem:
        try:
            return self.problems[problem_id]
        except KeyError as exc:
            raise KeyError(f"评测集中不存在题目: {problem_id}") from exc

    def __iter__(self) -> Iterator[Problem]:
        return iter(self.problems.values())

    def __len__(self) -> int:
        return len(self.problems)


class DatasetManager:
    """JSONL 评测集：源码与官方测试一起版本化，不保存自制标准 patch。"""

    def load(self, path: Path | str) -> VersionedDataset:
        dataset_path = Path(path)
        raw = dataset_path.read_bytes()
        problems: dict[str, Problem] = {}
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            problem = Problem.model_validate(payload)
            if problem.problem_id in problems:
                raise ValueError(f"重复 problem_id: {problem.problem_id} (line {line_number})")
            problems[problem.problem_id] = problem
        if not problems:
            raise ValueError(f"评测集为空: {dataset_path}")
        return VersionedDataset(
            path=dataset_path,
            problems=dict(sorted(problems.items())),
            digest=hashlib.sha256(raw).hexdigest(),
        )

    def save(self, problems: Iterable[Problem], path: Path | str) -> VersionedDataset:
        dataset_path = Path(path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(problems, key=lambda item: item.problem_id)
        if not ordered:
            raise ValueError("不能保存空评测集")
        content = "".join(
            problem.model_dump_json() + "\n" for problem in ordered
        )
        temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, dataset_path)
        return self.load(dataset_path)

    def import_quixbugs(
        self,
        root: Path | str,
        output: Path | str,
        *,
        version: str,
        source: str = "https://github.com/jkoppel/QuixBugs",
    ) -> VersionedDataset:
        root_path = Path(root)
        programs = root_path / "python_programs"
        tests = root_path / "python_testcases"
        json_tests = root_path / "json_testcases"
        if not programs.is_dir() or not tests.is_dir():
            raise FileNotFoundError("QuixBugs 根目录必须包含 python_programs 与 python_testcases")

        support: dict[str, str] = {"conftest.py": _DEFAULT_CONFTEST}
        official_conftest = root_path / "conftest.py"
        if official_conftest.exists():
            support["conftest.py"] = official_conftest.read_text(encoding="utf-8")
        for name in ("load_testdata.py", "node.py"):
            path = tests / name
            if path.exists():
                support[f"python_testcases/{name}"] = path.read_text(encoding="utf-8")

        today = date.today()
        problems: list[Problem] = []
        for test_path in sorted(tests.glob("test_*.py")):
            problem_id = test_path.stem.removeprefix("test_")
            source_path = programs / f"{problem_id}.py"
            if not source_path.exists():
                continue
            support_files = dict(support)
            json_path = json_tests / f"{problem_id}.json"
            if json_path.exists():
                support_files[f"json_testcases/{problem_id}.json"] = json_path.read_text(
                    encoding="utf-8"
                )
            problems.append(
                Problem(
                    problem_id=problem_id,
                    source=source,
                    version=version,
                    added_at=today,
                    entry_path=f"python_programs/{problem_id}.py",
                    test_path=f"python_testcases/test_{problem_id}.py",
                    source_code=source_path.read_text(encoding="utf-8"),
                    test_code=test_path.read_text(encoding="utf-8"),
                    support_files=support_files,
                    gold_files=[f"python_programs/{problem_id}.py"],
                )
            )
        if not problems:
            raise ValueError("未找到可配对的 QuixBugs 源码与官方测试")
        return self.save(problems, output)
