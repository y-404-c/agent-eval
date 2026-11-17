from __future__ import annotations

import difflib
from datetime import date

import pytest

from agent_eval.dataset import VersionedDataset
from agent_eval.domain import Problem


@pytest.fixture
def problem() -> Problem:
    return Problem(
        problem_id="sample",
        source="unit-fixture",
        version="v1",
        added_at=date(2026, 1, 1),
        entry_path="python_programs/sample.py",
        test_path="python_testcases/test_sample.py",
        source_code="def add(a, b):\n    return a - b\n",
        test_code=(
            "from python_programs.sample import add\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n"
        ),
        gold_files=["python_programs/sample.py"],
    )


@pytest.fixture
def dataset(problem: Problem, tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(problem.model_dump_json() + "\n", encoding="utf-8")
    return VersionedDataset(path=path, problems={"sample": problem}, digest="test-digest")


def unified_patch(problem: Problem, fixed_code: str) -> str:
    return "".join(
        difflib.unified_diff(
            problem.source_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile=f"a/{problem.entry_path}",
            tofile=f"b/{problem.entry_path}",
        )
    )
