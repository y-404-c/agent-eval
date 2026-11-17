from __future__ import annotations

from conftest import unified_patch

from agent_eval.dataset import DatasetManager, LeakageChecker
from agent_eval.domain import TestStatus
from agent_eval.evaluation import ObjectiveGrader, PatchError, apply_unified_diff


def test_dataset_import_is_versioned_and_excludes_correct_patch(tmp_path) -> None:
    root = tmp_path / "QuixBugs"
    programs = root / "python_programs"
    tests = root / "python_testcases"
    json_tests = root / "json_testcases"
    programs.mkdir(parents=True)
    tests.mkdir()
    json_tests.mkdir()
    (programs / "sample.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tests / "test_sample.py").write_text("def test_add():\n    assert True\n", encoding="utf-8")
    (tests / "load_testdata.py").write_text("loader = True\n", encoding="utf-8")
    (json_tests / "sample.json").write_text("[]\n", encoding="utf-8")
    (root / "conftest.py").write_text("official = True\n", encoding="utf-8")
    output = tmp_path / "quixbugs.v1.jsonl"

    dataset = DatasetManager().import_quixbugs(root, output, version="v1")

    assert len(dataset) == 1
    assert dataset.get("sample").test_code == "def test_add():\n    assert True\n"
    assert "correct_code" not in output.read_text(encoding="utf-8")
    assert dataset.get("sample").support_files["conftest.py"] == "official = True\n"
    assert dataset.digest == DatasetManager().load(output).digest


def test_leakage_checker_detects_problem_id_and_filename(problem) -> None:
    checker = LeakageChecker()

    assert checker.check([problem], "example: python_programs/sample.py") == ["sample"]
    assert checker.check([problem], "an unrelated sorting example") == []


def test_objective_grader_runs_official_test(problem) -> None:
    grader = ObjectiveGrader(timeout_s=10)
    patch = unified_patch(problem, "def add(a, b):\n    return a + b\n")

    passed = grader.grade(problem, patch)
    failed = grader.grade(problem, "")

    assert passed.status == TestStatus.PASS
    assert passed.passed == 1
    assert failed.status == TestStatus.FAIL
    assert failed.failed == 1
    assert "test_add" in failed.failed_tests[0]


def test_grader_rejects_test_tampering(problem) -> None:
    patch = (
        "--- a/python_testcases/test_sample.py\n"
        "+++ b/python_testcases/test_sample.py\n"
        "@@ -1,3 +1 @@\n"
        "-from python_programs.sample import add\n"
        "-\n"
        "-def test_add():\n"
        "+def test_add():\n"
    )
    outcome = ObjectiveGrader().grade(problem, patch)

    assert outcome.status == TestStatus.FAIL
    assert "禁止修改官方测试" in outcome.output


def test_patch_rejects_path_traversal(problem) -> None:
    patch = "--- a/../outside.py\n+++ b/../outside.py\n@@ -0,0 +1 @@\n+value = 1\n"
    try:
        apply_unified_diff(patch, {})
    except PatchError as exc:
        assert "不安全" in str(exc)
    else:
        raise AssertionError("目录穿越 patch 应被拒绝")
