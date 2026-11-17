from __future__ import annotations

import json
import sqlite3

from agent_eval.adapters import MiniSweAgentAdapter, OursAdapter, get_adapter
from agent_eval.domain import TestStatus


def test_ours_adapter_reads_project_one_sqlite(tmp_path) -> None:
    raw = {
        "run_id": "ours-1",
        "strategy": "S4",
        "problem_id": "sample",
        "repeat_index": 2,
        "model": "test-model",
        "passed": True,
        "error_category": "none",
        "started_at": "2026-01-01T00:00:00+00:00",
        "meta": {"final_stop_reason": "tests_passed"},
        "attempts": [
            {
                "attempt": 1,
                "steps": [
                    {
                        "step": 1,
                        "thought": "inspect",
                        "action": "read_file",
                        "args": {"path": "python_programs/sample.py"},
                        "observation": "source",
                        "tokens": 10,
                        "latency_ms": 20,
                    }
                ],
                "patch_text": "old-patch",
                "changed_files": ["python_programs/sample.py"],
                "test_report": {
                    "returncode": 1,
                    "timed_out": False,
                    "passed": 0,
                    "failed": 1,
                    "errors": 0,
                    "failed_tests": ["test_add"],
                    "output_tail": "AssertionError",
                },
                "passed": False,
            },
            {
                "attempt": 2,
                "steps": [
                    {
                        "step": 1,
                        "action": "run_test",
                        "args": {},
                        "observation": "1 passed",
                        "tokens": 5,
                        "latency_ms": 30,
                    }
                ],
                "patch_text": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
                "changed_files": ["python_programs/sample.py"],
                "test_report": {
                    "returncode": 0,
                    "timed_out": False,
                    "passed": 1,
                    "failed": 0,
                    "errors": 0,
                    "output_tail": "1 passed",
                },
                "passed": True,
            },
        ],
    }
    path = tmp_path / "ours.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE runs (run_id TEXT, started_at TEXT, trace_json TEXT)")
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            ("ours-1", raw["started_at"], json.dumps(raw)),
        )

    trace = list(OursAdapter().load(path, experiment_id="exp", dataset_digest="abc"))[0]

    assert trace.agent_name == "ours"
    assert trace.experiment_id == "exp"
    assert trace.attempt_index == 2
    assert trace.retry_count == 1
    assert trace.test_result == TestStatus.PASS
    assert [step.step_id for step in trace.steps] == [1, 2]
    assert trace.total_tokens == 15
    assert len(trace.test_history) == 2
    assert trace.final_patch.startswith("--- a/x")
    assert trace.metadata["dataset_digest"] == "abc"
    assert trace.metadata["stop_reason"] == "tests_passed"


def test_mini_swe_adapter_maps_messages_and_never_treats_submit_as_pass(tmp_path) -> None:
    diff = (
        "--- a/python_programs/sample.py\n+++ b/python_programs/sample.py\n"
        "@@ -1,2 +1,2 @@\n def add(a, b):\n"
        "-    return a - b\n+    return a + b\n"
    )
    raw = {
        "info": {
            "instance_id": "sample",
            "submission": diff,
            "exit_status": "submitted",
            "mini_version": "test",
            "model_stats": {"input_tokens": 100, "output_tokens": 20},
        },
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "Inspect\n```bash\ncat python_programs/sample.py\n```",
                "usage": {"total_tokens": 12},
                "latency_ms": 50,
            },
            {"role": "user", "content": "def add(a, b): ..."},
            {"role": "assistant", "content": "```bash\npytest -q\n```"},
            {"role": "user", "content": "1 failed, 0 passed\nFAILED test_sample.py::test_add"},
            {"role": "assistant", "content": "```bash\ngit diff\n```"},
            {"role": "user", "content": diff},
        ],
    }
    path = tmp_path / "sample_rep-1.traj.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    trace = list(MiniSweAgentAdapter().load(path))[0]

    assert trace.problem_id == "sample"
    assert trace.agent_name == "mini-swe-agent"
    assert trace.attempt_index == 1
    assert trace.test_result == TestStatus.FAIL
    assert trace.final_patch == diff
    assert trace.changed_files == ["python_programs/sample.py"]
    assert [step.action for step in trace.steps[:2]] == ["read", "run_test"]
    assert trace.steps[0].tokens == 12
    assert trace.total_tokens == 120
    assert len(trace.test_history) == 1
    assert trace.test_history[0].failed_tests == ["test_sample.py::test_add"]


def test_adapter_registry_supports_both_agents() -> None:
    assert get_adapter("ours").name == "ours"
    assert get_adapter("mini-swe-agent").name == "mini-swe-agent"
