from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TestStatus(str, Enum):
    __test__ = False  # 防止 pytest 将领域枚举误认为测试类

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"


class ErrorCategory(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    WRONG_LOCATION = "wrong_location"
    INCOMPLETE_FIX = "incomplete_fix"
    REGRESSION = "regression"
    ASSERTION_MISMATCH = "assertion_mismatch"
    TOOL_FAILURE = "tool_failure"
    LOOP_EXHAUSTED = "loop_exhausted"
    ENVIRONMENT = "environment"


ERROR_TAXONOMY: dict[ErrorCategory, str] = {
    ErrorCategory.SYNTAX_ERROR: "生成代码语法错误",
    ErrorCategory.WRONG_LOCATION: "改错文件/函数，未定位到真实 bug",
    ErrorCategory.INCOMPLETE_FIX: "改了正确位置但逻辑不完整",
    ErrorCategory.REGRESSION: "修好了目标但引入了新失败",
    ErrorCategory.ASSERTION_MISMATCH: "断言失败但修复方向对",
    ErrorCategory.TOOL_FAILURE: "工具调用失败/参数错误导致",
    ErrorCategory.LOOP_EXHAUSTED: "重复无效动作直到步数耗尽",
    ErrorCategory.ENVIRONMENT: "超时/OOM/依赖缺失（不可纠）",
}


class Step(BaseModel):
    step_id: int = Field(ge=1)
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    thought: str = ""
    ok: bool = True
    error: str | None = None


class TestOutcome(BaseModel):
    __test__ = False

    attempt: int = Field(default=1, ge=1)
    status: TestStatus
    output: str = ""
    returncode: int = 0
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    failed_tests: list[str] = Field(default_factory=list)
    duration_ms: int = Field(default=0, ge=0)


class RunTrace(BaseModel):
    """一次独立生成的统一轨迹；不同 Agent 必须先映射到该模型再评测。"""

    run_id: str
    agent_name: str
    strategy: str | None = None
    problem_id: str
    steps: list[Step] = Field(default_factory=list)
    final_patch: str = ""
    test_result: TestStatus
    error_category: ErrorCategory | None = None
    experiment_id: str = "default"
    attempt_index: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    test_history: list[TestOutcome] = Field(default_factory=list)
    test_output: str = ""
    changed_files: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    source_uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "agent_name", "problem_id", "experiment_id")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不能为空")
        return value.strip()

    @property
    def passed(self) -> bool:
        return self.test_result == TestStatus.PASS

    @property
    def total_tokens(self) -> int:
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        return sum(step.tokens for step in self.steps)

    @property
    def total_latency_ms(self) -> int:
        return sum(step.latency_ms for step in self.steps)


class Problem(BaseModel):
    problem_id: str
    source: str
    version: str
    added_at: date
    entry_path: str
    test_path: str
    source_code: str
    test_code: str
    support_files: dict[str, str] = Field(default_factory=dict)
    gold_files: list[str] = Field(default_factory=list)

    @field_validator("entry_path", "test_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("评测文件必须使用安全相对路径")
        return normalized

    def model_post_init(self, __context: Any) -> None:
        if not self.gold_files:
            self.gold_files = [self.entry_path]


class Attribution(BaseModel):
    run_id: str
    category: ErrorCategory
    method: Literal["rule", "llm_assisted", "manual"]
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    needs_review: bool = False
    created_at: datetime


class AuditRecord(BaseModel):
    run_id: str
    reviewer: str
    manual_category: ErrorCategory
    predicted_category: ErrorCategory
    agrees: bool
    note: str = ""
    created_at: datetime


class WatchlistEntry(BaseModel):
    problem_id: str
    failure_count: int = Field(ge=1)
    categories: list[ErrorCategory]
    run_ids: list[str]
    reason: str
    updated_at: datetime
