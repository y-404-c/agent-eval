from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from agent_eval.attribution.rules import RuleBasedAttributor
from agent_eval.domain import Attribution, ErrorCategory, Problem, RunTrace

_ALLOWED_CATEGORIES = tuple(category.value for category in ErrorCategory)


class LLMAssistedAttributor:
    """仅对低置信规则结果请求 OpenAI 兼容接口；任何失败都回退规则结果。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        timeout_s: float = 20.0,
        review_threshold: float = 0.65,
        rules: RuleBasedAttributor | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("base_url 和 model 不能为空")
        if not 0 <= review_threshold <= 1:
            raise ValueError("review_threshold 必须位于 [0, 1]")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.review_threshold = review_threshold
        self.rules = rules or RuleBasedAttributor()
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns_client = client is None

    def attribute(self, trace: RunTrace, problem: Problem) -> Attribution | None:
        baseline = self.rules.attribute(trace, problem)
        if baseline is None or not (
            baseline.needs_review or baseline.confidence < self.review_threshold
        ):
            return baseline

        try:
            suggestion = self._request_suggestion(trace, problem, baseline)
        except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError):
            return baseline
        if suggestion is None:
            return baseline

        category, llm_rationale = suggestion
        rationale = "LLM 辅助归类建议（必须人工抽检）"
        if llm_rationale:
            rationale += f"：{llm_rationale}"
        else:
            rationale += f"：{category.value}。"
        return baseline.model_copy(
            update={
                "category": category,
                "method": "llm_assisted",
                "rationale": rationale,
                "evidence": [
                    *baseline.evidence,
                    "llm_assisted：类别来自 OpenAI 兼容接口的受限 JSON 响应；"
                    "不修改客观测试结果，且 needs_review=true。",
                ],
                "needs_review": True,
                "created_at": datetime.now(timezone.utc),
            }
        )

    def _request_suggestion(
        self,
        trace: RunTrace,
        problem: Problem,
        baseline: Attribution,
    ) -> tuple[ErrorCategory, str] | None:
        system_prompt = (
            "你是失败归因辅助器，不参与测试评分。只能输出一个 JSON 对象，格式为 "
            '{"category":"八类之一","rationale":"简短依据"}。category 只能是：'
            + ", ".join(_ALLOWED_CATEGORIES)
            + "。不得输出其他类别，不得编造未提供的 trace 证据。"
        )
        payload = {
            "trace": trace.model_dump(
                mode="json",
                include={
                    "run_id",
                    "problem_id",
                    "steps",
                    "final_patch",
                    "test_result",
                    "test_history",
                    "test_output",
                    "changed_files",
                    "metadata",
                },
            ),
            "problem": problem.model_dump(
                mode="json", include={"problem_id", "entry_path", "gold_files"}
            ),
            "low_confidence_rule_result": baseline.model_dump(
                mode="json", include={"category", "confidence", "rationale", "evidence"}
            ),
        }
        user_prompt = json.dumps(payload, ensure_ascii=False)[:30000]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None
        parsed = self._parse_json_object(content)
        if parsed is None:
            return None
        category = ErrorCategory(parsed["category"])
        llm_rationale = parsed.get("rationale", "")
        if not isinstance(llm_rationale, str):
            return None
        return category, llm_rationale.strip()[:1000]

    def _parse_json_object(self, content: str) -> dict[str, Any] | None:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("category"), str):
            return None
        return parsed

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMAssistedAttributor:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
