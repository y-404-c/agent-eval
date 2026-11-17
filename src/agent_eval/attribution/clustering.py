from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_eval.domain import Attribution, ErrorCategory, RunTrace

_PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:/[^\s/]+)+|(?:[A-Za-z]:)?(?:\\[^\s\\]+)+")
_HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_error_text(text: str) -> str:
    normalized = _PATH_RE.sub(" <path> ", text.casefold())
    normalized = _HEX_RE.sub(" <hex> ", normalized)
    normalized = _NUMBER_RE.sub(" <num> ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _action_token(action: str) -> str:
    token = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", action.casefold()).strip("_")
    return token or "unknown"


class FailureClusterer:
    """对失败轨迹的动作序列与规范化报错做 TF-IDF + KMeans 聚类。"""

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        random_state: int = 42,
        top_n: int = 8,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters 必须至少为 1")
        if top_n < 1:
            raise ValueError("top_n 必须至少为 1")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.top_n = top_n

    def cluster(
        self,
        traces: Iterable[RunTrace],
        attributions: Iterable[Attribution] | None = None,
    ) -> list[dict[str, Any]]:
        failures = sorted(
            (trace for trace in traces if not trace.passed),
            key=lambda item: item.run_id,
        )
        if not failures:
            return []

        documents = [self._document(trace) for trace in failures]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        matrix = vectorizer.fit_transform(documents)
        terms = np.asarray(vectorizer.get_feature_names_out())
        cluster_count = min(self.n_clusters, len(failures))

        if len(failures) == 1:
            labels = np.zeros(1, dtype=int)
            centers = matrix.toarray()
        else:
            model = KMeans(
                n_clusters=cluster_count,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(matrix)
            centers = model.cluster_centers_

        category_by_run = {item.run_id: item.category for item in attributions or []}
        clusters: list[dict[str, Any]] = []
        for label in range(cluster_count):
            indices = np.flatnonzero(labels == label)
            if not len(indices):
                continue
            members = [failures[int(index)] for index in indices]
            run_ids = [trace.run_id for trace in members]
            scores = np.asarray(matrix[indices].mean(axis=0)).ravel()
            order = np.argsort(-scores, kind="stable")
            top_terms = [str(terms[index]) for index in order if scores[index] > 0][: self.top_n]
            center = centers[label]
            distances = np.linalg.norm(matrix[indices].toarray() - center, axis=1)
            representative_index = int(indices[int(np.argmin(distances))])
            clusters.append(
                {
                    "cluster_id": label,
                    "run_ids": run_ids,
                    "size": len(run_ids),
                    "top_terms": top_terms,
                    "category": self._dominant_category(members, category_by_run),
                    "representative_run_id": failures[representative_index].run_id,
                }
            )

        clusters.sort(key=lambda item: (-item["size"], item["run_ids"][0]))
        for cluster_id, item in enumerate(clusters):
            item["cluster_id"] = cluster_id
        return clusters

    def _document(self, trace: RunTrace) -> str:
        actions = " ".join(f"action_{_action_token(step.action)}" for step in trace.steps)
        error_parts = [trace.test_output]
        error_parts.extend(outcome.output for outcome in trace.test_history)
        error_parts.extend(
            part
            for step in trace.steps
            for part in (step.error, step.observation if not step.ok else "")
            if part
        )
        errors = _normalize_error_text("\n".join(part for part in error_parts if part))
        return f"{actions or 'action_none'} {errors or 'error_none'}"

    def _dominant_category(
        self,
        members: list[RunTrace],
        category_by_run: dict[str, ErrorCategory],
    ) -> str | None:
        categories = [category_by_run.get(trace.run_id, trace.error_category) for trace in members]
        present = [category for category in categories if category is not None]
        if not present:
            return None
        counts = Counter(present)
        dominant = min(counts, key=lambda category: (-counts[category], category.value))
        return dominant.value
