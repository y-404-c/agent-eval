from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from agent_eval.domain import RunTrace


class TraceAdapter(ABC):
    """外部轨迹接入协议：适配器只负责映射，不负责打分。"""

    name: str

    @abstractmethod
    def load(self, source: Path | str, **kwargs) -> Iterable[RunTrace]:
        """从一个数据库、文件或目录读取零条或多条统一轨迹。"""
