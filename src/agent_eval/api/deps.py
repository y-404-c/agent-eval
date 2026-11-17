from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from agent_eval.config import Settings
from agent_eval.dataset import DatasetManager, VersionedDataset
from agent_eval.storage import EvaluationRepository


@dataclass
class Container:
    settings: Settings
    repository: EvaluationRepository
    dataset_path: Path

    def dataset(self) -> VersionedDataset:
        return DatasetManager().load(self.dataset_path)


def get_container(request: Request) -> Container:
    return request.app.state.container
