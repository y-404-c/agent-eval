from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVAL_", env_file=".env", extra="ignore"
    )

    db_path: Path = Path("var/agent_eval.db")
    reports_dir: Path = Path("reports")
    test_timeout_s: float = Field(default=10.0, gt=0)
    max_output_chars: int = Field(default=12000, gt=0)
    python_executable: str = "python"
    watchlist_min_failures: int = Field(default=2, ge=1)

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
