from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    experiment_id: str
    agent_names: list[str] | None = None
    strategy: str | None = None
    k: int = Field(default=3, ge=1)


class AuditRequest(BaseModel):
    reviewer: str
    manual_category: str
    note: str = ""
