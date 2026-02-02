from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class JobStatus(str, Enum):
    uploading = "uploading"
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class TableRow(BaseModel):
    row_id: int = Field(validation_alias=AliasChoices("row_id", "rowId", "idRow"))
    data: List[str]
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class JobTable(BaseModel):
    header: List[str]
    rows: List[TableRow]
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_rows(self) -> "JobTable":
        width = len(self.header)
        row_ids = set()
        for row in self.rows:
            if len(row.data) != width:
                raise ValueError(
                    "Row data length does not match header length for row "
                    f"{row.row_id}"
                )
            if row.row_id in row_ids:
                raise ValueError(f"Duplicate row_id detected: {row.row_id}")
            row_ids.add(row.row_id)
        return self


class JobSelection(BaseModel):
    columns: List[Union[int, str]]
    model_config = ConfigDict(extra="ignore")


class JobCreateRequest(BaseModel):
    task: str = "CEA"
    table: JobTable
    selection: Optional[JobSelection] = None
    config: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="ignore")


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    task: str
    table: Optional[JobTable] = None
    table_id: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    link_columns: Optional[List[Union[int, str]]] = None
    row_range: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    rows_total: Optional[int] = None
    rows_processed: Optional[int] = None
    cells_total: Optional[int] = None
    cells_processed: Optional[int] = None
    selection: Optional[JobSelection] = None
    config: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[Union[str, Dict[str, Any]]] = None
    model_config = ConfigDict(extra="ignore")


class JobEnqueueResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[Union[str, Dict[str, Any]]] = None
    model_config = ConfigDict(extra="ignore")


class PredictionEntity(BaseModel):
    id: str
    source: str
    label: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class PredictionRecord(BaseModel):
    job_id: str
    row_id: int
    col_id: int
    mention: str
    entity: PredictionEntity
    score: Optional[float] = None
    status: Optional[str] = None
    seq: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="ignore")


class PredictionResponse(BaseModel):
    row_id: int
    col_id: int
    mention: str
    entity: PredictionEntity
    score: Optional[float] = None
    model_config = ConfigDict(extra="ignore")


class PredictionPageResponse(BaseModel):
    job_id: str
    offset: int
    limit: int
    total: int
    predictions: List[PredictionResponse]
    model_config = ConfigDict(extra="ignore")


class PromptRecord(BaseModel):
    job_id: str
    seq: int
    prompt: str
    response: Optional[str] = None
    error: Optional[str] = None
    task_ids: Optional[List[str]] = None
    batch_index: Optional[int] = None
    task_count: Optional[int] = None
    prompt_type: Optional[str] = Field(default=None, alias="type")
    created_at: datetime
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class PromptResponse(BaseModel):
    seq: int
    prompt: str
    response: Optional[str] = None
    error: Optional[str] = None
    task_ids: Optional[List[str]] = None
    batch_index: Optional[int] = None
    task_count: Optional[int] = None
    prompt_type: Optional[str] = Field(default=None, alias="type")
    created_at: datetime
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class PromptPageResponse(BaseModel):
    job_id: str
    offset: int
    limit: int
    total: int
    prompts: List[PromptResponse]
    model_config = ConfigDict(extra="ignore")


class CapabilitiesResponse(BaseModel):
    tasks: List[str]
    providers: List[str]
    retrievers: List[str]
    max_recommended_rows: int
    model_config = ConfigDict(extra="ignore")
