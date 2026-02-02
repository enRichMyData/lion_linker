from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class InputMode(str, Enum):
    inline = "inline"
    uri = "uri"
    upload_id = "upload_id"
    multipart = "multipart"


class InputFormat(str, Enum):
    csv = "text/csv"
    json = "application/json"


class RowRange(BaseModel):
    start: int = 0
    limit: int
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_range(self) -> "RowRange":
        if self.start < 0 or self.limit < 1:
            raise ValueError("row_range.start must be >= 0 and limit must be >= 1")
        return self


class InlineRow(BaseModel):
    row_id: Optional[Union[str, int]] = Field(
        default=None, validation_alias=AliasChoices("row_id", "rowId", "idRow")
    )
    cells: List[Any]
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class InlineTable(BaseModel):
    header: List[str]
    rows: List[InlineRow]
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_rows(self) -> "InlineTable":
        width = len(self.header)
        for row in self.rows:
            if len(row.cells) != width:
                raise ValueError("Row cells length does not match header length")
        return self


class JobInput(BaseModel):
    mode: InputMode
    format: InputFormat
    table: Optional[InlineTable] = None
    uri: Optional[str] = None
    upload_id: Optional[str] = None
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_input(self) -> "JobInput":
        if self.mode == InputMode.inline:
            if self.table is None:
                raise ValueError("inline input requires table")
        elif self.mode == InputMode.uri:
            if not self.uri:
                raise ValueError("uri input requires uri")
        elif self.mode == InputMode.upload_id:
            if not self.upload_id:
                raise ValueError("upload_id input requires upload_id")
        elif self.mode == InputMode.multipart:
            # multipart uploads are finalized after job creation
            pass
        return self


class JobCreateRequest(BaseModel):
    table_id: Optional[str] = None
    input: JobInput
    link_columns: Optional[List[Union[int, str]]] = None
    row_range: Optional[RowRange] = None
    top_k: Optional[int] = None
    execution: Optional[str] = "async"
    config: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="ignore")


class JobCreateResponse(BaseModel):
    ok: bool = True
    job_id: str
    table_id: Optional[str] = None
    status: str
    created_at: datetime
    limits: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="ignore")


class JobProgress(BaseModel):
    rows_total: Optional[int] = None
    rows_processed: Optional[int] = None
    cells_total: Optional[int] = None
    cells_processed: Optional[int] = None
    model_config = ConfigDict(extra="ignore")


class JobStatusResponse(BaseModel):
    ok: bool = True
    job_id: str
    table_id: Optional[str] = None
    status: str
    progress: Optional[JobProgress] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[Union[str, Dict[str, Any]]] = None
    model_config = ConfigDict(extra="ignore")


class FinalDecision(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    types: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None
    score: Optional[float] = None
    match: Optional[bool] = None
    model_config = ConfigDict(extra="ignore")


class CellResult(BaseModel):
    row: int
    col: int
    cell_id: str
    mention: str
    candidate_ranking: List[FinalDecision]
    explanation: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class ResultsPageResponse(BaseModel):
    ok: bool = True
    job_id: str
    cursor: Optional[str] = None
    next_cursor: Optional[str] = None
    results: List[CellResult]
    model_config = ConfigDict(extra="ignore")


class CapabilitiesData(BaseModel):
    input_modes: List[str]
    supported_formats: List[str]
    max_inline_bytes: int
    max_rows_inline: int
    max_top_k: int
    max_link_columns: int
    supports_sse: bool
    supports_exports: bool
    default_top_k: int
    default_timeout_seconds: int
    model_config = ConfigDict(extra="ignore")


class CapabilitiesResponse(BaseModel):
    ok: bool = True
    data: CapabilitiesData
    model_config = ConfigDict(extra="ignore")


class UploadCreateRequest(BaseModel):
    content_type: str
    content_length: int
    model_config = ConfigDict(extra="ignore")


class UploadCreateResponse(BaseModel):
    ok: bool = True
    upload_id: str
    upload_url: str
    expires_at: datetime
    model_config = ConfigDict(extra="ignore")


class UploadFinalizeRequest(BaseModel):
    total_parts: Optional[int] = None
    model_config = ConfigDict(extra="ignore")


class DownloadResponse(BaseModel):
    ok: bool = True
    job_id: str
    download_url: str
    expires_at: datetime
    model_config = ConfigDict(extra="ignore")


class ArtifactEntry(BaseModel):
    name: str
    type: str
    size_bytes: int
    model_config = ConfigDict(extra="ignore")


class ArtifactsResponse(BaseModel):
    ok: bool = True
    job_id: str
    artifacts: List[ArtifactEntry]
    model_config = ConfigDict(extra="ignore")
