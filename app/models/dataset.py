from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.jobs import PredictionSummary


class TableAnnotationRequest(BaseModel):
    row_ids: Optional[List[int]] = Field(default=None, alias="rowIds")
    lion_config: Optional[Dict[str, Any]] = Field(default=None, alias="lionConfig")
    retriever_config: Optional[Dict[str, Any]] = Field(default=None, alias="retrieverConfig")
    token: Optional[str] = None


class TableRowPayload(BaseModel):
    id_row: int = Field(alias="idRow")
    data: List[str]
    model_config = ConfigDict(populate_by_name=True)


class TableRowRecord(TableRowPayload):
    annotations: List[PredictionSummary] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True)


class DatasetPayload(BaseModel):
    dataset_name: str = Field(alias="datasetName")
    table_name: str = Field(alias="tableName")
    header: List[str]
    rows: List[TableRowPayload]
    metadata: Dict[str, object] = Field(default_factory=dict)
    lion_config: Optional[Dict[str, Any]] = Field(default=None, alias="lionConfig")
    retriever_config: Optional[Dict[str, Any]] = Field(default=None, alias="retrieverConfig")

    @model_validator(mode="after")
    def validate_rows(self) -> "DatasetPayload":
        width = len(self.header)
        for row in self.rows:
            if len(row.data) != width:
                raise ValueError(
                    "Row data length does not match header length for row " f"{row.id_row}"
                )
        return self


class DatasetMeta(BaseModel):
    dataset_id: str
    dataset_name: str
    created_at: datetime
    updated_at: datetime


class DatasetTableRecord(BaseModel):
    dataset_id: str
    dataset_name: str
    table_id: str
    table_name: str
    header: List[str]
    rows: List[TableRowRecord]
    metadata: Dict[str, object]
    created_at: datetime
    updated_at: datetime
    lion_config: Optional[Dict[str, Any]] = None
    retriever_config: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    datasetId: str
    tableId: str
    datasetName: str
    tableName: str
    header: List[str]
    rowCount: int
    createdAt: datetime
    updatedAt: datetime
