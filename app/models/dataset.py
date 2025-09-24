from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TableRowPayload(BaseModel):
    id_row: int = Field(alias="idRow")
    data: List[str]

    model_config = ConfigDict(populate_by_name=True)


class SemanticAnnotationsPayload(BaseModel):
    cea: List[Dict[str, object]] = Field(default_factory=list)
    cta: List[Dict[str, object]] = Field(default_factory=list)
    cpa: List[Dict[str, object]] = Field(default_factory=list)


class DatasetPayload(BaseModel):
    dataset_name: str = Field(alias="datasetName")
    table_name: str = Field(alias="tableName")
    header: List[str]
    rows: List[TableRowPayload]
    semantic_annotations: Optional[SemanticAnnotationsPayload] = Field(
        default=None, alias="semanticAnnotations"
    )
    metadata: Dict[str, object] = Field(default_factory=dict)
    kg_reference: str = Field(default="wikidata", alias="kgReference")

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
    rows: List[TableRowPayload]
    semantic_annotations: Optional[SemanticAnnotationsPayload] = None
    metadata: Dict[str, object]
    kg_reference: str
    created_at: datetime
    updated_at: datetime


class DatasetResponse(BaseModel):
    datasetId: str
    tableId: str
    datasetName: str
    tableName: str
    header: List[str]
    rowCount: int
    createdAt: datetime
    updatedAt: datetime
