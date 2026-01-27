# Entity Linking REST API

This document describes a standalone HTTP API for an LLM-based Entity Linking (EL) service.
The service accepts tabular data, links entity mentions per cell, and stores results in MongoDB.

## Authentication

All endpoints require an API key. Provide it via `X-API-Key: <key>` or
`Authorization: Bearer <key>`.

## Endpoint Overview

- `GET /health` — liveness check
- `GET /capabilities` — supported modes, limits, defaults
- `POST /jobs` — create a job (inline | uri | upload_id); always returns `job_id`
- `GET /jobs/{job_id}` — job status + progress
- `GET /jobs/{job_id}/results` — cursor-paged cell-level finals (no candidates)
- `GET /jobs/{job_id}/cells/{row}/{col}/candidates` — lazy candidates for one cell
- `POST /jobs/{job_id}:cancel` — cancel a job

Optional:
- `POST /uploads` — pre-signed upload URL + `upload_id`
- `GET /jobs/{job_id}/events` — SSE progress/events
- `GET /jobs/{job_id}/results/download` — signed URL export
- `GET /jobs/{job_id}/artifacts` — list stored artifacts

## Stable Identifiers

- `job_id`: opaque string, globally unique per run.
- `table_id`: optional opaque client-provided identifier (echoed back).
- `row`: zero-based row index in the submitted table.
- `col`: zero-based column index in the submitted table.
- `cell_id`: stable derived identifier (for example `row_id:col`), returned on results.

## Result Models

- Final prediction per cell:
  - `entity_id` (string; `NIL` allowed)
  - `label` (optional string)
  - `score` (float 0..1)
  - `status` (`linked` | `nil` | `error`)
- Candidates (per-cell endpoint):
  - `rank` (int)
  - `entity_id` (string)
  - `label` (optional string)
  - `score` (float 0..1)
  - `features` (optional map)

## Limits and Errors

Limits are exposed via `GET /capabilities` (for example `max_inline_bytes`, `max_top_k`).

Common errors:
- `400` invalid input
- `401` invalid API key
- `404` job not found
- `409` job not cancellable (already finished)
- `413` payload too large
- `422` invalid schema or format
- `429` rate limited
- `503` model/retriever unavailable

## Endpoints with Examples

### `GET /health`

Response:
```json
{
  "ok": true,
  "status": "healthy",
  "time": "2025-01-15T10:12:03Z"
}
```

### `GET /capabilities`

Response:
```json
{
  "ok": true,
  "data": {
    "input_modes": ["inline", "uri", "upload_id"],
    "supported_formats": ["text/csv", "application/json"],
    "max_inline_bytes": 1048576,
    "max_rows_inline": 5000,
    "max_top_k": 20,
    "max_link_columns": 50,
    "supports_sse": true,
    "supports_exports": true,
    "default_top_k": 5,
    "default_timeout_seconds": 900
  }
}
```

### `POST /jobs` (inline input)

Request:
```json
{
  "table_id": "my_table_001",
  "input": {
    "mode": "inline",
    "format": "application/json",
    "table": {
      "header": ["Title", "Director", "Year"],
      "rows": [
        {"row_id": "r1", "cells": ["Inception", "Christopher Nolan", "2010"]},
        {"row_id": "r2", "cells": ["Alien", "Ridley Scott", "1979"]}
      ]
    }
  },
  "link_columns": ["Title"],
  "row_range": {"start": 0, "limit": 1000},
  "top_k": 5,
  "execution": "async"
}
```

If you need to pass a per-request model API key (for example for OpenRouter), supply
`config.lion.model_api_key` in the job payload or send it via the `X-LLM-API-Key` header.
The API stores it ephemerally and does not persist it in the job record.

Response (always returns a `job_id`):
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "table_id": "my_table_001",
  "status": "queued",
  "created_at": "2025-01-15T10:12:08Z",
  "limits": {
    "top_k": 5
  }
}
```

### `POST /jobs` (large input via URI)

Request:
```json
{
  "table_id": "big_table_2025_01",
  "input": {
    "mode": "uri",
    "format": "text/csv",
    "uri": "s3://bucket/path/table.csv"
  },
  "link_columns": [0],
  "row_range": {"start": 0, "limit": 2000000},
  "top_k": 10,
  "execution": "async"
}
```

### `POST /uploads` (optional pre-signed upload)

Request:
```json
{
  "content_type": "text/csv",
  "content_length": 52428800
}
```

Response:
```json
{
  "ok": true,
  "upload_id": "up_01J2H2B5X7FQ",
  "upload_url": "/uploads/up_01J2H2B5X7FQ",
  "expires_at": "2025-01-15T10:27:08Z"
}
```

### `POST /jobs` (use upload_id)

Request:
```json
{
  "table_id": "big_table_2025_01",
  "input": {
    "mode": "upload_id",
    "format": "text/csv",
    "upload_id": "up_01J2H2B5X7FQ"
  },
  "link_columns": ["Title"],
  "top_k": 5,
  "execution": "async"
}
```

### `GET /jobs/{job_id}`

Response:
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "table_id": "my_table_001",
  "status": "running",
  "progress": {
    "rows_total": 2000000,
    "rows_processed": 325000,
    "cells_total": 2000000,
    "cells_processed": 325000
  },
  "started_at": "2025-01-15T10:12:12Z",
  "updated_at": "2025-01-15T10:20:43Z"
}
```

### `GET /jobs/{job_id}/results` (cursor-based)

Response:
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "cursor": "eyJqIjoiam9iXzAxSjJIMkE5SzlNMCIsInIiOjEwMDAsImMiOjB9",
  "next_cursor": "eyJqIjoiam9iXzAxSjJIMkE5SzlNMCIsInIiOjIwMDAsImMiOjB9",
  "results": [
    {
      "row": 0,
      "col": 0,
      "cell_id": "r1:0",
      "mention": "Inception",
      "final": {
        "entity_id": "Q25188",
        "label": "Inception",
        "score": 0.97,
        "status": "linked"
      }
    }
  ]
}
```

### `GET /jobs/{job_id}/cells/{row}/{col}/candidates`

Response:
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "row": 0,
  "col": 0,
  "cell_id": "r1:0",
  "mention": "Inception",
  "candidates": [
    {"rank": 1, "entity_id": "Q25188", "label": "Inception", "score": 0.97},
    {"rank": 2, "entity_id": "Q3797611", "label": "Inception (soundtrack)", "score": 0.21}
  ]
}
```

### `POST /jobs/{job_id}:cancel`

Response:
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "status": "cancelling"
}
```

### `GET /jobs/{job_id}/events` (optional SSE)

```
event: progress
data: {"job_id":"job_01J2H2A9K9M0","rows_processed":400000,"cells_processed":400000}
```

### `GET /jobs/{job_id}/results/download` (optional export)

Response:
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "download_url": "/jobs/job_01J2H2A9K9M0/artifacts/results.csv",
  "expires_at": "2025-01-15T11:12:08Z"
}
```

### `GET /jobs/{job_id}/artifacts` (optional)

Response:
```json
{
  "ok": true,
  "job_id": "job_01J2H2A9K9M0",
  "artifacts": [
    {"name": "results.csv", "type": "export", "size_bytes": 1048576}
  ]
}
```

## MongoDB Storage Guidance (High Level)

Collections:
- `jobs`: job metadata, config, status, timestamps
- `job_events`: progress/events (SSE source), append-only
- `cell_predictions`: final decisions per cell
- `cell_candidates`: per-cell candidates (or embedded for small jobs)
- `uploads`: upload handles, URIs, expiry

Chunking strategy:
- Store predictions in row-based chunks (for example 5k-20k rows per chunk) to avoid oversized documents.
- Store candidates separately keyed by `(job_id, row, col)` to keep `/results` lightweight.
- Stream writes and batch inserts for massive tables to avoid write amplification.

Index recommendations:
- `cell_predictions`: compound `(job_id, row, col)`
- `cell_predictions`: `(job_id, status, score)`
- `cell_predictions`: `(job_id, row)` for pagination ranges
- `cell_candidates`: `(job_id, row, col)`
- `jobs`: `(status, updated_at)` and `(table_id)`

## OpenAPI-like Outline

- `GET /health`
  - Returns liveness state.
- `GET /capabilities`
  - Returns limits, supported input modes, defaults.
- `POST /jobs`
  - Creates a job. Key fields: `table_id`, `input`, `link_columns`, `row_range`, `top_k`, `execution`.
- `GET /jobs/{job_id}`
  - Returns status and progress counters.
- `GET /jobs/{job_id}/results`
  - Returns cursor-paged final decisions (no candidate lists).
- `GET /jobs/{job_id}/cells/{row}/{col}/candidates`
  - Returns candidates for a specific cell.
- `POST /jobs/{job_id}:cancel`
  - Cancels a job.
- Optional:
  - `POST /uploads`
  - `GET /jobs/{job_id}/events`
  - `GET /jobs/{job_id}/results/download`
  - `GET /jobs/{job_id}/artifacts`
