# LionLinker REST API

This document describes the REST endpoints exposed by the LionLinker service. All endpoints return JSON responses and use snake_case field names. LionLinker-specific behaviour is controlled through the payload via `lionConfig` and `retrieverConfig` objects; no implicit defaults are injected from server settings.

## Base URL

```
http://<host>:9000
```

## Authentication

The reference implementation accepts an optional `token` query parameter on job-related endpoints. If provided during submission, the same value must be supplied when querying status.

## Endpoints

### 1. Register Tables

```
POST /dataset
Content-Type: application/json
```

Registers one or more tables in the store. Each element matches the onboarding payload:

```json
[
  {
    "datasetName": "EMD-BC",
    "tableName": "SN-BC-1753173071015729",
    "header": ["Point of Interest", "Place"],
    "rows": [
      {"idRow": 1, "data": ["John F. Kennedy Presidential Library and Museum", "Columbia Point"]},
      {"idRow": 2, "data": ["Petrie Museum of Egyptian Archaeology", "London"]}
    ],
    "lionConfig": {
      "model_name": "gemma2:2b",
      "chunk_size": 16,
      "mention_columns": ["Point of Interest"],
      "table_ctx_size": 2
    },
    "retrieverConfig": {
      "class_path": "lion_linker.retrievers.LamapiClient",
      "endpoint": "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval",
      "token": "lamapi_demo_2023",
      "num_candidates": 5
    }
  }
]
```

**Response** – array of objects containing `datasetId`, `tableId`, and metadata.

### 2. Trigger Annotation Runs

```
POST /annotate
Content-Type: application/json
```

Enqueue LionLinker runs for one or more tables. The payload matches `POST /dataset`; any tables not yet registered will be upserted automatically.

Optional query parameter:

- `token`: value stored with the job and required when polling status (optional).

**Response** – array of job descriptors:

```json
[
  {
    "jobId": "91723dec3eb64dd68a358a057de0bc27",
    "datasetId": "6177cc032e0d40628b6f00fcfa9d8310",
    "tableId": "50743956de014184acb49288874f4110",
    "status": "pending",
    "createdAt": "2025-09-24T15:04:16.108905Z"
  }
]
```

### 3. Poll Annotation Status (latest for table)

```
GET /dataset/{datasetId}/table/{tableId}?page=1&per_page=50[&token=...]
```

Returns the most recent job for the given table along with prediction rows. Predictions are streamed from MongoDB in pages directly from the stored table rows; if a job predates prediction persistence the API falls back to the legacy JSON file.

**Response**

```json
{
  "datasetId": "6177cc032e0d40628b6f00fcfa9d8310",
  "tableId": "50743956de014184acb49288874f4110",
  "jobId": "91723dec3eb64dd68a358a057de0bc27",
  "status": "completed",
  "page": 1,
  "perPage": 50,
  "totalRows": 2,
  "rows": [
    {
      "idRow": 1,
      "data": ["John F. Kennedy Presidential Library and Museum", "Columbia Point"],
      "predictions": [
        {
          "column": "Point of Interest",
          "answer": "ANSWER:Q632682",
          "identifier": "Q632682"
        }
      ]
    }
  ],
  "message": "Linking completed",
  "updatedAt": "2025-09-24T15:05:22.101000",
  "predictionBatches": 1,
  "predictionBatchSize": 200
}
```

### 4. Fetch Annotation Metadata by ID

```
GET /annotate/{jobId}[?token=...]
```

Returns job state without streaming rows. Helpful for checking configuration echoes (`lionConfig`, `retrieverConfig`) and understanding how many prediction batches are stored.

**Response**

```json
{
  "jobId": "91723dec3eb64dd68a358a057de0bc27",
  "datasetId": "6177cc032e0d40628b6f00fcfa9d8310",
  "tableId": "50743956de014184acb49288874f4110",
  "status": "completed",
  "totalRows": 2,
  "processedRows": 2,
  "message": "Linking completed",
  "updatedAt": "2025-09-24T15:05:22.101000",
  "lionConfig": {
    "model_name": "gemma2:2b",
    "chunk_size": 16,
    "mention_columns": ["Point of Interest"],
    "table_ctx_size": 2
  },
  "retrieverConfig": {
    "class_path": "lion_linker.retrievers.LamapiClient",
    "endpoint": "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval",
    "num_candidates": 5,
    "token": "lamapi_demo_2023"
  },
  "predictionBatches": 1,
  "predictionBatchSize": 200
}
```

## Request Configuration Reference

### `lionConfig`

All keys use snake_case and map directly to `LionLinker` parameters:

| Key | Type | Description |
| --- | --- | --- |
| `model_name` | string | Model identifier (default `gemma2:2b`). |
| `chunk_size` | integer | Rows per batch; must be ≥ 1. |
| `mention_columns` | array[string] | Columns used to extract mentions. |
| `table_ctx_size` | integer | Context window around each row. |
| `prompt_template` | string | Template name or path. |
| `prompt_file_path` | string | Explicit path to a prompt file. |
| `few_shot_examples_file_path` | string | Path to few-shot examples. |
| `format_candidates` | boolean | Whether to format candidates (defaults to true). |
| `compact_candidates` | boolean | Whether to compact candidate entries (defaults to true). |
| `model_api_provider` | string | One of `ollama`, `openrouter`, `huggingface`. |
| `ollama_host` | string | Base URL for Ollama (default `http://ollama:11434`). |
| `model_api_key` | string | API key, required for OpenRouter. |
| `gt_columns` | array[string] | Columns to exclude from processing. |
| `answer_format` | string | Custom answer format. |
| `prediction_batch_size` | integer | Overrides default Mongo batch size (defaults to env `PREDICTION_BATCH_ROWS`). |

### `retrieverConfig`

Keys may be provided in either `snake_case` or camelCase; the API normalises
them to snake_case before processing.

| Key | Type | Description |
| --- | --- | --- |
| `class_path` | string | Fully qualified retriever class (defaults to `lion_linker.retrievers.LamapiClient`). |
| `endpoint` | string | Retriever endpoint URL. |
| `token` | string | Authentication token for the retriever. |
| `num_candidates` | integer | Candidate count per mention. |
| `kg` | string | Knowledge graph name (defaults to table metadata). |
| `cache` | boolean | Enable/disable retriever-side caching. |
| `max_retries` | integer | Optional retry count (if supported by the retriever). |
| `backoff_factor` | number | Optional backoff factor (if supported by the retriever). |
| `retrieverConfig` | object | Optional per-table retriever overrides. Keys may be provided in either `snake_case` or camelCase; the API normalises them to snake_case. |

Any additional fields in `retrieverConfig` are forwarded to the retriever constructor.

## Response Codes

| Status | Meaning |
| ------ | ------- |
| `200 OK` | Request processed successfully. |
| `400 Bad Request` | Malformed payload or invalid parameters. |
| `403 Forbidden` | Token mismatch when polling job status/info. |
| `404 Not Found` | Dataset, table, or job not found. |
| `500 Internal Server Error` | Unexpected failure during processing. |

## Notes

- Tables are stored row-by-row in MongoDB (`lion_table_rows`). Aggregations should use `datasetId` and `tableId` as keys.
- Predictions are persisted directly on each table row. Use `predictionBatches` and `predictionBatchSize` in job responses to understand the chunking used when the annotations were produced.
- The service still emits legacy CSV/JSON artifacts under `/app/data/api_runs/...` for compatibility, but MongoDB is the primary source of truth for processed rows.

For interactive exploration, open the FastAPI-generated docs at `http://<host>:9000/docs` or the ReDoc view at `http://<host>:9000/redoc`.
