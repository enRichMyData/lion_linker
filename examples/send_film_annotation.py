#!/usr/bin/env python3
"""Submit the demo film table to the LionLinker API and print annotation results.

This script demonstrates how to:

1. Read the sample CSV at ``data/film_with_QIDs_10.csv``.
2. Register it via ``POST /datasets`` and capture the returned IDs.
3. Submit an annotation request with ``POST /datasets/{datasetId}/tables/{tableId}/annotate``
   (optionally restricting to a subset of rows).
4. Poll ``GET /dataset/{datasetId}/table/{tableId}`` until the job finishes.
5. Display a compact summary of the annotated rows.

The script configures LionLinker to use the ``openrouter`` model provider. Make sure you
export ``OPENROUTER_API_KEY`` in your environment before running it.

Example usage::

    export OPENROUTER_API_KEY="sk-or-..."
    python examples/send_film_annotation.py

Optional environment variables:
    LION_LINKER_API_URL   Base URL for the API (default: http://localhost:9000)
    TABLE_CSV_PATH        Path to the CSV to upload (default: data/film_with_QIDs_10.csv)
    DATASET_NAME          Dataset name to register (default: Film Demo Dataset)
    TABLE_NAME            Table name to register (default: derived from CSV filename)
    OPENROUTER_MODEL_NAME OpenRouter model alias (default: anthropic/claude-3-haiku)
    ANNOTATION_TOKEN      Optional token passed as ?token= value on submission
    POLL_INTERVAL_SECONDS Delay between status polls (default: 5)
    ROW_IDS              Optional comma-separated list of row indices to annotate
    ROW_IDS              Optional comma-separated list of row indices to annotate
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_API_URL = "http://localhost:9000"
DEFAULT_MODEL_NAME = "openai/gpt-5-mini"
DEFAULT_POLL_INTERVAL = 5.0


def _read_table(csv_path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:  # Empty file
            raise ValueError(f"CSV file is empty: {csv_path}") from exc

        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(reader, start=1):
            rows.append({"idRow": idx - 1, "data": row})

    return header, rows


def _build_registration_payload(
    dataset_name: str,
    table_name: str,
    header: List[str],
    rows: List[Dict[str, Any]],
    *,
    model_name: str,
    model_api_key: str,
    retriever_config: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    lion_config: Dict[str, Any] = {
        "model_name": model_name,
        "model_api_provider": "openrouter",
        "model_api_key": model_api_key,
        "mention_columns": ["title"],
        "chunk_size": 32,
        "table_ctx_size": 1,
        "format_candidates": True,
        "compact_candidates": True,
    }

    payload = [
        {
            "datasetName": dataset_name,
            "tableName": table_name,
            "header": header,
            "rows": rows,
            "lionConfig": lion_config,
        }
    ]
    if retriever_config:
        payload[0]["retrieverConfig"] = retriever_config
    return payload


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_retriever_config() -> Optional[Dict[str, Any]]:
    config: Dict[str, Any] = {}

    class_path = os.getenv("RETRIEVER_CLASS_PATH")
    endpoint = os.getenv("RETRIEVER_ENDPOINT")
    token = os.getenv("RETRIEVER_TOKEN")
    num_candidates = os.getenv("RETRIEVER_NUM_CANDIDATES")
    cache_flag = os.getenv("RETRIEVER_CACHE")
    extra_json = os.getenv("RETRIEVER_EXTRA_JSON")

    if class_path:
        config["class_path"] = class_path
    if endpoint:
        config["endpoint"] = endpoint
    if token:
        config["token"] = token
    if num_candidates:
        try:
            config["num_candidates"] = int(num_candidates)
        except ValueError as exc:
            raise ValueError("RETRIEVER_NUM_CANDIDATES must be an integer") from exc
    if cache_flag:
        config["cache"] = _parse_bool(cache_flag)
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as exc:
            raise ValueError("RETRIEVER_EXTRA_JSON must be valid JSON") from exc
        if not isinstance(extra, dict):
            raise ValueError("RETRIEVER_EXTRA_JSON must decode to a JSON object")
        config.update(extra)

    return config or None


def main() -> None:
    api_url = os.getenv("LION_LINKER_API_URL", DEFAULT_API_URL).rstrip("/")
    csv_path = Path(os.getenv("TABLE_CSV_PATH", "data/film_with_QIDs_10.csv"))
    dataset_name = os.getenv("DATASET_NAME", "Film Demo Dataset")
    table_name = os.getenv("TABLE_NAME", csv_path.stem)
    model_name = os.getenv("OPENROUTER_MODEL_NAME", DEFAULT_MODEL_NAME)
    token = os.getenv("ANNOTATION_TOKEN")
    poll_interval = float(os.getenv("POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL))

    model_api_key = os.getenv("OPENROUTER_API_KEY")
    if not model_api_key:
        print("OPENROUTER_API_KEY must be set in the environment", file=sys.stderr)
        sys.exit(1)

    retriever_config = _build_retriever_config()

    header, rows = _read_table(csv_path)
    payload = _build_registration_payload(
        dataset_name,
        table_name,
        header,
        rows,
        model_name=model_name,
        model_api_key=model_api_key,
        retriever_config=retriever_config,
    )

    print(f"Registering table '{table_name}' ({len(rows)} rows) at {api_url}/datasets")
    register_resp = requests.post(
        f"{api_url}/datasets",
        json=payload,
        timeout=60,
    )
    register_resp.raise_for_status()
    registrations = register_resp.json()
    if not registrations:
        raise RuntimeError("Dataset registration returned an empty response")

    registration = registrations[0]
    dataset_id = registration["datasetId"]
    table_id = registration["tableId"]
    print(f"Registered dataset={dataset_id}, table={table_id}")

    row_ids_env = os.getenv("ROW_IDS")
    row_ids: Optional[List[int]] = None
    if row_ids_env:
        try:
            row_ids = [int(part.strip()) for part in row_ids_env.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("ROW_IDS must be a comma-separated list of integers") from exc

    annotation_payload: Dict[str, Any] = {
        "lionConfig": payload[0].get("lionConfig"),
    }
    if retriever_config:
        annotation_payload["retrieverConfig"] = retriever_config
    if row_ids:
        annotation_payload["rowIds"] = row_ids
    if token:
        annotation_payload["token"] = token

    print(
        "Submitting annotation request to "
        f"{api_url}/datasets/{dataset_id}/tables/{table_id}/annotate"
    )
    annotate_resp = requests.post(
        f"{api_url}/datasets/{dataset_id}/tables/{table_id}/annotate",
        json=annotation_payload,
        timeout=60,
    )
    annotate_resp.raise_for_status()
    job = annotate_resp.json()
    job_id = job["jobId"]
    print(f"Job {job_id} created")

    status_url = f"{api_url}/dataset/{dataset_id}/table/{table_id}"
    while True:
        time.sleep(poll_interval)
        try:
            status_response = requests.get(
                status_url,
                params={"page": 1, "per_page": 50},
                timeout=(5, 180),
            )
            status_response.raise_for_status()
        except requests.exceptions.ReadTimeout:
            print("Status request timed out; will retry on next poll...")
            continue

        status_payload = status_response.json()

        status = status_payload.get("status")
        message = status_payload.get("message")
        print(f"Status: {status} - {message or 'no message'}")

        if status == "completed":
            rows_payload = status_payload.get("rows", [])
            print(f"\nTop {len(rows_payload)} annotated rows:")
            for row in rows_payload:
                title = row["data"][0]
                predictions = ", ".join(
                    f"{pred['column']} → {pred['answer']}" for pred in row.get("predictions", [])
                )
                print(f"  • {title}: {predictions or 'no predictions returned'}")
                meta = row.get("annotationMeta")
                if meta:
                    print(f"    meta job={meta.get('jobId')} updatedAt={meta.get('updatedAt')}")
            break
        if status == "failed":
            raise RuntimeError(f"Annotation failed: {message}")

    info_response = requests.get(
        f"{api_url}/annotate/{job_id}",
        timeout=(5, 60),
        params={"token": token} if token else None,
    )
    info_response.raise_for_status()
    info_payload = info_response.json()
    print("\nJob metadata:")
    lion_config = info_payload.get("lionConfig") or {}
    retriever_config = info_payload.get("retrieverConfig") or {}
    print(f"  Model: {lion_config.get('model_name')}")
    if retriever_config:
        print(f"  Retriever config keys: {list(retriever_config.keys())}")
    else:
        print("  Retriever config: <none provided>")
    row_ids_info = info_payload.get("rowIds")
    if row_ids_info:
        print(f"  Row IDs: {row_ids_info}")


if __name__ == "__main__":
    main()
