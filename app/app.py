# app/app.py
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient

app = FastAPI()


class EntityLinkRequest(BaseModel):
    # Required parameter
    input_csv: str
    # Optional parameters with defaults
    model_name: str = "gemma2:2b"
    output_csv: Optional[str] = "output.csv"
    prompt_file_path: Optional[str] = "lion_linker/prompt/prompt_template.txt"
    chunk_size: int = 64
    mention_columns: Optional[List[str]] = ["title"]
    compact_candidates: bool = True
    model_api_provider: str = "ollama"
    ollama_host: Optional[str] = None  # If not provided, fallback to env/default will be used
    model_api_key: Optional[str] = None
    gt_columns: Optional[List[str]] = None
    table_ctx_size: int = 1
    format_candidates: bool = True
    num_candidates: int = 20  # For retriever configuration


@app.post("/entity_link")
async def run_entity_link(request: EntityLinkRequest):
    # Use the provided ollama_host or default to environment variable (or Docker network name)
    ollama_host = request.ollama_host or os.getenv("OLLAMA_HOST", "http://ollama:11434")

    # Load retriever configuration from environment variables
    retriever_endpoint = os.getenv("RETRIEVER_ENDPOINT")
    retriever_token = os.getenv("RETRIEVER_TOKEN")
    if not retriever_endpoint or not retriever_token:
        raise HTTPException(
            status_code=500,
            detail=(
                "Retriever configuration (endpoint/token) is "
                "not properly set in the environment."
            ),
        )

    # Initialize the retriever using the num_candidates parameter from the request
    retriever = LamapiClient(
        retriever_endpoint, retriever_token, num_candidates=request.num_candidates
    )

    # Create a new LionLinker instance using the provided parameters
    lion_linker = LionLinker(
        input_csv=request.input_csv,
        model_name=request.model_name,
        retriever=retriever,
        output_csv=request.output_csv,
        prompt_file_path=request.prompt_file_path,
        chunk_size=request.chunk_size,
        mention_columns=request.mention_columns,
        compact_candidates=request.compact_candidates,
        model_api_provider=request.model_api_provider,
        ollama_host=ollama_host,
        model_api_key=request.model_api_key,
        gt_columns=request.gt_columns,
        table_ctx_size=request.table_ctx_size,
        format_candidates=request.format_candidates,
    )

    # Execute the entity linking process
    await lion_linker.run()

    # Return a confirmation; you could also read and return the output CSV if desired.
    return {"message": "Entity linking completed", "output_csv": request.output_csv}
