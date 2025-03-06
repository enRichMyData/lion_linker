import argparse
import asyncio
import logging
import os
import traceback

from dotenv import load_dotenv

from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient

# Load environment variables from .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main():
    parser = argparse.ArgumentParser(description="Entity Linking with LionLinker.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")
    parser.add_argument(
        "--retriever_endpoint",
        default=os.getenv("RETRIEVER_ENDPOINT"),
        help="Entity retrieval URL.",
    )
    parser.add_argument(
        "--retriever_token", default=os.getenv("RETRIEVER_TOKEN"), help="Optional retriever token."
    )
    parser.add_argument(
        "--prompt_file_path", required=True, help="File containing prompt template."
    )
    parser.add_argument("--model", required=True, help="LLM model name.")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for processing.")
    parser.add_argument(
        "--mention_columns",
        nargs="*",
        required=True,
        help="List of columns containing entity mentions.",
    )
    parser.add_argument(
        "--num_candidates", type=int, default=20, help="Limit for API calls per batch."
    )
    parser.add_argument(
        "--compact_candidates", action="store_true", help="Whether to compact candidates."
    )
    parser.add_argument(
        "--format_candidates",
        action="store_true",
        help="Whether to format candidates in the prompt as in TableLlama.",
    )
    parser.add_argument(
        "--table_ctx_size",
        type=int,
        default=1,
        help="Number of rows to include in the table context.",
    )
    parser.add_argument(
        "--gt_columns",
        nargs="*",
        default=[],
        help="Columns containing ground truth data to exclude",
    )
    parser.add_argument(
        "--model_api_provider", default="ollama", help="Optional model API provider name."
    )
    parser.add_argument("--ollama_host", default=None, help="Optional OLLAMA host.")
    parser.add_argument("--model_api_key", default="", help="Optional model API key.")

    args = parser.parse_args()

    retriever = LamapiClient(
        endpoint=args.retriever_endpoint,
        token=args.retriever_token,
        num_candidates=args.num_candidates,
    )

    # Initialize the LionLinker instance with the parsed arguments
    lion_linker = LionLinker(
        input_csv=args.input_csv,
        prompt_file_path=args.prompt_file_path,
        model_name=args.model,
        retriever=retriever,
        output_csv=args.output_csv,
        chunk_size=args.chunk_size,
        mention_columns=args.mention_columns,
        compact_candidates=args.compact_candidates,
        model_api_provider=args.model_api_provider,
        ollama_host=args.ollama_host,
        model_api_key=args.model_api_key,
        gt_columns=args.gt_columns,
        table_ctx_size=args.table_ctx_size,
        format_candidates=args.format_candidates,
    )

    # Run the processing
    try:
        await lion_linker.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    # Use asyncio.run to run the async main function
    asyncio.run(main())
