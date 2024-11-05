import argparse
import logging
import asyncio
import os
from dotenv import load_dotenv
from lion_linker.lion_linker import LionLinker

# Load environment variables from .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    parser = argparse.ArgumentParser(description='Entity Linking with LionLinker.')
    parser.add_argument('input_csv', help='Path to the input CSV file.')
    parser.add_argument('output_csv', help='Path to the output CSV file.')
    parser.add_argument('--api-url', default=os.getenv('API_URL'), help='Entity retrieval API URL.')
    parser.add_argument('--api-token', default=os.getenv('API_TOKEN'), help='Optional API token.')
    parser.add_argument('--prompt-file', required=True, help='File containing prompt template.')
    parser.add_argument('--model', required=True, help='LLM model name.')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing.')
    parser.add_argument('--mention_columns', nargs='*', required=True, help='List of columns containing entity mentions.')
    parser.add_argument('--api-limit', type=int, default=20, help='Limit for API calls per batch.')
    parser.add_argument('--compact-candidates', action='store_true', help='Whether to compact candidates.')
    parser.add_argument('--gt_columns', nargs='*', default=[], help='Columns containing ground truth data to exclude')
    parser.add_argument('--model-api-provider', default='ollama', help='Optional model API provider name.')
    parser.add_argument('--model-api-key', default='', help='Optional model API key.')

    args = parser.parse_args()

    # Initialize the LionLinker instance with the parsed arguments
    lion_linker = LionLinker(
        input_csv=args.input_csv,
        prompt_file=args.prompt_file,
        model_name=args.model,
        api_url=args.api_url,
        api_token=args.api_token,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        mention_columns=args.mention_columns,
        api_limit=args.api_limit,
        compact_candidates=args.compact_candidates,
        model_api_provider=args.model_api_provider,
        model_api_key=args.model_api_key,
        gt_columns=args.gt_columns
    )

    # Run the processing
    try:
        await lion_linker.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use asyncio.run to run the async main function
    asyncio.run(main())