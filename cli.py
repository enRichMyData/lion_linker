import argparse
import logging
import asyncio  # Import asyncio to handle the async execution
from lion_linker.lion_linker import LionLinker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    parser = argparse.ArgumentParser(description='Entity Linking with lion_linker.')
    parser.add_argument('input_csv', help='Path to the input CSV file.')
    parser.add_argument('output_csv', help='Path to the output CSV file.')
    parser.add_argument('--api-url', required=True, help='Entity retrieval API URL.')
    parser.add_argument('--api-token', help='Optional API token.')
    parser.add_argument('--prompt-file', required=True, help='File containing prompt template.')
    parser.add_argument('--model', required=True, help='LLM model name.')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing.')
    parser.add_argument('--mention_columns', nargs='*', help='List of columns containing entity mentions.')

    args = parser.parse_args()

    # Initialize the LionLinker instance
    lion_linker = LionLinker(
        input_csv=args.input_csv,
        prompt_file=args.prompt_file,
        model_name=args.model,
        api_url=args.api_url,
        api_token=args.api_token,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        mention_columns=args.mention_columns
    )

    # Run the processing
    try:
        await lion_linker.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use asyncio.run to run the async main function
    asyncio.run(main())