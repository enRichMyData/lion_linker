import argparse
import pandas as pd
from lion_linker.core import APIClient, PromptGenerator, LLMInteraction
from lion_linker.utils import clean_data, process_in_batches, compute_table_summary

def main():
    parser = argparse.ArgumentParser(description='Entity Linking with lion_linker.')
    parser.add_argument('input_csv', help='Path to the input CSV file.')
    parser.add_argument('output_csv', help='Path to the output CSV file.')
    parser.add_argument('--api-url', required=True, help='Entity retrieval API URL.')
    parser.add_argument('--api-token', help='Optional API token.')
    parser.add_argument('--prompt-file', required=True, help='File containing prompt template.')
    parser.add_argument('--model', required=True, help='LLM model name.')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing.')
    parser.add_argument('--sample-size', type=int, default=5, help='Number of rows to sample for the table summary.')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    cleaned_df = clean_data(df)
    table_summary = compute_table_summary(df, sample_size=args.sample_size)

    api_client = APIClient(args.api_url, token=args.api_token)
    prompt_generator = PromptGenerator(args.prompt_file)
    llm_interaction = LLMInteraction(args.model)

    def process_batch(batch_df):
        results = []
        for index, row in batch_df.iterrows():
            entity_mention = row['title']  # Using 'title' as the entity to be linked
            candidates = api_client.fetch_entities(entity_mention)  # Fetch candidates based on the entity mention
            prompt = prompt_generator.generate_prompt(
                table_summary,
                row.to_dict(),
                'title',
                entity_mention,
                candidates
            )
            llm_response = llm_interaction.get_entity_links(prompt)
            results.append({
                'Mention Identifier': f"{args.input_csv}-{index}-{row.name}",
                'LLM Answer': llm_response,
                'Predicted Entity ID': llm_response  # Process the response to extract the ID
            })
        return results

    results = process_in_batches(cleaned_df, args.batch_size, process_batch)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()