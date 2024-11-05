import unittest
import pandas as pd
from lion_linker.lion_linker import LionLinker
import os
import asyncio

class TestLionLinker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define paths to the files
        cls.input_csv = './tests/data/film_with_QIDs.csv'  # Input file with GT column
        cls.output_csv = 'output_test.csv'  # File where LionLinker's results will be saved
        cls.prompt_file = 'prompt_template.txt'  # File containing the prompt template
        cls.gt_column_name = 'Title_QID'  # Ground truth column

        # Define LionLinker parameters
        cls.model_name = 'llama3.2:1b'  # Replace with your model name
        cls.api_url = 'https://lamapi.hel.sintef.cloud/lookup/entity-retrieval'  # API URL
        cls.api_token = 'lamapi_demo_2023'  # Your API token (if needed)
        cls.batch_size = 10  # Define the batch size

        # Initialize LionLinker with the GT column specified for exclusion
        cls.lion_linker = LionLinker(
            input_csv=cls.input_csv,  # Input CSV with GT column
            prompt_file=cls.prompt_file,
            model_name=cls.model_name,
            api_url=cls.api_url,
            api_token=cls.api_token,
            output_csv=cls.output_csv,  # Output file for results
            batch_size=cls.batch_size,
            mention_columns=["title"],  # List of columns to use for entity linking
            api_limit=10,
            compact_candidates=True,
            gt_columns=[cls.gt_column_name],  # GT column to exclude from the input
        )
    
    def test_run_lion_linker(self):
        # Run the asynchronous test within an event loop
        asyncio.run(self._run_lion_linker_test())

    async def _run_lion_linker_test(self):
        # Run LionLinker asynchronously
        await self.lion_linker.run()

        # Load LionLinker's results
        results_df = pd.read_csv(self.output_csv)

        # Verify that the output file is not empty
        self.assertFalse(results_df.empty, "The output CSV should not be empty after processing")

        # Load ground truth data for comparison
        input_df = pd.read_csv(self.input_csv)
        gt_column = input_df[self.gt_column_name]

        # Ensure LionLinker has a column 'Extracted Identifier' in the results
        self.assertIn('Extracted Identifier', results_df.columns, 
                      "The output CSV should contain an 'Extracted Identifier' column")

        # Compute accuracy by comparing predicted QIDs with the ground truth
        predicted_QIDs = results_df['Extracted Identifier']
        accuracy = (predicted_QIDs == gt_column).mean()

        # Print accuracy for information purposes
        print(f"Accuracy: {accuracy * 100:.2f}%")

    @classmethod
    def tearDownClass(cls):
        # Clean up the output file after tests
        if os.path.exists(cls.output_csv):
            os.remove(cls.output_csv)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)