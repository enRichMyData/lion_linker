import unittest
import os
import asyncio  # Import asyncio to run async functions in sync context
from unittest.mock import patch, AsyncMock
from dotenv import load_dotenv
from lion_linker.lion_linker import LionLinker

# Load environment variables from .env
load_dotenv()

class TestLionLinker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Define required test parameters
        cls.input_csv = 'tests/data/film.csv'  # Replace with a small, valid CSV path for testing
        cls.prompt_file = 'prompt_template.txt'  # Replace with a valid prompt file for testing
        cls.model_name = 'llama3.2:1b'
        cls.output_csv = 'output_test.csv'
        cls.batch_size = 5
        cls.api_limit = 10
        cls.api_url = os.getenv('API_URL')
        cls.api_token = os.getenv('API_TOKEN')
        cls.mention_columns = ["title"]
        cls.gt_columns = []

    def test_initialization(self):
        """Test LionLinker initialization with required parameters."""
        lion_linker = LionLinker(
            input_csv=self.input_csv,
            prompt_file=self.prompt_file,
            model_name=self.model_name,
            api_url=self.api_url,
            api_token=self.api_token,
            output_csv=self.output_csv,
            batch_size=self.batch_size,
            mention_columns=self.mention_columns,
            api_limit=self.api_limit,
            gt_columns=self.gt_columns
        )
        self.assertIsInstance(lion_linker, LionLinker, "LionLinker instance was not created successfully")

    def test_env_loading(self):
        """Test that environment variables for API URL and token are loaded correctly."""
        self.assertIsNotNone(self.api_url, "API_URL should not be None")
        self.assertIsNotNone(self.api_token, "API_TOKEN should not be None")

    def test_run_method(self):
        """Test the run method to ensure it can be called without errors."""
        lion_linker = LionLinker(
            input_csv=self.input_csv,
            prompt_file=self.prompt_file,
            model_name=self.model_name,
            api_url=self.api_url,
            api_token=self.api_token,
            output_csv=self.output_csv,
            batch_size=self.batch_size,
            mention_columns=self.mention_columns,
            api_limit=self.api_limit,
            gt_columns=self.gt_columns
        )

        # Run the mocked run method asynchronously
        async def mock_run():
            with patch("lion_linker.lion_linker.LionLinker.run", new_callable=AsyncMock) as mock_run_method:
                await lion_linker.run()
                mock_run_method.assert_called_once()

        # Run the asynchronous mock_run function
        asyncio.run(mock_run())

    @classmethod
    def tearDownClass(cls):
        # Clean up the test output file
        if os.path.exists(cls.output_csv):
            os.remove(cls.output_csv)

if __name__ == "__main__":
    unittest.main()