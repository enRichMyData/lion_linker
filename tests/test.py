import asyncio  # Import asyncio to run async functions in sync context
import os
import unittest
from unittest.mock import AsyncMock, patch

from dotenv import load_dotenv

from lion_linker import PROJECT_ROOT
from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient

# Load environment variables from .env
load_dotenv()


class TestLionLinker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define required test parameters
        cls.input_csv = "tests/data/film.csv"  # Replace with a small, valid CSV path for testing
        cls.prompt_file_path = os.path.join(
            PROJECT_ROOT, "prompt", "prompt_template.txt"
        )  # Replace with a valid prompt file for testing
        cls.model_name = "llama3.2:1b"
        cls.output_csv = "output_test.csv"
        cls.chunk_size = 5
        cls.num_candidates = 5
        cls.retriever_endpoint = os.getenv("RETRIEVER_ENDPOINT")
        cls.retriever_token = os.getenv("RETRIEVER_TOKEN")
        cls.mention_columns = ["title"]
        cls.gt_columns = []
        cls.retriever = LamapiClient(
            endpoint=cls.retriever_endpoint,
            token=cls.retriever_token,
            num_candidates=cls.num_candidates,
        )

    def test_initialization(self):
        """Test LionLinker initialization with required parameters."""
        lion_linker = LionLinker(
            input_csv=self.input_csv,
            prompt_file_path=self.prompt_file_path,
            model_name=self.model_name,
            retriever=self.retriever,
            output_csv=self.output_csv,
            chunk_size=self.chunk_size,
            mention_columns=self.mention_columns,
            gt_columns=self.gt_columns,
        )
        self.assertIsInstance(
            lion_linker, LionLinker, "LionLinker instance was not created successfully"
        )

    def test_env_loading(self):
        """Test that environment variables for retriever
        endpoint and token are loaded correctly."""
        self.assertIsNotNone(self.retriever.endpoint, "RETRIEVER_URL should not be None")
        self.assertIsNotNone(self.retriever.token, "RETRIEVER_TOKEN should not be None")

    def test_run_method(self):
        """Test the run method to ensure it can be called without errors."""
        retriever = LamapiClient(
            endpoint=self.retriever_endpoint,
            token=self.retriever_token,
            num_candidates=self.num_candidates,
        )
        lion_linker = LionLinker(
            input_csv=self.input_csv,
            prompt_file_path=self.prompt_file_path,
            model_name=self.model_name,
            retriever=retriever,
            output_csv=self.output_csv,
            chunk_size=self.chunk_size,
            mention_columns=self.mention_columns,
            gt_columns=self.gt_columns,
        )

        # Run the mocked run method asynchronously
        async def mock_run():
            with patch(
                "lion_linker.lion_linker.LionLinker.run", new_callable=AsyncMock
            ) as mock_run_method:
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
