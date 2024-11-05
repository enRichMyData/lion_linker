# lion_linker

**lion_linker** is a Python library that uses Large Language Models (LLMs) to perform entity linking over tabular data. It efficiently links entity mentions in tables to relevant knowledge graph entities using customizable prompts and batch processing.

<img src="logo/logo.webp" alt="lion_linker Logo" width="200"/>

## Overview

**lion_linker** simplifies the process of entity linking by leveraging LLMs to identify and connect textual data in tables to relevant entities in a knowledge base. It is highly customizable, making it ideal for projects involving research, NLP, and large-scale data processing.

## Features

- **Entity Linking with LLMs**: Provides accurate and context-aware entity linking using large language models.
- **Command Line Interface (CLI)**: Process CSV files via the command line for ease of use and automation.
- **Customizable Prompt Templates**: Adjust the linking prompt to fit your data structure.
- **Scalable for Large Datasets**: Process large datasets by batching the work and customizing API result limits.
- **Flexible API**: Programmatic interface for advanced users.

## Installation

To use **lion_linker**, you will need to install both the Python package and **Ollama**, which must be running locally.

### Step 1: Install **lion_linker** via pip

You can install the latest version directly from the GitHub repository using the following command:

```bash
pip install git+https://github.com/enRichMyData/lion_linker.git
```

Alternatively, clone the repository and install from source:

```bash
git clone https://github.com/enRichMyData/lion_linker.git
cd lion_linker
pip install -e .
```

### Step 2: Install **Ollama**

**Ollama** is required for the large language models to function with **lion_linker**. You must install it separately and ensure it is running locally.

1. Download and install **Ollama** from [Ollama's official website](https://ollama.com/download).
2. After installation, start the **Ollama** service locally by running the following command:

```bash
ollama serve
```

This ensures that the LLM models can be accessed by **lion_linker** during the entity linking process.

## Usage

### Python Example

```python
from lion_linker.lion_linker import LionLinker

# Define necessary file paths and parameters
input_csv = 'tests/data/film.csv'
prompt_file = 'prompt_template.txt'
model_name = 'gemma2:2b'  # Use the correct model name
api_url = 'https://lamapi.hel.sintef.cloud/lookup/entity-retrieval'  # Replace with actual API URL
api_token = 'your_api_token'  # Replace with your API token if needed
output_csv = 'output_test.csv'
batch_size = 10  # Set batch size for processing
api_limit = 20  # Maximum number of results from the API per request

# Initialize the LionLinker instance
lion_linker = LionLinker(
    input_csv=input_csv,
    prompt_file=prompt_file,
    model_name=model_name,
    api_url=api_url,
    api_token=api_token,
    output_csv=output_csv,
    batch_size=batch_size,
    mention_columns=["title"],  # Columns to link entities from
    api_limit=api_limit  # Maximum number of API results per batch
)

# Run the entity linking
await lion_linker.run()
```

### CLI Example

```bash
python3 cli.py tests/data/film.csv output_test.csv \
  --api-url https://lamapi.hel.sintef.cloud/lookup/entity-retrieval \
  --api-token your_api_token \
  --prompt-file prompt_template.txt \
  --model gemma2:2b \
  --batch-size 10 \
  --mention_columns title \
  --api-limit 20
```

### Explanation of Parameters

- `input_csv`: Path to your input CSV file.
- `output_csv`: Path where the output file will be saved.
- `--api-url`: URL for the entity retrieval API.
- `--api-token`: Optional API token for authentication.
- `--prompt-file`: Path to a file containing a custom prompt template.
- `--model`: The LLM model to use for entity linking.
- `--batch-size`: Defines how many rows to process at once.
- `--mention_columns`: Columns in the CSV that contain entity mentions.
- `--api-limit`: Maximum number of results returned by the API per batch.

## Running Tests

You can run the tests with:

```bash
python -m unittest discover -s tests
```

This will execute all unit tests in the `tests/` directory.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
