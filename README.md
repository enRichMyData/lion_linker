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

To use the [HuggingFace backend](https://huggingface.co/docs/transformers/en/index) run the following:

```bash
pip install git+https://github.com/enrichMyData/lion_linker.git#egg=lion_linker[tablellama]
pip install -U huggingface-hub
```

### Step 2: Install **Ollama**

**Ollama** is required for the large language models to function with **lion_linker**. You must install it separately and ensure it is running locally.

1. Download and install **Ollama** from [Ollama's official website](https://ollama.com/download).
2. After installation, start the **Ollama** service locally by running the following command:

```bash
ollama serve
```

This ensures that the LLM models can be accessed by **lion_linker** during the entity linking process.

### Step 3:
lion_linker requires certain environment variables to be set, such as the API_URL and API_TOKEN for the entity retrieval API. To make this process easier, a .env.template file is provided in the repository.

1.	Create a .env file by copying .env.template:
```bash
cp .env.template .env
```

2.	Edit the .env file to add your specific API details. Open .env in a text editor and fill in the required values:
```bash
RETRIEVER_ENDPOINT=https://lamapi.hel.sintef.cloud/lookup/entity-retrieval
RETRIEVER_TOKEN=your_api_token  # Replace with your actual API token
```
The .env file will be used to securely store your Retriever credentials and other sensitive configuration data, so make sure it is not committed to version control.

3.	Verify the .env file by checking that RETRIEVER_ENDPOINT and RETRIEVER_TOKEN are correctly set, as these values will be automatically loaded by lion_linker when it runs.

## Usage

### Python Example

```python
import os
import subprocess

from dotenv import load_dotenv

from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient

# Load environment variables from the .env file
load_dotenv()

# Define necessary file paths and parameters
input_csv = "tests/data/film.csv"
prompt_file_path = "lion_linker/prompt/prompt_template.txt"
model_name = "gemma2:2b"  # Use the correct model name
output_csv = "output_test.csv"
chunk_size = 16  # How many rows to process
num_candidates = 20  # Maximum number of candidates from the Retriever per mention
format_candidates = True  # Format candidates as in TableLlama prompt
table_ctx_size = 1

# Load API parameters from environment variables
retriever_endpoint = os.getenv("RETRIEVER_ENDPOINT")
retriever_token = os.getenv("RETRIEVER_TOKEN")

# Additional parameters as per the latest LionLinker version
mention_columns = ["title"]  # Columns to link entities from
compact_candidates = True  # Whether to compact candidates list
model_api_provider = "ollama"  # Optional model API provider
ollama_host = "http://localhost:11434"  # Default Ollama host if not specified it will use the Default Ollama host anyway
model_api_key = None  # Optional model API key if required
gt_columns = []  # Specify any ground truth columns to exclude for testing

# Initialize the retriever instance
retriever = LamapiClient(retriever_endpoint, retriever_token, num_candidates=num_candidates)

# Initialize the LionLinker instance
lion_linker = LionLinker(
    input_csv=input_csv,
    model_name=model_name,
    retriever=retriever,
    output_csv=output_csv,
    prompt_file_path=prompt_file_path,
    chunk_size=chunk_size,
    mention_columns=mention_columns,
    compact_candidates=compact_candidates,
    model_api_provider=model_api_provider,
    ollama_host=ollama_host,
    model_api_key=model_api_key,
    gt_columns=gt_columns,
    table_ctx_size=table_ctx_size,
    format_candidates=format_candidates,
)

# Start the Ollama server as a background process
process = subprocess.Popen(["ollama", "serve"])

# Run the entity linking
await lion_linker.run()

# Stop the Ollama server
process.terminate()
```

### CLI Example

```bash
python -m lion_linker.cli \
  --lion.input_csv "./data/film.csv" \
  --lion.model_name "gemma2:2b" \
  --lion.mention_columns '[title]' \
  --lion.ollama_host "http://localhost:11434" \
  --lion.format_candidates True \
  --retriever.class_path lion_linker.retrievers.LamapiClient \
  --retriever.endpoint "https://lamapi.hel.sintef.cloud/lookup/entity-retrieval" \
  --retriever.token "lamapi_demo_2023" \
  --retriever.kg wikidata \
  --retriever.num_candidates 5 \
  --retriever.cache False
```

If one wants to change the retriever and for example use the Wikidata Lookup Service instead, the following can be used instead:

```bash
python -m lion_linker.cli \
  --lion.input_csv "./data/film.csv" \
  --lion.model_name "gemma2:2b" \
  --lion.mention_columns '[title]' \
  --lion.ollama_host "http://localhost:11434" \
  --lion.format_candidates True \
  --retriever.class_path lion_linker.retrievers.WikidataClient \
  --retriever.endpoint "https://query.wikidata.org/sparql" \
  --retriever.language "en" \
  --retriever.num_candidates 5
```

Another possibility is to retrieve candidates for mentions through [OpenRefine](https://openrefine.org/):

```bash
python -m lion_linker.cli \
  --lion.input_csv "./data/film.csv" \
  --lion.model_name "gemma2:2b" \
  --lion.mention_columns '[title]' \
  --lion.ollama_host "http://localhost:11434" \
  --lion.format_candidates True \
  --retriever.class_path lion_linker.retrievers.OpenRefineClient \
  --retriever.endpoint "https://wikidata.reconci.link/api" \
  --retriever.num_candidates 5
```

### Explanation of Parameters

- `input_csv`: Path to your input CSV file.
- `output_csv`: Path where the output file will be saved.
- `ollama_host`: The host where the Ollama service is running.
- `--prompt_file_path`: Path to a file containing a custom prompt template.
- `--model`: The LLM model to use for entity linking.
- `--chunk_size`: Defines how many rows to process at once.
- `--mention_columns`: Columns in the CSV that contain entity mentions.
- `--num_candidates`: Maximum number of candidates returned by the API per mention.

## Running Tests

You can run the tests with:

```bash
python -m unittest discover -s tests
```

This will execute all unit tests in the `tests/` directory.

## License

This project is licensed under the terms of the [MIT License](LICENSE).