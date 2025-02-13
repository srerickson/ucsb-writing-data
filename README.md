# Code for 'Using AI to Understand Students’ Self-Assessments of their Writing'

This code implements a simple Retrieval-Augmented Generation (RAG) pipeline to
explore AI methods for qualitative data analysis. It was developed in
collaboration with Madeleine Sorapure to explore how/whether RAG can be used
analyze student responses to a self-assessment survey used in UCSB's Writing
Program. 

## Installation

This project uses [uv](https://docs.astral.sh) to manage dependencies. 

To install dependencies with:

```bash
$ uv pip install -r pyproject.toml
```

## Overview

### `data/`

The data directory where raw survey data should be saved, if available. It is
intentionally left empty in the public dataset.

### `embeddings.py`

This script is used to generate the two embeddings files included in the public dataset
using raw survey results (not included in the public dataset):

- `outputs/mxbai_embeddings_nonnorm.parquet`
- `outputs/openai_3small.parquet`

The script expects the raw survey results in `data/reflections.csv`, an OpenAPI
key, and the Ollama service running locally. For this project, the script was
run on MacOS v15.3.1 with Ollama v0.4.4. 

Usage example:
```bash
# Set OpenAI API Key
$ export OPENAI_API_KEY=...

# Make sure ollama service is running
$ brew services start ollama

# run the script (expect it take around an hour or so)
uv run ./embeddings.py
```

### `lib/`

This directory includes python utility functions used in by `query.ipynb` and `rag.ipynb`

### `outputs/`

This directory includes files with embeddings of student responses to survey
questions (see `embeddings.py`). Embeddings are stored as [Parquet
files](https://parquet.apache.org/):

- `outputs/mxbai_embeddings_nonnorm.parquet`
- `outputs/openai_3small.parquet`

Each parquet file includes the same three columns:

- `embedding` ([]float): the embedding for a student's response to a survey question.
- `student_id` (string): opaque student identifier 
- `question_id` (string): survey question id for the response

### `query.ipynb`

This notebook demonstrates how embeddings can be used to query the survey
dataset. The raw data is read as a pandas data frame; the `search_display()`
function prints text responses in the data frame that are most similar to a given
prompt. This assumes embeddings have already been generated using
`embeddings.py`.

Use `uv run jupyter lab` to start Jupyter Lab.

### `rag.ipynb`

This notebook demonstrates a basic retrieval-augmented generation (RAG) process.
It uses search_df() to generate a "context" for a given question/prompt; the
context is included in a ChatGPT prompt and used to generate a response the
question.

Use `uv run jupyter lab` to start Jupyter Lab.
