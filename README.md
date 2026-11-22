# University Policy RAG (Learning Project)

A small RAG learning project. The policy files in `data/source_documents/` are invented examples, not real university rules.

## What it does

1. Reads Markdown policy files and splits them into source-aware chunks.
2. Creates an embedding for each chunk and saves the vectors in a local Chroma database.
3. Embeds a question and retrieves the closest chunks.
4. Sends the retrieved evidence to an LLM, asks it to answer only from that evidence, and displays the source chunks.
5. Compares retrieval results with expected chunk IDs in the evaluation set.

## Setup

Requirements: Python 3.10 or newer and an OpenRouter API key. The configured NVIDIA chat and embedding endpoints currently have zero model charges, but free access has rate limits and availability may change. See [OpenRouter's current free-model list](https://openrouter.ai/collections/free-models) and [free endpoint limits](https://openrouter.ai/docs/limits).

From this project folder, create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then replace the placeholder with your OpenRouter API key. Keep `.env` private; it is excluded by `.gitignore`.

The sample policy files are invented examples. Do not send confidential documents or personal information to the configured free NVIDIA endpoint; its notice says requests may be logged and used to improve NVIDIA products and services.

## Use

Build or refresh the local vector index after changing source documents:

```powershell
python src/rag.py --build
```

Ask one question using the saved index:

```powershell
python src/rag.py
```

Check whether retrieval finds the expected evidence chunks:

```powershell
python src/rag.py --evaluate
```

The Chroma database is stored in `data/chroma/`. The prepared JSONL chunks are stored in `data/processed/`. Both are generated locally and are excluded from Git.

## Limitations

- This baseline uses vector search only. It does not yet use BM25, hybrid search, RRF, or a cross-encoder reranker.
- The sample policy files are short and synthetic.
- The LLM is instructed to abstain when the context is insufficient, but this prompt alone cannot guarantee that it will always abstain or cite correctly.
- Retrieval evaluation compares chunk IDs. It does not by itself measure answer correctness.
- The code re-creates chunk embeddings when `--build` is run. Repeated builds use API requests.

This project starts with a transparent baseline. Later lessons will improve it with LangChain and more systematic evaluation.
