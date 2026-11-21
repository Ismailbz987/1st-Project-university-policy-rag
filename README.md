# University Policy RAG (Learning Project)

This is a small learning project. The policy files in `data/source_documents/` are invented examples and are not real university rules.

## Day 51: create source-aware chunks

The first step reads Markdown policy files, splits their level-two sections into chunks, keeps source metadata, and writes JSON Lines to `data/processed/chunks.jsonl`.

Requirements: Python 3.10 or newer. This first step uses only Python's standard library.

From this project folder, run:

```powershell
python src/ingest.py
```

The script reads `.md` files from `data/source_documents/`. It expects each policy section to begin with a Markdown level-two heading (`##`). It writes one JSON object per line. The sample evaluation questions are in `data/evaluation/questions.jsonl`.

This first version reads Markdown text only. PDF loading, embeddings, vector search, and answer generation will be added in later lessons.
