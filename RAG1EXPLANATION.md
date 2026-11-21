# RAG 1: `ingest.py` Code Explanation

This file prepares policy documents for a RAG system. It reads Markdown files, splits their sections into smaller text chunks, adds source information, and saves each chunk as a JSONL record. These chunks are prepared for later embedding and retrieval; this script does not yet create embeddings or answer questions.

## 1. Imports

```python
import json
from pathlib import Path
```

- **`json`** provides tools to convert Python data into JSON text for the output file.
- **`Path`** helps build and work with file and folder paths.

## 2. Project paths and chunk settings

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "source_documents"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
MAX_CHARS = 700
OVERLAP_CHARS = 100
```

- **`__file__`** is the path of this Python file, `ingest.py`.
- **`resolve()`** makes that path absolute.
- **`parents[1]`** goes up two folder levels from `src/ingest.py` to the project folder.
- **`PROJECT_ROOT`** stores the project folder path.
- **`SOURCE_DIR`** points to the folder containing the Markdown policy files.
- **`OUTPUT_FILE`** points to the JSONL file this program will create.
- **`MAX_CHARS`** sets the maximum number of characters in a chunk.
- **`OVERLAP_CHARS`** sets how many characters neighboring chunks share. Overlap can help keep meaning that crosses a chunk boundary.

## 3. Split one section into chunks

```python
def split_section(text: str) -> list[str]:
    """Split one policy section into character-sized pieces with overlap."""
    text = " ".join(text.split())
    if not text:
        return []

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - OVERLAP_CHARS
    return pieces
```

- **`def split_section(...)`** defines a function that splits one section of text.
- **`text: str`** means the input should be text.
- **`-> list[str]`** means the function returns a list of text pieces.
- **`text.split()`** separates the text at whitespace and removes extra spaces and line breaks; **`" ".join(...)`** joins the words with single spaces.
- **`if not text`** checks whether the cleaned text is empty. If it is, **`return []`** returns an empty list.
- **`pieces`** is the list that will hold the chunks.
- **`start`** is the first character position for the current chunk.
- **`while start < len(text)`** keeps making chunks until the whole text has been handled. **`len(text)`** is the number of characters in the text.
- **`end`** is the position where the current chunk stops. **`min(...)`** prevents it from going past the end of the text.
- **`text[start:end]`** selects the current piece of text, and **`pieces.append(...)`** adds it to the list.
- **`if end == len(text)`** checks whether this piece reached the end. **`break`** stops the loop if it did.
- **`start = end - OVERLAP_CHARS`** moves the next chunk's start back by 100 characters, creating overlap.
- **`return pieces`** gives the completed chunk list back to the part of the program that called the function.

**Role in RAG:** this makes long policy sections small enough to search as individual passages, while overlap helps preserve nearby context.

## 4. Find sections in a Markdown document

```python
def sections_from_markdown(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs, using each level-two heading."""
    sections = []
    heading = None
    body_lines = []

    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(body_lines).strip()))
            heading = line[3:].strip()
            body_lines = []
        elif heading is not None:
            body_lines.append(line)

    if heading is not None:
        sections.append((heading, "\n".join(body_lines).strip()))

    return sections
```

- **`sections_from_markdown`** defines a function that finds sections in Markdown text.
- **`text`** is the whole document. **`list[tuple[str, str]]`** means the result is a list of pairs: each pair contains a heading and its body.
- **`sections`** collects the completed heading/body pairs.
- **`heading`** stores the current section heading. It starts as **`None`**, meaning no heading has been found yet.
- **`body_lines`** collects the lines belonging to the current section.
- **`text.splitlines()`** gives the function one document line at a time. **`line`** is the current line.
- **`line.startswith("## ")`** checks for a level-two Markdown heading, such as `## Dropping a course`.
- When a new heading appears, **`if heading is not None`** checks whether there is an earlier section to save. **`"\n".join(body_lines).strip()`** combines its body lines and removes whitespace at the ends. **`sections.append((heading, ...))`** saves that heading and body as a pair.
- **`line[3:].strip()`** takes the text after the `## ` marker and removes extra whitespace. That becomes the new **`heading`**.
- **`body_lines = []`** starts collecting the new section's body.
- **`elif heading is not None`** adds a line to **`body_lines`** only after a heading has been found.
- After the loop, the final section is saved too, because there may not be another heading to trigger that step.
- **`return sections`** gives all the section pairs back to the caller.

**Role in RAG:** preserving section headings helps keep policy topics attached to their text when the text is split into chunks.

## 5. Turn one source file into metadata-rich chunks

```python
def make_chunks(source_file: Path) -> list[dict]:
    """Read one Markdown file and return chunks with source metadata."""
    document_text = source_file.read_text(encoding="utf-8")
    chunks = []
    chunk_number = 0

    for heading, body in sections_from_markdown(document_text):
        section_text = f"{heading}\n{body}"
        for piece in split_section(section_text):
            chunk_number += 1
            chunks.append(
                {
                    "chunk_id": f"{source_file.stem}:chunk-{chunk_number:03d}",
                    "text": piece,
                    "source": source_file.name,
                    "page": None,
                    "topic": source_file.stem,
                }
            )

    return chunks
```

- **`source_file`** is the path to one Markdown document. **`Path`** says it is a file path; **`list[dict]`** says the function returns a list of dictionaries (records).
- **`source_file.read_text(encoding="utf-8")`** reads the document as text. **`document_text`** holds that text.
- **`chunks`** starts as an empty list to collect this file's chunks.
- **`chunk_number`** starts at zero and counts chunks made from this file.
- **`for heading, body in sections_from_markdown(document_text)`** gets each section pair in order. On each pass, **`heading`** is that section's title and **`body`** is its text.
- **`section_text = f"{heading}\n{body}"`** joins the heading and body, with a newline between them. The `f` string inserts the values of `heading` and `body`.
- **`for piece in split_section(section_text)`** splits that section into smaller pieces. **`piece`** is the current piece.
- **`chunk_number += 1`** increases the counter by one for each new piece. The number is used in its ID.
- **`chunks.append({...})`** adds a dictionary describing the current piece:
  - **`"chunk_id"`** is a unique label for this chunk. **`source_file.stem`** is the filename without its extension; **`{chunk_number:03d}`** formats the number with at least three digits, such as `001`.
  - **`"text"`** stores the actual chunk text.
  - **`"source"`** stores the filename, including its extension, so the result can be traced to its document.
  - **`"page": None`** records that no page number is available. Markdown files do not provide PDF page numbers here.
  - **`"topic"`** stores the filename without its extension. This is the topic label used by this version of the code.
- The loops process every piece of every section. **`return chunks`** gives the finished records back to the caller.

**Role in RAG:** this is where document text becomes searchable chunk records, each with information that can later identify its source.

## 6. Process all policy files and write the output

```python
def main() -> None:
    """Read all Markdown policies and write one JSON object per line."""
    markdown_files = sorted(SOURCE_DIR.glob("*.md"))
    if not markdown_files:
        raise SystemExit(f"No Markdown policy files found in: {SOURCE_DIR}")

    all_chunks = []
    for source_file in markdown_files:
        all_chunks.extend(make_chunks(source_file))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for chunk in all_chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Created {len(all_chunks)} chunks at {OUTPUT_FILE}")
```

- **`main()`** is the function that coordinates the full preparation process. **`-> None`** means it does not return a value.
- **`SOURCE_DIR.glob("*.md")`** finds Markdown files in the source folder. **`sorted(...)`** puts them in a consistent order. **`markdown_files`** holds their paths.
- **`if not markdown_files`** checks whether the folder had any matching files. **`raise SystemExit(...)`** stops the program with a helpful message if none were found.
- **`all_chunks`** is an empty list for chunks from every document.
- **`for source_file in markdown_files`** handles one file at a time. **`make_chunks(source_file)`** creates its chunk records. **`all_chunks.extend(...)`** adds all of those records to `all_chunks`.
- **`OUTPUT_FILE.parent`** means the folder that should contain the output file. **`mkdir(parents=True, exist_ok=True)`** creates that folder and any missing parent folders; it does not fail if the folder already exists.
- **`with OUTPUT_FILE.open("w", encoding="utf-8") as output`** opens the output file for writing. **`"w"`** means write mode, and **`output`** is the variable used to write to it. The `with` block closes the file automatically afterward.
- **`for chunk in all_chunks`** goes through each chunk record. **`chunk`** is the current dictionary.
- **`json.dumps(chunk, ensure_ascii=False)`** turns the Python dictionary into JSON text. **`ensure_ascii=False`** keeps readable characters instead of converting them to escape codes.
- **`+ "\n"`** adds a newline, so each JSON record is on its own line. This format is called JSONL.
- **`print(...)`** displays how many chunks were written and where the file was saved. **`len(all_chunks)`** counts the records.

**Role in RAG:** this gathers all document chunks and saves them in a format that a later indexing and retrieval step can load.

## 7. Run `main()` when this file is started directly

```python
if __name__ == "__main__":
    main()
```

- **`__name__`** is a Python variable that tells how this file is being used.
- **`"__main__"`** means the file was started directly, for example with `python src/ingest.py`.
- If that condition is true, **`main()`** starts the document processing. If another Python file imports `ingest.py`, `main()` does not automatically run.

**Role in RAG:** this is the entry point that starts ingestion when you run the script.

## Overall flow

1. Find Markdown policy files.
2. Read one document.
3. Find its heading and body sections.
4. Split each section into overlapping chunks.
5. Attach a chunk ID and source details to each piece.
6. Collect chunks from all documents.
7. Save them as `chunks.jsonl` for later RAG steps.

The next RAG stages can load these records, create embeddings, store them in a searchable index, and retrieve relevant chunks for a question. Those stages are not performed by `ingest.py` itself.
