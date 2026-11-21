"""Turn Markdown policy files into searchable, source-aware chunks."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "source_documents"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
MAX_CHARS = 700
OVERLAP_CHARS = 100


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


if __name__ == "__main__":
    main()
