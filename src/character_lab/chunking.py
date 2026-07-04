from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

from .models import LoreChunk

WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)
SOURCE_RE = re.compile(r"^\s*Source ID:\s*([A-Za-z0-9_\-]+)\s*$", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)\s*$")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "chunk"


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def extract_source_id(text: str, fallback: str) -> str:
    match = SOURCE_RE.search(text)
    return match.group(1).strip() if match else fallback


def split_markdown_sections(text: str) -> List[tuple[str, int, int, str]]:
    lines = text.splitlines()
    sections: List[tuple[str, int, int, str]] = []
    current_title = "Untitled"
    current_start = 1
    current_lines: List[str] = []

    for i, line in enumerate(lines, start=1):
        heading = HEADING_RE.match(line)
        if heading and current_lines:
            sections.append((current_title, current_start, i - 1, "\n".join(current_lines).strip()))
            current_title = heading.group(1).strip()
            current_start = i
            current_lines = [line]
        elif heading and not current_lines:
            current_title = heading.group(1).strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_start, len(lines), "\n".join(current_lines).strip()))
    return [s for s in sections if s[3]]


def split_by_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def chunk_file(path: Path, max_words: int = 220) -> List[LoreChunk]:
    text = path.read_text(encoding="utf-8")
    source_id = extract_source_id(text, fallback=slugify(path.stem))
    output: List[LoreChunk] = []
    counter = 0
    for title, start_line, end_line, section_text in split_markdown_sections(text):
        for part in split_by_words(section_text, max_words=max_words):
            counter += 1
            output.append(
                LoreChunk(
                    id=f"{source_id}:{counter:03d}",
                    source_id=source_id,
                    source_file=str(path),
                    title=title,
                    text=part,
                    start_line=start_line,
                    end_line=end_line,
                    metadata={"word_count": word_count(part)},
                )
            )
    return output


def chunk_lore_dir(lore_dir: Path, max_words: int = 220) -> List[LoreChunk]:
    chunks: List[LoreChunk] = []
    for path in sorted(lore_dir.glob("*.md")):
        chunks.extend(chunk_file(path, max_words=max_words))
    return chunks


def chunk_to_dict(chunk: LoreChunk) -> dict:
    return {
        "id": chunk.id,
        "source_id": chunk.source_id,
        "source_file": chunk.source_file,
        "title": chunk.title,
        "text": chunk.text,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "metadata": chunk.metadata,
    }


def chunk_from_dict(obj: dict) -> LoreChunk:
    return LoreChunk(
        id=obj["id"],
        source_id=obj["source_id"],
        source_file=obj.get("source_file", ""),
        title=obj.get("title", ""),
        text=obj["text"],
        start_line=int(obj.get("start_line", 0)),
        end_line=int(obj.get("end_line", 0)),
        metadata=obj.get("metadata", {}),
    )


def write_chunks(chunks: Iterable[LoreChunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> List[LoreChunk]:
    chunks: List[LoreChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(chunk_from_dict(json.loads(line)))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk markdown lore files into JSONL passages.")
    parser.add_argument("--lore-dir", type=Path, default=Path("data/lore"))
    parser.add_argument("--out", type=Path, default=Path("data/index/chunks.jsonl"))
    parser.add_argument("--max-words", type=int, default=220)
    args = parser.parse_args()

    chunks = chunk_lore_dir(args.lore_dir, max_words=args.max_words)
    write_chunks(chunks, args.out)
    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
