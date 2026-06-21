"""
Structure-aware chunking.

Design decisions, traceable back to the guide:
- Chunk along markdown ## section headers first (guide file 04, section 2:
  "chunking strategy should match document structure, not be a one-size-fits-all
  fixed size"), only falling back to size-based splitting within an oversized section.
- Never split a markdown table mid-row (guide file 04, section 2: "a table
  split mid-row is worse than a slightly-too-large chunk" — directly relevant
  here since hardship_loss_mitigation_guide.md has a real fee table).
- Prepend document title + section header to each chunk before embedding
  (contextual prefixing, guide file 06 section 4) so the chunk's embedding
  carries hierarchical context that raw chunking would otherwise discard.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.metadata_extraction import build_document_metadata, find_tables
from app.config import settings

SECTION_HEADER_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    text: str                  # the contextually-prefixed text that gets embedded
    raw_text: str               # the original chunk text, shown to the user/LLM as source
    metadata: dict = field(default_factory=dict)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Splits document body on '## ' headers. Returns list of (section_title, section_text).
    Content before the first '##' (the header metadata block) is dropped here —
    it's already captured separately via metadata_extraction.
    """
    matches = list(SECTION_HEADER_PATTERN.finditer(text))
    if not matches:
        return [("(no section header)", text)]

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip()))
    return sections


def _split_section_respecting_tables(section_text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Size-based fallback split WITHIN a section, but table spans are treated
    as atomic — never split inside one, even if it pushes a chunk over chunk_size.
    A slightly oversized chunk is the explicit, deliberate trade-off here
    (guide file 04, section 2).
    """
    tables = find_tables(section_text)
    if not tables:
        return _fixed_size_split(section_text, chunk_size, overlap)

    # Build a list of (text_span, is_table) segments, splitting only the
    # non-table segments, and keeping each table whole as its own chunk.
    pieces = []
    cursor = 0
    for t_start, t_end in tables:
        if t_start > cursor:
            pieces.extend(_fixed_size_split(section_text[cursor:t_start], chunk_size, overlap))
        pieces.append(section_text[t_start:t_end])  # table kept atomic
        cursor = t_end
    if cursor < len(section_text):
        pieces.extend(_fixed_size_split(section_text[cursor:], chunk_size, overlap))

    return [p for p in pieces if p.strip()]


def _fixed_size_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Plain fixed-size splitting with overlap, used only as the within-section fallback."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_document(filepath: Path) -> list[Chunk]:
    """
    Full pipeline for one document: extract metadata, split by section,
    split oversized sections (respecting tables), and build contextually-
    prefixed Chunk objects ready for embedding.
    """
    text = filepath.read_text(encoding="utf-8")
    doc_metadata = build_document_metadata(filepath, text)
    sections = _split_into_sections(text)

    chunks = []
    chunk_index = 0
    for section_title, section_text in sections:
        if not section_text.strip():
            continue
        pieces = _split_section_respecting_tables(
            section_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
        )
        for piece in pieces:
            # Contextual prefix — embedded text carries doc title + section,
            # raw_text (shown to the LLM/user) stays clean.
            prefixed = f"{doc_metadata['title']} > {section_title}:\n{piece}"
            chunk_id = f"{filepath.stem}::s{chunk_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=prefixed,
                    raw_text=piece,
                    metadata={
                        **doc_metadata,
                        "section": section_title,
                    },
                )
            )
            chunk_index += 1
    return chunks


def chunk_all_documents(data_dir: Path) -> list[Chunk]:
    all_chunks = []
    for filepath in sorted(data_dir.glob("*.md")):
        all_chunks.extend(chunk_document(filepath))
    return all_chunks
