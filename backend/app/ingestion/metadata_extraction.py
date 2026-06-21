"""
Deterministic metadata extraction from document headers.

Design decision (see guide file 06, "Feature Engineering for AI"): use the
cheapest reliable method per field. Our sample documents have a structured
header block (Document Type, Effective Date, Version, etc.) written in a
consistent "Key: Value" format, so regex extraction is the right tool here —
no LLM call needed, no latency cost, fully deterministic and testable.

If source documents did NOT have this structure (e.g., raw scanned PDFs,
free-form emails), this is exactly where you'd swap in LLM-based extraction
instead — the interface below (extract_metadata returning a dict) would stay
the same regardless of which extraction method backs it.
"""

import re
from pathlib import Path
from typing import Optional

HEADER_FIELD_PATTERN = re.compile(
    r"^(Document Type|Effective Date|Version|Region|Supersedes|Status|Owner):\s*(.+)$",
    re.MULTILINE,
)

# Detects a markdown table block so we can flag chunks containing one —
# used by the chunker to avoid splitting tables mid-row (guide file 04, section 2).
TABLE_PATTERN = re.compile(r"(\|.+\|\n\|[-:\s|]+\|\n(?:\|.+\|\n?)+)")


def extract_header_metadata(text: str) -> dict:
    """
    Parses the 'Key: Value' header block present in our sample policy docs.
    Returns a flat dict of normalized metadata fields.
    """
    metadata = {}
    for match in HEADER_FIELD_PATTERN.finditer(text):
        key, value = match.group(1), match.group(2).strip()
        normalized_key = key.lower().replace(" ", "_")
        metadata[normalized_key] = value

    # Derive a clean boolean flag for filtering — this is the concrete fix
    # for the "superseded document gets retrieved as if current" failure
    # mode described in guide file 06, section 2.
    status = metadata.get("status", "")
    metadata["is_current"] = "superseded" not in status.lower()

    return metadata


def extract_title(text: str) -> Optional[str]:
    """First markdown H1 heading, used as the document title."""
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def find_tables(text: str) -> list[tuple[int, int]]:
    """
    Returns (start_char, end_char) spans of every markdown table in the text.
    The chunker uses these spans to avoid ever cutting inside one.
    """
    return [(m.start(), m.end()) for m in TABLE_PATTERN.finditer(text)]


def build_document_metadata(filepath: Path, text: str) -> dict:
    """
    Full metadata record for one source document — combines header fields,
    title, and provenance (file path), which together support both the
    pre-filtering pattern (guide file 03, section 4) and citation/traceability
    (guide file 07, output guardrails).
    """
    metadata = extract_header_metadata(text)
    metadata["title"] = extract_title(text) or filepath.stem
    metadata["source_file"] = filepath.name
    return metadata
