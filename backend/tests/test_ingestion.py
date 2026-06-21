"""
Unit tests for the deterministic, non-API-dependent parts of the pipeline:
metadata extraction and chunking. These run without Ollama running, which
matters for CI — you don't want every test run to cost money or require
secrets just to verify chunking logic.

Run with: pytest tests/
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.metadata_extraction import extract_header_metadata, extract_title, find_tables
from app.ingestion.chunking import _split_into_sections, _split_section_respecting_tables


SAMPLE_DOC = """# Test Policy Document

Document Type: Internal Policy
Effective Date: 2025-01-01
Version: 3.2
Status: Current

## Section One

Some content here about the first topic. This section is short.

## Section Two

| Col A | Col B |
|---|---|
| 1 | 2 |
| 3 | 4 |

Some text after the table.
"""

SUPERSEDED_DOC = """# Old Policy

Document Type: Internal Policy
Status: SUPERSEDED — replaced by newer version

## Section One

Old content.
"""


def test_extract_header_metadata():
    metadata = extract_header_metadata(SAMPLE_DOC)
    assert metadata["document_type"] == "Internal Policy"
    assert metadata["version"] == "3.2"
    assert metadata["is_current"] is True


def test_extract_header_metadata_superseded_flagged_not_current():
    metadata = extract_header_metadata(SUPERSEDED_DOC)
    assert metadata["is_current"] is False


def test_extract_title():
    assert extract_title(SAMPLE_DOC) == "Test Policy Document"


def test_extract_title_missing_returns_none():
    assert extract_title("no heading here") is None


def test_find_tables_detects_table():
    tables = find_tables(SAMPLE_DOC)
    assert len(tables) == 1
    start, end = tables[0]
    table_text = SAMPLE_DOC[start:end]
    assert "Col A" in table_text
    assert "Col B" in table_text


def test_split_into_sections():
    sections = _split_into_sections(SAMPLE_DOC)
    titles = [s[0] for s in sections]
    assert "Section One" in titles
    assert "Section Two" in titles


def test_table_never_split_mid_row():
    """
    The core correctness property from guide file 04 section 2:
    a table must never be split across chunk boundaries, even if that
    means producing an oversized chunk.
    """
    sections = _split_into_sections(SAMPLE_DOC)
    section_two_text = dict(sections)["Section Two"]

    # Use a deliberately tiny chunk_size to force splitting behavior,
    # and confirm the table still comes out as one atomic piece.
    pieces = _split_section_respecting_tables(section_two_text, chunk_size=20, overlap=5)

    table_pieces = [p for p in pieces if "Col A" in p]
    assert len(table_pieces) == 1, "Table should appear whole in exactly one chunk, never split"
    assert "Col A" in table_pieces[0] and "Col B" in table_pieces[0]
    assert "1" in table_pieces[0] and "4" in table_pieces[0], "All table rows must be in the same chunk"
