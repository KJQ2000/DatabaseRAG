"""
tests/test_rag.py
-----------------
Smoke tests for utils/rag.py — ensures the RAG helper works correctly
against the knowledge files shipped with the project.

Run with:
    python -m pytest tests/test_rag.py -v
"""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.rag import load_knowledge, retrieve_relevant_chunks

BILINGUAL_README = Path(__file__).resolve().parent.parent / "knowledge" / "Bilingual README.txt"
KNOWLEDGE_TXT    = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge.txt"


# ---------------------------------------------------------------------------
# load_knowledge tests
# ---------------------------------------------------------------------------

class TestLoadKnowledge:
    def test_loads_bilingual_readme(self):
        chunks = load_knowledge(BILINGUAL_README)
        assert isinstance(chunks, list), "Should return a list"
        assert len(chunks) > 0, "Should return at least one chunk"

    def test_bilingual_readme_sections(self):
        """Each section should be non-empty and > 10 chars."""
        chunks = load_knowledge(BILINGUAL_README)
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk.strip()) > 10

    def test_loads_knowledge_txt(self):
        chunks = load_knowledge(KNOWLEDGE_TXT)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_knowledge("non_existent_file.txt")


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks tests
# ---------------------------------------------------------------------------

class TestRetrieveRelevantChunks:
    def test_returns_string(self):
        result = retrieve_relevant_chunks("stock items", BILINGUAL_README)
        assert isinstance(result, str)

    def test_returns_non_empty_for_valid_query(self):
        result = retrieve_relevant_chunks("customer name email", BILINGUAL_README)
        assert len(result.strip()) > 0

    def test_top_k_respected(self):
        """Result should not exceed top_k sections."""
        result = retrieve_relevant_chunks("booking payment", BILINGUAL_README, top_k=2)
        sections = result.split("\n\n---\n\n")
        assert len(sections) <= 2

    def test_knowledge_txt_retrieval(self):
        result = retrieve_relevant_chunks("refund policy", KNOWLEDGE_TXT)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_irrelevant_query_still_returns_fallback(self):
        """Even a nonsense query should return fallback chunks (not empty)."""
        result = retrieve_relevant_chunks("xyzzy quux randomtoken", BILINGUAL_README)
        assert isinstance(result, str)
        # Fallback returns first top_k chunks regardless
        assert len(result) > 0
