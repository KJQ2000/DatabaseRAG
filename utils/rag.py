"""
utils/rag.py
------------
Lightweight TF-IDF based Retrieval-Augmented Generation helper.
No external vector database required — uses scikit-learn TfidfVectorizer
with cosine similarity to rank and return the most relevant text chunks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split *text* into overlapping word-level chunks.

    Parameters
    ----------
    text:
        Raw document text.
    chunk_size:
        Approximate number of words per chunk.
    overlap:
        Number of words shared between adjacent chunks.

    Returns
    -------
    List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def _split_by_section(text: str) -> List[str]:
    """Split text on section delimiters (========) and return non-empty sections."""
    sections = re.split(r"={3,}", text)
    return [s.strip() for s in sections if s.strip()]


# ---------------------------------------------------------------------------
# Knowledge loader
# ---------------------------------------------------------------------------

def load_knowledge(filepath: str | Path) -> List[str]:
    """Read a knowledge file and return a list of text chunks.

    For the bilingual README (which uses ====== delimiters to separate table
    sections), each table block becomes one chunk.  For other files the text
    is split into overlapping word-based windows.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the knowledge file.

    Returns
    -------
    List of chunk strings.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {filepath}")

    text = path.read_text(encoding="utf-8", errors="ignore")

    # Use section-based splitting for the bilingual README
    if "======" in text:
        chunks = _split_by_section(text)
    else:
        chunks = _split_into_chunks(text, chunk_size=200, overlap=40)

    return [c for c in chunks if len(c.strip()) > 10]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    filepath: str | Path,
    top_k: int = 5,
) -> str:
    """Return the *top_k* most query-relevant chunks from *filepath* as a
    single concatenated string ready to be injected into an LLM prompt.

    Parameters
    ----------
    query:
        The user's question or restructured question.
    filepath:
        Path to the knowledge file to search.
    top_k:
        Number of chunks to return.

    Returns
    -------
    A newline-separated string of the most relevant text chunks.
    """
    chunks = load_knowledge(filepath)
    if not chunks:
        return ""

    # Combine query + corpus for a joint TF-IDF space
    corpus = [query] + chunks
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # query is index 0; chunks start at index 1
    query_vec = tfidf_matrix[0]
    chunk_vecs = tfidf_matrix[1:]

    scores = cosine_similarity(query_vec, chunk_vecs).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    selected = [chunks[i] for i in top_indices if scores[i] > 0]
    if not selected:
        # Fallback: return the first top_k chunks if nothing scored
        selected = chunks[:top_k]

    return "\n\n---\n\n".join(selected)
