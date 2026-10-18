"""
TechNova Solutions - Modular Retrieval-Augmented Generation (RAG) Engine
Backward-compatible root entrypoint exposing the modular src/technova_rag package.
"""

import sys
from pathlib import Path

# Ensure 'src' is accessible on the Python module search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from technova_rag.engine import TechNovaRAG
from technova_rag.prompts import STRICT_RAG_PROMPT_TEMPLATE
from technova_rag.config import (
    DEFAULT_TEST_QUESTIONS,
    DEFAULT_POLICY_DOCUMENT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHAT_MODEL,
    QueryResult,
)

__all__ = [
    "TechNovaRAG",
    "STRICT_RAG_PROMPT_TEMPLATE",
    "DEFAULT_TEST_QUESTIONS",
    "DEFAULT_POLICY_DOCUMENT",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_TOP_K",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_CHAT_MODEL",
    "QueryResult",
]
