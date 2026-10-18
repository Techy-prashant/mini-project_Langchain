"""
TechNova Solutions - Document-Based RAG System Package
Enterprise Policy Intelligence powered by LangChain LCEL, FAISS, and Google Gemini.
"""

from .engine import TechNovaRAG
from .prompts import STRICT_RAG_PROMPT_TEMPLATE
from .config import (
    DEFAULT_TEST_QUESTIONS,
    DEFAULT_POLICY_DOCUMENT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHAT_MODEL,
    QueryResult,
)

__version__ = "1.1.0"
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
    "__version__",
]
