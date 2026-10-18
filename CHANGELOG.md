# Changelog

All notable changes to the **TechNova Solutions RAG System** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-18

### Added
- **Modular Package Structure (`src/technova_rag`):** Re-architected pipeline into a clean package layout featuring `engine.py`, `config.py`, and `prompts.py`.
- **Standard Packaging Configuration (`pyproject.toml`):** Added modern PEP 517/621 packaging metadata and tool configurations.
- **Dedicated Data & Docs Directories:** Created `data/` for source policy documents and `docs/` for architecture and benchmark results.
- **Backward-Compatible Root Facade (`rag_engine.py`):** Root-level facade re-exporting `src.technova_rag` modules to preserve legacy imports and CLI commands.
- **Enhanced Test Discovery (`tests/__init__.py`):** Added package marker and automated compatibility unit test.

### Changed
- Standardized environment variable fallback for Gemini chat model (`GEMINI_CHAT_MODEL`).
- Updated `main.py` to prioritize `src/technova_rag` with fallback to `rag_engine`.
- Enhanced `README.md` repository structure and architectural overview.

## [1.0.0] - 2026-10-17

### Added
- **Core RAG Architecture:** Implementation of LangChain Expression Language (LCEL) chain `prompt | llm | StrOutputParser()`.
- **Vector Retrieval:** Integrated FAISS CPU vector index with `gemini-embedding-001` embeddings.
- **Strict Anti-Hallucination Guardrails:** Prompt engineering enforcing strict context grounding, prohibiting extrapolation, and requiring explicit absence declarations for missing policies.
- **Dual Document Ingestion:** Supports loading from external `technova_policy.txt` as well as an in-code fallback document to prevent runtime failures.
- **Multi-Modal CLI:** Interactive terminal assistant (`--interactive`), benchmark evaluator (`--eval`), and single-shot query runner (`--query`).
- **Assignment Submission Script:** Self-contained script `24f2001089.py` complete with verified benchmark evaluation outputs.
- **Security Protocols:** Isolated environment credentials via `.gitignore` and template `.env.example`.
- **Test Suite:** Unit tests in `tests/test_rag.py` covering chunking, retriever configuration, and prompt template invariants.
- **CI/CD:** Automated GitHub Actions pipeline verifying Python 3.11 tests and secret isolation.
