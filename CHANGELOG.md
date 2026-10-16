# Changelog

All notable changes to the **TechNova Solutions RAG System** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
