# 🏗 System Architecture & Design Documentation

## Overview

The **TechNova Solutions RAG System** is an enterprise HR document-grounded question-answering engine engineered with LangChain Expression Language (LCEL), FAISS, and Google Gemini.

---

## 📐 Data Pipeline & Retrieval Flow

```
┌─────────────────────────────────┐
│     technova_policy.txt         │
│ (Raw Organizational Document)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ RecursiveCharacterTextSplitter  │
│  chunk_size: 300, overlap: 50   │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐ ┌───────────────────────────────────┐
│ Document      │ │ GoogleGenerativeAIEmbeddings       │
│ Chunks (1-6)  │ │ (model: gemini-embedding-001)     │
└───────┬───────┘ └─────────────────┬─────────────────┘
        │                           │
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │  FAISS Vector Store Index │
        │  (In-Memory Similarity)   │
        └─────────────┬─────────────┘
                      │
══════════════════════╪═════════════════════════════════
                  QUERY TIME
                      │
               Query Question
                      │
                      ▼
        ┌───────────────────────────┐
        │ FAISS Retriever (top_k=3) │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ Top 3 Retrieved Chunks    │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ Strict Grounding Prompt   │
        │ (Anti-Hallucination Guard)│
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ ChatGoogleGenerativeAI    │
        │ (temperature=0.0)         │
        └─────────────┬─────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │ StrOutputParser           │
        └─────────────┬─────────────┘
                      │
                      ▼
             Final Grounded Answer
```

---

## 🛡️ Anti-Hallucination & Grounding Guardrails

1. **Deterministic Sampling:** `temperature=0.0` prevents stochastic variation, ensuring deterministic and reproducible responses.
2. **Explicit Absence Acknowledgment:** The prompt mandates that if a topic (such as parental leave) is not covered in the retrieved context, the assistant must explicitly declare that the policy document does not provide this information.
3. **Compound Request Decomposition:** When queries ask about multiple policies (e.g., remote work on Wednesday and leave on Friday), each request is separately addressed with notice periods, submission methods, and manager approval requirements.

---

## 📦 Package Layout (`src/` Architecture)

- **`src/technova_rag/engine.py`**: Encapsulates `TechNovaRAG` pipeline lifecycle: ingestion, indexing, retrieval, and LCEL chaining.
- **`src/technova_rag/config.py`**: Model constants, chunk size/overlap parameters, default test questions, and fallback policy text.
- **`src/technova_rag/prompts.py`**: Strict system prompt templates.
- **`rag_engine.py`**: Backward-compatibility shim preserving legacy imports (`from rag_engine import TechNovaRAG`).
- **`main.py`**: Multi-mode CLI entrypoint.
- **`data/`**: Source policy document (`technova_policy.txt`).
- **`tests/`**: Automated unit test suite with 100% offline component verification.
