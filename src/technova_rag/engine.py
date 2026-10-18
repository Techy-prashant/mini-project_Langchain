"""
TechNova Solutions - Modular Retrieval-Augmented Generation (RAG) Engine
Core module encapsulating document ingestion, text splitting, vector indexing,
and strict context-grounded response generation via LangChain and Google Gemini.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain Core and Community Integrations
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_POLICY_DOCUMENT,
    QueryResult,
)
from .prompts import STRICT_RAG_PROMPT_TEMPLATE


class TechNovaRAG:
    """
    Retrieval-Augmented Generation (RAG) pipeline for TechNova Solutions HR policy.

    Attributes:
        chunk_size (int): Max character length per chunk.
        chunk_overlap (int): Overlap length to preserve cross-chunk context.
        top_k (int): Number of semantic chunks retrieved per query.
        embedding_model (str): Gemini embeddings model identifier.
        chat_model (str): Gemini chat model identifier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        policy_file: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.chat_model = os.getenv("GEMINI_CHAT_MODEL") or chat_model or DEFAULT_CHAT_MODEL
        self.policy_file = policy_file

        # Resolve API key from arguments or environment
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        self.chunks: List[Document] = []
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None

    def _resolve_policy_file_path(self) -> Optional[Path]:
        """Resolves the policy file path, checking explicit path, data directory, and project root."""
        if self.policy_file:
            explicit_path = Path(self.policy_file)
            if explicit_path.exists():
                return explicit_path

        # Search standard project locations relative to this module
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            repo_root / "data" / "technova_policy.txt",
            repo_root / "technova_policy.txt",
            Path.cwd() / "data" / "technova_policy.txt",
            Path.cwd() / "technova_policy.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_and_split(self) -> List[Document]:
        """Loads policy text (from file if found, otherwise built-in) and splits into semantic chunks."""
        resolved_file = self._resolve_policy_file_path()

        if resolved_file:
            loader = TextLoader(str(resolved_file), encoding="utf-8")
            documents = loader.load()
        else:
            documents = [
                Document(
                    page_content=DEFAULT_POLICY_DOCUMENT,
                    metadata={"source": "technova_policy.txt"}
                )
            ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = text_splitter.split_documents(documents)
        return self.chunks

    def initialize_pipeline(self) -> None:
        """Initializes embeddings, builds FAISS vector store index, and constructs the LCEL RAG chain."""
        if not self.chunks:
            self.load_and_split()

        if not self.api_key or self.api_key.strip() == "your_google_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is not configured. Please set GOOGLE_API_KEY in your .env or .env.local file."
            )

        # 1. Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.api_key
        )

        # 2. FAISS Vector Store Index
        self.vectorstore = FAISS.from_documents(self.chunks, embeddings)

        # 3. Retriever (top-k)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # 4. Strict Grounding Prompt Template
        prompt = PromptTemplate(
            template=STRICT_RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # 5. Gemini Chat Model with temperature=0 for factual consistency
        llm = ChatGoogleGenerativeAI(
            model=self.chat_model,
            temperature=0,
            google_api_key=self.api_key
        )

        # 6. Assemble LCEL Chain: prompt | llm | output_parser
        self.rag_chain = prompt | llm | StrOutputParser()

    @staticmethod
    def format_context_chunks(docs: List[Document]) -> str:
        """Formats retrieved document chunks into clean readable context blocks."""
        formatted = []
        for idx, doc in enumerate(docs, 1):
            formatted.append(f"[Context Chunk {idx}]:\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    def get_chunk_count(self) -> int:
        """Returns the number of generated document chunks."""
        if not self.chunks:
            self.load_and_split()
        return len(self.chunks)

    def query(self, question: str) -> QueryResult:
        """
        Executes a RAG query: retrieves top-k chunks, generates a grounded response,
        and returns context alongside the answer.
        """
        if not self.retriever or not self.rag_chain:
            self.initialize_pipeline()

        retrieved_docs = self.retriever.invoke(question)
        context_str = self.format_context_chunks(retrieved_docs)

        answer = self.rag_chain.invoke({
            "context": context_str,
            "question": question
        })

        return {
            "question": question,
            "retrieved_chunks": retrieved_docs,
            "context_str": context_str,
            "answer": answer.strip()
        }
