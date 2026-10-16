"""
TechNova Solutions - Modular Retrieval-Augmented Generation (RAG) Engine
Core module encapsulating document ingestion, text splitting, vector indexing,
and strict context-grounded response generation via LangChain and Google Gemini.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# LangChain Core and Community Integrations
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Default in-code policy document ensuring zero external dependency failure
DEFAULT_POLICY_DOCUMENT = """TechNova Solutions - Employee Policy Document


1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.


2. Work From Home: Employees may work remotely on Wednesdays and Fridays. Employees must inform their manager by email at least one working day in advance. Remote work is subject to manager approval.


3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM, Monday through Friday. Employees receive a one-hour lunch break.


4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.


5. Learning Allowance: Each full-time employee can claim up to ₹12,000 per calendar year for approved job-related courses and certifications. Claims must include proof of payment and course completion."""

DEFAULT_TEST_QUESTIONS = [
    "How many paid annual leave days does a full-time employee receive per year?",
    "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?",
    "What is the company's parental leave policy?"
]

STRICT_RAG_PROMPT_TEMPLATE = """You are an HR policy assistant for TechNova Solutions.
Answer the user's question based ONLY on the following policy context.

Strict Guidelines:
1. Base your answer strictly on the provided context. Never invent, assume, or extrapolate any policies.
2. If the context does not specify a policy (such as for parental leave), explicitly state that TechNova Solutions' policy document does not provide that information.
3. When answering questions with multiple requests (such as remote work and leave), address each request separately, detailing its specific notice period, submission method, and manager approval requirement. Do not imply approval is automatic.

Context:
{context}

Question:
{question}

Answer:"""


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
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 3,
        embedding_model: str = "gemini-embedding-001",
        chat_model: str = "gemini-1.5-flash",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.policy_file = policy_file

        # Resolve API key from arguments or environment
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        self.chunks: List[Document] = []
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None

    def load_and_split(self) -> List[Document]:
        """Loads policy text (from file if found, otherwise built-in) and splits into semantic chunks."""
        if self.policy_file and os.path.exists(self.policy_file):
            loader = TextLoader(self.policy_file, encoding="utf-8")
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

    def query(self, question: str) -> Dict[str, Any]:
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
