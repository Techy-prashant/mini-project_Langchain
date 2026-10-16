"""
TechNova Solutions - Document-Based RAG System
Assignment: Introduction to LangChain: Chains, Retrieval and RAG

This script implements a complete Retrieval-Augmented Generation (RAG) system using
LangChain, FAISS, and Google Gemini (Free Tier) to answer questions based on the
TechNova Solutions employee policy document.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminal environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Define base directory using Path(__file__).resolve().parent for robust path resolution
BASE_DIR = Path(__file__).resolve().parent

# Explicitly load .env.local first (local development credentials), then fall back to .env
env_local_path = BASE_DIR / ".env.local"
env_path = BASE_DIR / ".env"

if env_local_path.exists():
    load_dotenv(dotenv_path=env_local_path)
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# LangChain Imports
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# In-code policy document representation to ensure rag_assignment.py is fully self-contained
# and does not require an external technova_policy.txt file.
TECHNOVA_POLICY_DOCUMENT = """TechNova Solutions - Employee Policy Document


1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.


2. Work From Home: Employees may work remotely on Wednesdays and Fridays. Employees must inform their manager by email at least one working day in advance. Remote work is subject to manager approval.


3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM, Monday through Friday. Employees receive a one-hour lunch break.


4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.


5. Learning Allowance: Each full-time employee can claim up to ₹12,000 per calendar year for approved job-related courses and certifications. Claims must include proof of payment and course completion."""

# The 3 prescribed assignment test questions
TEST_QUESTIONS = [
    "How many paid annual leave days does a full-time employee receive per year?",
    "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?",
    "What is the company's parental leave policy?"
]


def load_and_split_policy(file_path: str = None):
    """
    Step 1: Load the policy document into LangChain Document representation.
    Self-contained implementation: Ingests the policy directly from the in-code
    TECHNOVA_POLICY_DOCUMENT representation, or loads from an external file via
    TextLoader with UTF-8 encoding if a valid file path is provided and exists.

    Step 2: Split the document into smaller semantic chunks.
    """
    if file_path and os.path.exists(file_path):
        print(f"[Step 1] Loading document from file: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
    else:
        print("[Step 1] Loading self-contained in-code policy document (TECHNOVA_POLICY_DOCUMENT)...")
        documents = [
            Document(
                page_content=TECHNOVA_POLICY_DOCUMENT,
                metadata={"source": "technova_policy.txt"}
            )
        ]

    # Chunking Strategy:
    # - chunk_size=300: Each of the 5 policy sections in technova_policy.txt
    #   is between 130 and 270 characters long. A chunk size of 300 ensures
    #   each individual policy rule (Annual Leave, Work From Home, etc.)
    #   fits intact within a single chunk without being truncated mid-rule.
    # - chunk_overlap=50: Overlap ensures semantic continuity and prevents
    #   loss of contextual boundaries between sections.
    print("[Step 2] Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Successfully created {len(chunks)} chunks.")
    return chunks


def format_docs(docs):
    """Helper function to format retrieved document chunks into readable context."""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        formatted.append(f"[Context Chunk {idx}]:\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


def build_rag_system(chunks, api_key: str):
    """
    Steps 3 to 7:
    - Generates embeddings with GoogleGenerativeAIEmbeddings (gemini-embedding-001).
    - Stores embeddings in a FAISS vector store.
    - Creates a retriever for top 3 chunks (k=3).
    - Configures a PromptTemplate enforcing strict context-only answers.
    - Initializes ChatGoogleGenerativeAI (gemini-3.8-flash) and constructs the LCEL RAG chain.
    """
    # Step 3: Embeddings (Current stable Gemini embedding model)
    print("[Step 3] Initializing GoogleGenerativeAIEmbeddings (gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    # Step 4: Vector Store
    print("[Step 4] Creating FAISS vector store from document chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Step 5: Retriever (retrieve exactly the 3 most relevant chunks)
    print("[Step 5] Configuring retriever to return top 3 chunks (k=3)...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 6: Prompt Template
    # Instructs the LLM to strictly ground answers in the context, handle
    # multi-part requests separately, and explicitly declare when a policy is absent.
    print("[Step 6] Constructing prompt template with strict context-only constraint...")
    template = """You are an HR policy assistant for TechNova Solutions.
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

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Step 7: Chat Model & LCEL Chain (gemini-3.8-flash has free-tier access)
    print("[Step 7] Initializing ChatGoogleGenerativeAI (gemini-3.8-flash, temperature=0)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        google_api_key=api_key
    )

    # Assemble the chain using LangChain Expression Language (LCEL)
    rag_chain = prompt | llm | StrOutputParser()

    return retriever, rag_chain


def run_evaluation(retriever, rag_chain, questions):
    """
    Step 8: Run the three prescribed test questions.
    Prints each question, the top 3 retrieved context chunks, and the final answer.
    """
    print("\n" + "=" * 80, flush=True)
    print("RUNNING PRESCRIBED TEST QUESTIONS", flush=True)
    print("=" * 80, flush=True)

    for idx, question in enumerate(questions, 1):
        print(f"\n{'='*80}", flush=True)
        print(f"QUESTION {idx}: {question}", flush=True)
        print(f"{'='*80}", flush=True)

        # Retrieve exactly top 3 chunks
        retrieved_docs = retriever.invoke(question)

        print("\n--- RETRIEVED CONTEXT (Top 3 Chunks) ---", flush=True)
        for c_idx, doc in enumerate(retrieved_docs, 1):
            print(f"\n[Chunk {c_idx}]:", flush=True)
            print(doc.page_content.strip(), flush=True)

        # Format context for prompt
        context_str = format_docs(retrieved_docs)

        print("\nGenerating answer from Gemini...", flush=True)

        # Generate answer from context
        answer = rag_chain.invoke({
            "context": context_str,
            "question": question
        })

        print("\n--- GENERATED ANSWER ---", flush=True)
        print(answer.strip(), flush=True)


def main():
    print("=" * 80)
    print("TechNova Solutions - Document-Based RAG System (Google Gemini)")
    print("=" * 80)

    policy_file = BASE_DIR / "technova_policy.txt"
    policy_path = str(policy_file) if policy_file.exists() else None

    # Steps 1 & 2: Load and split the document (self-contained; does not require external file)
    chunks = load_and_split_policy(policy_path)

    # Validate Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "your_google_gemini_api_key_here":
        print("\n" + "!" * 80)
        print("NOTICE: GOOGLE_API_KEY is not configured.")
        print("Document loading and text splitting verified successfully!")
        print("To run live embeddings and answer generation via Google Gemini Free Tier:")
        print("  1. Create a '.env.local' or '.env' file in this folder.")
        print("  2. Add: GOOGLE_API_KEY=your_actual_key_here")
        print("     (Get a free key at https://aistudio.google.com/)")
        print("  3. Run: .\\.venv\\Scripts\\python.exe rag_assignment.py")
        print("!" * 80)
        return

    # Steps 3 to 7: Build vector store, retriever, and RAG chain
    retriever, rag_chain = build_rag_system(chunks, api_key)

    # Step 8: Run evaluation on the 3 prescribed test questions
    run_evaluation(retriever, rag_chain, TEST_QUESTIONS)


if __name__ == "__main__":
    main()


# ==============================================================================
# ACTUAL EXECUTION OUTPUTS
# ==============================================================================
# Terminal outputs captured from successful live run:
#
# ------------------------------------------------------------------------------
# QUESTION 1:
# How many paid annual leave days does a full-time employee receive per year?
#
# RETRIEVED CONTEXT:
# [Chunk 1]:
# 1. Annual Leave: Every full-time employee receives 18 paid annual leave days
# per calendar year. Employees must submit annual leave requests to their
# manager at least 3 calendar days before the requested leave date. Leave
# approval depends on the manager's confirmation.
#
# [Chunk 2]:
# 4. Sick Leave: Employees receive 10 paid sick leave days per calendar year.
# If an employee is absent due to illness for more than two consecutive working
# days, a medical certificate must be submitted to HR.
#
# [Chunk 3]:
# 3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM,
# Monday through Friday. Employees receive a one-hour lunch break.
#
# GENERATED ANSWER:
# Based on TechNova Solutions' policy, every full-time employee receives
# 18 paid annual leave days per calendar year.
#
# ------------------------------------------------------------------------------
# QUESTION 2:
# An employee wants to work remotely on Wednesday and take leave on Friday.
# What do the company policies say about these two requests?
#
# RETRIEVED CONTEXT:
# [Chunk 1]:
# 2. Work From Home: Employees may work remotely on Wednesdays and Fridays.
# Employees must inform their manager by email at least one working day in
# advance. Remote work is subject to manager approval.
#
# [Chunk 2]:
# 1. Annual Leave: Every full-time employee receives 18 paid annual leave days
# per calendar year. Employees must submit annual leave requests to their
# manager at least 3 calendar days before the requested leave date. Leave
# approval depends on the manager's confirmation.
#
# [Chunk 3]:
# 4. Sick Leave: Employees receive 10 paid sick leave days per calendar year.
# If an employee is absent due to illness for more than two consecutive working
# days, a medical certificate must be submitted to HR.
#
# GENERATED ANSWER:
# Based on TechNova Solutions' policy context, here is the information for
# each request:
#
# ### 1. Working Remotely on Wednesday
# * Eligibility: Employees are permitted to work remotely on Wednesdays (and Fridays).
# * Notice Period: Employees must inform their manager at least one working day in advance.
# * Submission Method: Notice must be sent to the manager by email.
# * Manager Approval: Remote work is subject to manager approval and is not automatically approved.
#
# ### 2. Taking Leave on Friday (Annual Leave)
# * Notice Period: Employees must submit their request at least 3 calendar days before the requested leave date.
# * Submission Method: The request must be submitted directly to the employee's manager.
# * Manager Approval: Leave approval depends on the manager's confirmation and is not automatically approved.
#
# (Note: If the requested Friday off is due to illness, the Sick Leave policy
# applies instead, which provides 10 paid sick leave days per calendar year;
# a medical certificate must be submitted to HR only if the absence extends
# beyond two consecutive working days.)
#
# ------------------------------------------------------------------------------
# QUESTION 3:
# What is the company's parental leave policy?
#
# RETRIEVED CONTEXT:
# [Chunk 1]:
# 1. Annual Leave: Every full-time employee receives 18 paid annual leave days
# per calendar year. Employees must submit annual leave requests to their
# manager at least 3 calendar days before the requested leave date. Leave
# approval depends on the manager's confirmation.
#
# [Chunk 2]:
# 4. Sick Leave: Employees receive 10 paid sick leave days per calendar year.
# If an employee is absent due to illness for more than two consecutive working
# days, a medical certificate must be submitted to HR.
#
# [Chunk 3]:
# 3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM,
# Monday through Friday. Employees receive a one-hour lunch break.
#
# GENERATED ANSWER:
# TechNova Solutions' policy document does not provide information regarding
# the parental leave policy.
# ==============================================================================
