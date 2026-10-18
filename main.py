#!/usr/bin/env python3
"""
TechNova Solutions - Interactive RAG Policy Assistant & Benchmark Evaluator
CLI entrypoint supporting evaluation benchmarks, interactive terminal chat,
and single-shot question answering.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent

# Load local environment files (.env.local has priority over .env)
env_local = BASE_DIR / ".env.local"
env_default = BASE_DIR / ".env"
if env_local.exists():
    load_dotenv(dotenv_path=env_local)
elif env_default.exists():
    load_dotenv(dotenv_path=env_default)

SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from technova_rag import TechNovaRAG, DEFAULT_TEST_QUESTIONS
except ImportError:
    from rag_engine import TechNovaRAG, DEFAULT_TEST_QUESTIONS


BANNER = r"""
================================================================================
   ______           __    _   __                 ____  ___   ______
  /_  __/__  _____ / /_  / | / /___ _   ______ _/ __ \/   | / ____/
   / / / _ \/ ___// __ \/  |/ / __ \ | / / __ `/ /_/ / /| |/ / __  
  / / /  __/ /__ / / / / /|  / /_/ / |/ / /_/ / _, _/ ___ / /_/ /  
 /_/  \___/\___//_/ /_/_/ |_/\____/|___/\__,_/_/ |_/_/  |_\____/   
                                                                   
   TechNova Solutions • Policy Intelligence System (LangChain + Gemini)
================================================================================
"""


def run_benchmark(rag: TechNovaRAG, questions=None):
    """Executes the standard prescribed test benchmark."""
    questions = questions or DEFAULT_TEST_QUESTIONS
    print("\n" + "=" * 80)
    print("  RUNNING PRESCRIBED EVALUATION BENCHMARK")
    print("=" * 80)

    for idx, q in enumerate(questions, 1):
        print(f"\n{'-' * 80}")
        print(f"QUESTION {idx}: {q}")
        print(f"{'-' * 80}")

        result = rag.query(q)

        print("\n--- RETRIEVED CONTEXT (Top 3 Chunks) ---")
        for c_idx, doc in enumerate(result["retrieved_chunks"], 1):
            print(f"\n[Chunk {c_idx}]:")
            print(doc.page_content.strip())

        print("\n--- GENERATED ANSWER ---")
        print(result["answer"])


def run_interactive(rag: TechNovaRAG):
    """Starts an interactive Q&A loop with the policy assistant."""
    print("\n[Interactive Mode Enabled]")
    print("Type your questions about TechNova policies below. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("\n[Policy Question] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting Policy Assistant. Goodbye!")
                break

            result = rag.query(user_input)
            print("\n[Retrieved Chunks]:")
            for idx, doc in enumerate(result["retrieved_chunks"], 1):
                preview = doc.page_content.strip().replace("\n", " ")
                print(f"  ({idx}) {preview[:100]}...")

            print("\n[Assistant Answer]:")
            print(result["answer"])

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TechNova Solutions RAG Assistant & Benchmark CLI"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the evaluation benchmark on the 3 prescribed test questions"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive chat mode"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Ask a single question and print the grounded answer"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=str(BASE_DIR / "technova_policy.txt"),
        help="Path to external policy text file"
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="TechNova RAG Policy Assistant v1.0.0",
        help="Show program version and exit"
    )

    args = parser.parse_args()

    print(BANNER)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "your_google_gemini_api_key_here":
        print("!" * 80)
        print("NOTICE: GOOGLE_API_KEY is not configured in your environment.")
        print("To run live query retrieval and Gemini answer generation:")
        print("  1. Copy .env.example to .env (or .env.local)")
        print("  2. Add your free Google Gemini API key: GOOGLE_API_KEY=your_key")
        print("  3. Re-run: python main.py --eval")
        print("!" * 80 + "\n")
        sys.exit(0)

    rag = TechNovaRAG(
        api_key=api_key,
        policy_file=args.policy if os.path.exists(args.policy) else None
    )

    print("Initializing LangChain RAG pipeline (FAISS + Google Gemini)...")
    rag.initialize_pipeline()
    print("Pipeline ready!\n")

    if args.query:
        result = rag.query(args.query)
        print(f"QUESTION: {args.query}\n")
        print(f"ANSWER:\n{result['answer']}")
    elif args.interactive:
        run_interactive(rag)
    else:
        # Default behavior is to run the benchmark evaluation
        run_benchmark(rag)


if __name__ == "__main__":
    main()
