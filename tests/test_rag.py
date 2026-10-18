"""
Unit tests for TechNova Solutions RAG pipeline components.
Tests text loading, chunking strategy, prompt formatting, and configuration
without requiring live network API calls.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root and src directory are accessible on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag_engine import TechNovaRAG, STRICT_RAG_PROMPT_TEMPLATE, DEFAULT_TEST_QUESTIONS
from technova_rag import TechNovaRAG as PackageTechNovaRAG


class TestTechNovaRAG(unittest.TestCase):
    def setUp(self):
        self.rag = TechNovaRAG(
            api_key="mock_test_key_not_real",
            chunk_size=300,
            chunk_overlap=50,
            top_k=3
        )

    def test_load_and_split_policy(self):
        """Verify that document loading and semantic chunking produces expected chunks."""
        chunks = self.rag.load_and_split()
        self.assertGreater(len(chunks), 0)
        # Should be between 4 and 8 chunks with chunk_size 300
        self.assertGreaterEqual(len(chunks), 5)
        self.assertLessEqual(len(chunks), 10)

        # Ensure no chunk exceeds the chunk_size significantly
        for chunk in chunks:
            self.assertIsNotNone(chunk.page_content)
            self.assertGreater(len(chunk.page_content), 10)

    def test_format_context_chunks(self):
        """Verify helper formatting structure for retrieved context."""
        chunks = self.rag.load_and_split()
        formatted = self.rag.format_context_chunks(chunks[:3])
        self.assertIn("[Context Chunk 1]:", formatted)
        self.assertIn("[Context Chunk 2]:", formatted)
        self.assertIn("[Context Chunk 3]:", formatted)

    def test_prompt_template_guidelines(self):
        """Verify strict prompt template guidelines for anti-hallucination."""
        self.assertIn("Never invent, assume, or extrapolate", STRICT_RAG_PROMPT_TEMPLATE)
        self.assertIn("explicitly state that TechNova Solutions' policy document does not provide", STRICT_RAG_PROMPT_TEMPLATE)
        self.assertIn("{context}", STRICT_RAG_PROMPT_TEMPLATE)
        self.assertIn("{question}", STRICT_RAG_PROMPT_TEMPLATE)

    def test_default_test_questions(self):
        """Verify prescribed test questions count and contents."""
        self.assertEqual(len(DEFAULT_TEST_QUESTIONS), 3)
        self.assertTrue(any("annual leave" in q.lower() for q in DEFAULT_TEST_QUESTIONS))
        self.assertTrue(any("remotely on wednesday" in q.lower() for q in DEFAULT_TEST_QUESTIONS))
        self.assertTrue(any("parental leave" in q.lower() for q in DEFAULT_TEST_QUESTIONS))


    def test_missing_api_key_raises_value_error(self):
        """Verify that an invalid or placeholder API key raises a clear ValueError."""
        rag_invalid = TechNovaRAG(api_key="your_google_gemini_api_key_here")
        with self.assertRaises(ValueError) as ctx:
            rag_invalid.initialize_pipeline()
        self.assertIn("GOOGLE_API_KEY is not configured", str(ctx.exception))

    def test_top_k_parameter_configuration(self):
        """Verify configurable top_k retrieval parameter."""
        custom_rag = TechNovaRAG(top_k=5)
        self.assertEqual(custom_rag.top_k, 5)

    def test_in_code_policy_document_clauses(self):
        """Verify that all 5 key policy clauses are defined in the default document."""
        chunks = self.rag.load_and_split()
        combined_text = " ".join([c.page_content for c in chunks])
        self.assertIn("Annual Leave", combined_text)
        self.assertIn("Work From Home", combined_text)
        self.assertIn("Working Hours", combined_text)
        self.assertIn("Sick Leave", combined_text)
        self.assertIn("Learning Allowance", combined_text)
    def test_package_and_shim_compatibility(self):
        """Verify that src.technova_rag and root rag_engine export the exact same class."""
        self.assertIs(PackageTechNovaRAG, TechNovaRAG)


if __name__ == "__main__":
    unittest.main()
