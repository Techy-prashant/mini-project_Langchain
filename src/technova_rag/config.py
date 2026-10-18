"""
Configuration and constants for TechNova Solutions RAG pipeline.
Defines model selections, text splitting parameters, and default evaluation questions.
"""

import os
from typing import Dict, Any, List

# Type alias for RAG query results
QueryResult = Dict[str, Any]

# Default RAG Parameters
DEFAULT_CHUNK_SIZE: int = 300
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_TOP_K: int = 3

# Gemini Models
DEFAULT_EMBEDDING_MODEL: str = "gemini-embedding-001"
DEFAULT_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")

# Default in-code policy document ensuring zero external dependency failure
DEFAULT_POLICY_DOCUMENT: str = """TechNova Solutions - Employee Policy Document


1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.


2. Work From Home: Employees may work remotely on Wednesdays and Fridays. Employees must inform their manager by email at least one working day in advance. Remote work is subject to manager approval.


3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM, Monday through Friday. Employees receive a one-hour lunch break.


4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.


5. Learning Allowance: Each full-time employee can claim up to ₹12,000 per calendar year for approved job-related courses and certifications. Claims must include proof of payment and course completion."""

# Prescribed test benchmark questions
DEFAULT_TEST_QUESTIONS: List[str] = [
    "How many paid annual leave days does a full-time employee receive per year?",
    "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?",
    "What is the company's parental leave policy?"
]
