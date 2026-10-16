# Contributing to TechNova Solutions RAG

Thank you for your interest in contributing to the TechNova Solutions RAG System!

## Code of Conduct
Please ensure all communications remain respectful, collaborative, and inclusive.

## Workflow & PR Guidelines
1. **Fork the Repository:** Create your personal fork on GitHub.
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. **Set Up Development Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```
4. **Adhere to Code Standards:**
   - Use PEP 8 conventions.
   - Maintain type annotations and docstrings.
   - Keep answers strictly grounded in the document context.
   - **NEVER commit sensitive credentials or real API keys.**
5. **Run Tests:**
   ```bash
   python -m unittest discover tests -v
   ```
6. **Commit & Push:**
   Use semantic commit messages (e.g. `feat:`, `fix:`, `docs:`, `test:`).
7. **Open a Pull Request:** Describe your changes, motivations, and test results.
