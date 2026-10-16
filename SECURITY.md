# Security Policy

## 🔒 Credentials & Secret Isolation

TechNova Solutions RAG is committed to high security standards and strict isolation of sensitive API credentials.

### API Key Protection Principles
1. **Never Commit Secrets:** Under no circumstance should `.env` files, `.env.local` files, or raw Google Gemini API keys be committed to version control.
2. **Template Distribution:** Only `.env.example` containing inert placeholder tokens (`your_google_gemini_api_key_here`) is committed to the repository.
3. **Automated CI Enforcement:** Continuous integration checks run on every commit to verify that no `.env` or `.env.local` files exist in the repository tree.
4. **Local Loading Only:** The code utilizes `python-dotenv` to read credentials strictly from local uncommitted environment files or runtime shell variables.

## 🛡️ Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability or believe a secret has been exposed:
1. Please do **NOT** create a public GitHub issue.
2. Send an email directly to the project maintainer: `pt286355@gmail.com`.
3. Include relevant reproduction details and steps.

We take all security notices seriously and will respond promptly.
