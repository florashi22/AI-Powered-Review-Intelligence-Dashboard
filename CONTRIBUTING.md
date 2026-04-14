# Contributing

Thank you for your interest in contributing to AI Review Intelligence!

## Getting started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/ai-review-intelligence.git
   cd ai-review-intelligence
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   make install
   ```
4. Copy `.env.example` to `.env` and add your Anthropic API key.

## Development workflow

- Create a new branch for your change:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Make your changes.
- Run the tests to make sure nothing is broken:
  ```bash
  make test
  ```
- Run the linter:
  ```bash
  make lint
  ```
- Commit and push, then open a Pull Request against `main`.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run `make format` to auto-fix issues.

- Line length: 100 characters.
- All new code should have docstrings.
- All new functions should have type hints.

## Adding a new model

To add a new analysis model or approach:

1. Add a new method to `app/orchestrator.py` (e.g., `analyze_with_openai`).
2. Add corresponding unit tests in `tests/test_orchestrator.py` using mocked API responses.
3. Wire it into the Streamlit dashboard in `app/streamlit_app.py`.
4. Document it in the notebook and README.

## Reporting issues

Please open a GitHub Issue with:
- A clear description of the bug or feature request.
- Steps to reproduce (for bugs).
- Your Python version and OS.
