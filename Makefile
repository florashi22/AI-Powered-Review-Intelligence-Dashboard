# AI Review Intelligence — developer shortcuts
# Usage: make <target>

.PHONY: install install-full run test lint format clean help

## Install core dependencies (no PyTorch)
install:
	pip install anthropic streamlit pandas numpy scikit-learn matplotlib seaborn joblib python-dotenv pytest ruff

## Install all dependencies including HuggingFace (large download)
install-full:
	pip install -r requirements.txt

## Run the Streamlit dashboard
run:
	streamlit run app/streamlit_app.py

## Run unit tests
test:
	pytest tests/ -v

## Run linter
lint:
	ruff check app/ tests/

## Auto-fix linting issues
format:
	ruff check app/ tests/ --fix

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

## Show this help
help:
	@echo ""
	@echo "Available commands:"
	@echo "  make install       Install core dependencies"
	@echo "  make install-full  Install all deps including HuggingFace"
	@echo "  make run           Start the Streamlit dashboard"
	@echo "  make test          Run unit tests"
	@echo "  make lint          Run code linter (ruff)"
	@echo "  make format        Auto-fix lint issues"
	@echo "  make clean         Remove cache files"
	@echo ""
