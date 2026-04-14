# Changelog

All notable changes to this project will be documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025-04-14

### Added
- `ReviewOrchestrator` class — reusable AI orchestration layer using Claude API
- `ReviewAnalysis` dataclass — typed, structured output from Claude
- Streamlit dashboard with single-review and batch-analysis modes
- HuggingFace distilbert baseline comparison panel
- Batch analysis: upload CSV → analyze → download results
- `exploration.ipynb` notebook: BoW vs distilbert vs Claude walkthrough
- 8 unit tests with mocked API responses (no real API calls during testing)
- GitHub Actions CI — tests on Python 3.10, 3.11, 3.12
- `Makefile` for one-command developer workflow
- `pyproject.toml` for modern Python packaging
- `sample_reviews.csv` sample dataset
- Assets: rating distribution, model comparison, confusion matrix, architecture charts

### Architecture
- Input: raw review text (unstructured)
- Orchestration: Claude API reasons over text, returns typed JSON
- Output: stars, confidence, sentiment, themes, star-probability distribution, business advice
- Comparison: HuggingFace distilbert binary classifier shown side-by-side

---

## Planned for [0.2.0]

- Cloudflare Workers deployment for serverless API hosting
- Batch topic modeling — cluster recurring complaints across reviews
- Fine-tuned distilbert comparison tab
- Multi-language support via Claude's multilingual capabilities
- Export-to-PDF report feature
