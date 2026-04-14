"""
tests/test_orchestrator.py
===========================
Unit tests for the ReviewOrchestrator.

Run:
    pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestrator import ReviewOrchestrator, ReviewAnalysis, Theme


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_RESPONSE = {
    "stars": 4,
    "confidence": "82%",
    "sentiment": "Positive",
    "themes": [
        {"label": "Food quality", "type": "positive"},
        {"label": "Wait time",    "type": "negative"},
    ],
    "star_probs": {"1": 2, "2": 5, "3": 10, "4": 55, "5": 28},
    "advice": "Focus on reducing wait times.",
}


def make_mock_client(response_text: str):
    """Return a mock Anthropic client that returns `response_text`."""
    mock_content = MagicMock()
    mock_content.text = response_text

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    return mock_client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestReviewAnalysis:
    def test_from_dict_parses_correctly(self):
        analysis = ReviewAnalysis.from_dict(MOCK_RESPONSE, raw_review="Great food!")
        assert analysis.stars == 4
        assert analysis.confidence == "82%"
        assert analysis.sentiment == "Positive"
        assert len(analysis.themes) == 2
        assert analysis.themes[0].label == "Food quality"
        assert analysis.themes[0].type == "positive"
        assert analysis.star_probs["4"] == 55
        assert analysis.advice == "Focus on reducing wait times."
        assert analysis.raw_review == "Great food!"

    def test_to_dict_roundtrip(self):
        analysis = ReviewAnalysis.from_dict(MOCK_RESPONSE)
        d = analysis.to_dict()
        assert d["stars"] == 4
        assert d["sentiment"] == "Positive"
        assert len(d["themes"]) == 2

    def test_star_probs_are_ints(self):
        analysis = ReviewAnalysis.from_dict(MOCK_RESPONSE)
        for v in analysis.star_probs.values():
            assert isinstance(v, int)


class TestReviewOrchestrator:
    def setup_method(self):
        self.mock_client = make_mock_client(json.dumps(MOCK_RESPONSE))
        self.orch = ReviewOrchestrator.__new__(ReviewOrchestrator)
        self.orch.client = self.mock_client
        self.orch.model = ReviewOrchestrator.MODEL
        self.orch.max_retries = 2

    def test_analyze_returns_analysis(self):
        result = self.orch.analyze("The food was great but slow service.")
        assert isinstance(result, ReviewAnalysis)
        assert result.stars == 4
        assert result.sentiment == "Positive"

    def test_analyze_strips_markdown_fences(self):
        fenced = f"```json\n{json.dumps(MOCK_RESPONSE)}\n```"
        self.orch.client = make_mock_client(fenced)
        result = self.orch.analyze("Test review")
        assert result.stars == 4

    def test_analyze_invalid_json_raises(self):
        self.orch.client = make_mock_client("this is not json")
        self.orch.max_retries = 0
        with pytest.raises(ValueError, match="invalid JSON"):
            self.orch.analyze("Test review")

    def test_analyze_batch_returns_list(self):
        reviews = ["Great!", "Terrible.", "Okay."]
        results = self.orch.analyze_batch(reviews)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ReviewAnalysis)

    def test_summarize_batch(self):
        reviews = ["Great!", "Terrible.", "Okay."]
        results = self.orch.analyze_batch(reviews)
        summary = self.orch.summarize_batch(results)
        assert "average_stars" in summary
        assert "sentiment_distribution" in summary
        assert summary["total_reviews"] == 3
