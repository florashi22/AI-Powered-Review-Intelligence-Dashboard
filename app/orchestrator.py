"""
orchestrator.py
================
Reusable AI orchestration layer.

This module abstracts the Claude API call into a clean interface
that any script, notebook, or app can import and use.

Example
-------
>>> from app.orchestrator import ReviewOrchestrator
>>> orch = ReviewOrchestrator(api_key="sk-ant-...")
>>> result = orch.analyze("Amazing food, slow service.")
>>> print(result["stars"], result["sentiment"])
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Theme:
    label: str
    type: str  # "positive" | "negative" | "neutral"


@dataclass
class ReviewAnalysis:
    stars: int
    confidence: str
    sentiment: str
    themes: list[Theme]
    star_probs: dict[str, int]
    advice: str
    raw_review: str = ""
    latency_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: dict, raw_review: str = "", latency_ms: float = 0.0):
        themes = [Theme(**t) for t in d.get("themes", [])]
        return cls(
            stars=int(d.get("stars", 3)),
            confidence=d.get("confidence", "?"),
            sentiment=d.get("sentiment", "Mixed"),
            themes=themes,
            star_probs={str(k): int(v) for k, v in d.get("star_probs", {}).items()},
            advice=d.get("advice", ""),
            raw_review=raw_review,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict:
        return {
            "stars": self.stars,
            "confidence": self.confidence,
            "sentiment": self.sentiment,
            "themes": [{"label": t.label, "type": t.type} for t in self.themes],
            "star_probs": self.star_probs,
            "advice": self.advice,
            "raw_review": self.raw_review,
            "latency_ms": round(self.latency_ms, 1),
        }


# ── Main orchestrator class ────────────────────────────────────────────────────

class ReviewOrchestrator:
    """
    AI orchestration layer for customer review analysis.

    Uses Claude as the reasoning engine. Claude reads the raw review text
    and returns a structured JSON object — star rating, confidence,
    sentiment label, key themes, star-probability distribution, and
    an actionable business insight.

    This is the "orchestration" pattern:
        unstructured text → LLM reasoning → structured output

    Parameters
    ----------
    api_key : str
        Your Anthropic API key (from console.anthropic.com).
    model : str
        Claude model to use. Defaults to claude-sonnet-4-5.
    max_retries : int
        Number of retry attempts on API failure.
    """

    MODEL = "claude-sonnet-4-5"

    PROMPT_TEMPLATE = """You are an expert review analysis engine. Analyze the customer review below and respond ONLY with a valid JSON object — no markdown, no explanation, just raw JSON.

Review: "{review}"

Return exactly this structure:
{{
  "stars": 4,
  "confidence": "87%",
  "sentiment": "Positive",
  "themes": [
    {{"label": "Food quality", "type": "positive"}},
    {{"label": "Wait time",    "type": "negative"}},
    {{"label": "Pricing",      "type": "neutral"}}
  ],
  "star_probs": {{"1": 2, "2": 3, "3": 8, "4": 55, "5": 32}},
  "advice": "One or two actionable sentences for the business owner."
}}

Rules:
- stars: integer 1–5
- confidence: string like "87%"
- sentiment: one of "Positive", "Negative", "Mixed"
- themes: 3–5 items; type must be "positive", "negative", or "neutral"
- star_probs: integer percentages that sum to 100
- advice: specific and actionable"""

    def __init__(
        self,
        api_key: str,
        model: str = MODEL,
        max_retries: int = 2,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def analyze(self, review: str) -> ReviewAnalysis:
        """
        Analyze a single review.

        Parameters
        ----------
        review : str
            Raw customer review text.

        Returns
        -------
        ReviewAnalysis
            Structured analysis result.

        Raises
        ------
        ValueError
            If Claude returns unparseable JSON after all retries.
        """
        prompt = self.PROMPT_TEMPLATE.format(review=review.replace('"', "'"))

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw = message.content[0].text
                cleaned = re.sub(r"```json|```", "", raw).strip()
                data = json.loads(cleaned)
                return ReviewAnalysis.from_dict(data, raw_review=review, latency_ms=latency_ms)

            except json.JSONDecodeError as e:
                if attempt == self.max_retries:
                    raise ValueError(
                        f"Claude returned invalid JSON after {self.max_retries + 1} attempts: {e}"
                    ) from e
            except anthropic.APIError as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff

    def analyze_batch(
        self,
        reviews: list[str],
        on_progress: Optional[callable] = None,
    ) -> list[ReviewAnalysis | dict]:
        """
        Analyze a list of reviews sequentially.

        Parameters
        ----------
        reviews : list[str]
            List of review texts.
        on_progress : callable, optional
            Called with (index, total) after each review completes.

        Returns
        -------
        list
            List of ReviewAnalysis objects (or error dicts on failure).
        """
        results = []
        for i, review in enumerate(reviews):
            try:
                result = self.analyze(review)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "raw_review": review})
            if on_progress:
                on_progress(i + 1, len(reviews))
        return results

    def summarize_batch(self, results: list[ReviewAnalysis]) -> dict:
        """
        Compute aggregate statistics over a batch of analyses.

        Returns a dict with average stars, sentiment distribution,
        most common themes, and average confidence.
        """
        valid = [r for r in results if isinstance(r, ReviewAnalysis)]
        if not valid:
            return {}

        avg_stars = sum(r.stars for r in valid) / len(valid)

        sentiment_dist: dict[str, int] = {}
        for r in valid:
            sentiment_dist[r.sentiment] = sentiment_dist.get(r.sentiment, 0) + 1

        theme_counts: dict[str, int] = {}
        for r in valid:
            for t in r.themes:
                theme_counts[t.label] = theme_counts.get(t.label, 0) + 1
        top_themes = sorted(theme_counts.items(), key=lambda x: -x[1])[:5]

        conf_values = []
        for r in valid:
            try:
                conf_values.append(float(r.confidence.replace("%", "")))
            except ValueError:
                pass
        avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0

        return {
            "total_reviews": len(valid),
            "average_stars": round(avg_stars, 2),
            "sentiment_distribution": sentiment_dist,
            "top_themes": top_themes,
            "average_confidence_pct": round(avg_confidence, 1),
        }
