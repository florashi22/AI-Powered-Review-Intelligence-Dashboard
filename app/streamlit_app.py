"""
AI Review Intelligence Dashboard
=================================
Orchestration layer: Claude API (reasoning) + Hugging Face (baseline comparison)

Run:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import anthropic
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Review Intelligence",
    page_icon="★",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_reviews.csv"
STAR_DISPLAY = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}
SENTIMENT_COLOR = {"Positive": "#3B6D11", "Negative": "#A32D2D", "Mixed": "#185FA5"}

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Hugging Face model (first run ~30s)…")
def load_hf_pipeline():
    """Load a lightweight HuggingFace sentiment classifier as a baseline."""
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(
        {
            "text": [
                "The food was amazing and service was great!",
                "Terrible experience, not coming back.",
                "Pretty decent but a bit slow.",
                "Best ramen I've ever had, will return!",
                "Good food but expensive.",
            ],
            "stars": [5, 1, 3, 5, 4],
        }
    )


# ── AI orchestration ──────────────────────────────────────────────────────────
def analyze_with_claude(review: str, client: anthropic.Anthropic) -> dict:
    """
    Use Claude as the AI orchestration layer.

    Claude reasons over the raw review text and returns a structured JSON
    object containing star prediction, confidence, sentiment, key themes,
    star-probability distribution, and an actionable business insight.

    This is the core of the orchestration pattern:
      Input (unstructured text) → LLM reasoning → Output (structured JSON)
    """
    prompt = f"""You are an expert review analysis engine. Analyze the customer review below and respond ONLY with a valid JSON object — no markdown fences, no explanation, just raw JSON.

Review: "{review}"

Return exactly this structure (fill in real values based on the review):
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
- advice: specific and actionable, address the business directly"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


def analyze_with_hf(review: str, pipe) -> dict | None:
    """HuggingFace distilbert baseline for side-by-side comparison."""
    if pipe is None:
        return None
    result = pipe(review[:512])[0]
    return {
        "label": result["label"].capitalize(),
        "score": round(result["score"] * 100, 1),
    }


def batch_analyze(reviews: list[str], client: anthropic.Anthropic) -> list[dict]:
    """Analyze a list of reviews and return results."""
    results = []
    for review in reviews:
        try:
            results.append(analyze_with_claude(review, client))
        except Exception as e:
            results.append({"error": str(e)})
    return results


# ── Chart helpers ─────────────────────────────────────────────────────────────
def plot_star_probs(probs: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    stars = [5, 4, 3, 2, 1]
    values = [probs.get(str(s), 0) for s in stars]
    colors = ["#3B6D11", "#639922", "#BA7517", "#A32D2D", "#6B1111"]
    bars = ax.barh([f"{s} ★" for s in stars], values, color=colors, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, 105)
    ax.axis("off")
    fig.patch.set_facecolor("none")
    plt.tight_layout(pad=0.5)
    return fig


def plot_batch_distribution(results: list[dict]) -> plt.Figure:
    valid = [r for r in results if "stars" in r]
    if not valid:
        return None
    counts = {s: 0 for s in range(1, 6)}
    for r in valid:
        counts[r["stars"]] = counts.get(r["stars"], 0) + 1
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.keys(), counts.values(),
           color=["#6B1111", "#A32D2D", "#BA7517", "#639922", "#3B6D11"])
    ax.set_xlabel("Stars", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xticks(range(1, 6))
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ★ AI Review Intelligence")
    st.markdown("---")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="Get yours free at console.anthropic.com",
    )

    st.markdown("---")
    mode = st.radio("Mode", ["Single review", "Batch analysis"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Architecture**")
    st.code(
        "Review text\n"
        "  ↓\n"
        "Claude API  ←── orchestrator\n"
        "  ↓               (reasoning)\n"
        "Structured JSON\n"
        "  ↓\n"
        "Stars · Themes · Advice\n"
        "\n"
        "HuggingFace distilbert\n"
        "  ↓\n"
        "Baseline comparison",
        language="text",
    )

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown(
        "- `anthropic` Python SDK\n"
        "- Hugging Face `transformers`\n"
        "- `distilbert-base-uncased`\n"
        "- `streamlit`, `pandas`, `matplotlib`"
    )

    st.markdown("---")
    st.caption("Built as a portfolio project demonstrating AI orchestration patterns.")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("AI-Powered Review Intelligence Dashboard")
st.caption(
    "Claude API acts as the **reasoning orchestrator** — not just a classifier. "
    "Compared side-by-side with a Hugging Face baseline to show the difference."
)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Single review
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single review":

    samples = load_sample_data()
    sample_options = ["— type your own —"] + samples["text"].tolist()

    col_input, col_sample = st.columns([3, 1])
    with col_input:
        review_text = st.text_area(
            "Customer review",
            height=130,
            placeholder="Paste any customer review here…",
        )
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        chosen = st.selectbox("Load a sample", sample_options)
        if chosen != "— type your own —" and not review_text:
            review_text = chosen

    run = st.button("Analyze with AI", type="primary", use_container_width=True)

    if run:
        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar.")
            st.stop()
        if not review_text or not review_text.strip():
            st.warning("Please enter a review.")
            st.stop()

        client = anthropic.Anthropic(api_key=api_key)

        col_claude, col_hf = st.columns(2)

        # ── Claude column ──────────────────────────────────────────────────
        with col_claude:
            st.markdown("### Claude AI")
            st.caption("Full reasoning orchestration layer")

            with st.spinner("Calling Claude API…"):
                try:
                    result = analyze_with_claude(review_text, client)
                except json.JSONDecodeError:
                    st.error("Claude returned invalid JSON. Please try again.")
                    st.stop()
                except anthropic.APIError as e:
                    st.error(f"API error: {e}")
                    st.stop()

            stars = result.get("stars", "?")
            sentiment = result.get("sentiment", "Mixed")

            m1, m2, m3 = st.columns(3)
            m1.metric("Rating", STAR_DISPLAY.get(stars, str(stars)))
            m2.metric("Confidence", result.get("confidence", "?"))
            m3.metric(
                "Sentiment",
                sentiment,
                delta_color="off",
            )

            st.markdown("**Key themes**")
            theme_html = " ".join(
                f"<span style='background:{'#EAF3DE' if t['type']=='positive' else '#FCEBEB' if t['type']=='negative' else '#E6F1FB'};"
                f"color:{'#3B6D11' if t['type']=='positive' else '#A32D2D' if t['type']=='negative' else '#185FA5'};"
                f"padding:3px 10px;border-radius:20px;font-size:13px;margin:2px;display:inline-block'>"
                f"{t['label']}</span>"
                for t in result.get("themes", [])
            )
            st.markdown(theme_html, unsafe_allow_html=True)

            st.markdown("**Star probability breakdown**")
            fig = plot_star_probs(result.get("star_probs", {}))
            st.pyplot(fig, use_container_width=True)

            st.markdown("**Business insight**")
            st.info(result.get("advice", "No advice generated."))

        # ── HuggingFace column ─────────────────────────────────────────────
        with col_hf:
            st.markdown("### HuggingFace Baseline")
            st.caption("distilbert-base-uncased-finetuned-sst-2-english")

            hf_pipe = load_hf_pipeline()
            hf_result = analyze_with_hf(review_text, hf_pipe)

            if hf_result:
                label = hf_result["label"]
                score = hf_result["score"]
                st.metric("Sentiment", label)
                st.metric("Confidence", f"{score}%")
                st.metric("Granularity", "Binary only")

                st.markdown("---")
                st.markdown("**What this model can't do:**")
                st.markdown(
                    "- ✗ Predict a star rating (1–5)\n"
                    "- ✗ Identify specific themes\n"
                    "- ✗ Give business advice\n"
                    "- ✗ Reason about *why* the review is positive/negative"
                )
                st.markdown("**Why Claude is different:**")
                st.markdown(
                    "Claude reads the review like a domain expert — "
                    "it returns structured, actionable output that a traditional "
                    "classifier fundamentally cannot produce."
                )
            else:
                st.warning(
                    "HuggingFace model unavailable. "
                    "Install `transformers` and `torch` to enable comparison."
                )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Batch analysis
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### Batch Analysis")
    st.caption(
        "Upload a CSV with a `text` column. "
        "Claude will analyze each review and return a downloadable results file."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a `text` column.")
            st.stop()

        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"{len(df)} reviews loaded.")

        limit = st.slider(
            "Reviews to analyze (API cost control)", 1, min(50, len(df)), 10
        )

        run_batch = st.button("Run Batch Analysis", type="primary")

        if run_batch:
            if not api_key:
                st.error("Enter your Anthropic API key in the sidebar.")
                st.stop()

            client = anthropic.Anthropic(api_key=api_key)
            reviews_to_run = df["text"].tolist()[:limit]

            progress = st.progress(0, text="Analyzing reviews…")
            results = []
            for i, review in enumerate(reviews_to_run):
                try:
                    r = analyze_with_claude(review, client)
                    r["original_text"] = review
                    results.append(r)
                except Exception as e:
                    results.append({"original_text": review, "error": str(e)})
                progress.progress((i + 1) / limit, text=f"Analyzed {i+1}/{limit}…")

            progress.empty()

            # Build results dataframe
            rows = []
            for r in results:
                if "error" not in r:
                    rows.append({
                        "text": r.get("original_text", "")[:80] + "…",
                        "predicted_stars": r.get("stars"),
                        "sentiment": r.get("sentiment"),
                        "confidence": r.get("confidence"),
                        "advice": r.get("advice", ""),
                    })
            results_df = pd.DataFrame(rows)

            st.markdown("#### Results")
            st.dataframe(results_df, use_container_width=True)

            col_chart, col_dl = st.columns([2, 1])
            with col_chart:
                fig = plot_batch_distribution(results)
                if fig:
                    st.pyplot(fig, use_container_width=True)
            with col_dl:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.download_button(
                    "Download results CSV",
                    data=results_df.to_csv(index=False),
                    file_name="review_analysis_results.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption(
    "Source: [github.com/your-username/ai-review-intelligence](https://github.com) · "
    "Built with Anthropic Claude + Hugging Face Transformers"
)
