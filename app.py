"""Streamlit UI for the AI-Powered Experience Matcher."""

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE
from src.matcher import ExperienceMatcher

load_dotenv()

IS_CLOUD = bool(os.environ.get("STREAMLIT_SHARING"))


def _resolve_api_key(sidebar_input: str = "") -> str:
    """Resolve API key: sidebar input > st.secrets > .env."""
    if sidebar_input:
        return sidebar_input
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key and key != "your-key-here":
            return key
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Experience Matcher",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 8px;
    }
    .skill-tag {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.15);
        color: rgb(129, 140, 248);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 2px 3px;
        font-weight: 500;
    }
    .skill-match {
        display: inline-block;
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 2px 3px;
        font-weight: 600;
    }
    .skill-miss {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 2px 3px;
        font-weight: 500;
    }
    .fit-box {
        border-left: 4px solid #22c55e;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        background-color: rgba(34, 197, 94, 0.08);
    }
    .fit-box strong { color: #4ade80; }
    .category-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .cat-work { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    .cat-project { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .cat-education { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; }
    .cat-other { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
    .meta-footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    .app-footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.75rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
    .app-footer a { color: #818cf8; text-decoration: none; }
    .app-footer a:hover { text-decoration: underline; }
    /* Prevent copy button from overlapping code block text */
    .stCode pre { padding-right: 3rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=_resolve_api_key(),
        help="Enter your OpenAI API key. It's stored only in this session.",
    )

    st.divider()

    top_k = st.slider("Number of results", min_value=1, max_value=5, value=3)

    generate_ai = st.toggle("Generate AI-optimized descriptions", value=True)

    st.divider()

    st.subheader("Model Info")
    st.caption(f"**Embedding:** `{EMBEDDING_MODEL}`")
    st.caption(f"**LLM:** `{LLM_MODEL}`")
    st.caption(f"**Temperature:** `{LLM_TEMPERATURE}`")

    st.divider()

    with st.expander("About"):
        st.markdown(
            "**AI Experience Matcher** uses Retrieval-Augmented Generation "
            "(RAG) to semantically match your experiences to job descriptions. "
            "It retrieves the most relevant experiences via FAISS vector search, "
            "then uses GPT-4o-mini to tailor bullet points and provide a fit analysis."
        )


# ---------------------------------------------------------------------------
# Cached matcher initialisation
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_matcher(api_key: str) -> ExperienceMatcher:
    """Create and cache an ExperienceMatcher instance."""
    os.environ["OPENAI_API_KEY"] = api_key
    matcher = ExperienceMatcher()
    matcher.load_experiences("data/experiences.json")
    matcher.load_vector_store()
    return matcher


# ---------------------------------------------------------------------------
# Sample job descriptions
# ---------------------------------------------------------------------------

SAMPLE_JDS = {
    "Custom": "",
    "Data Analyst": (
        "We are seeking a Data Analyst with 1-3 years of experience. "
        "Required: Python, SQL, data visualization (Tableau or PowerBI). "
        "Nice to have: ETL pipeline experience, machine learning basics, "
        "cloud platforms (AWS/Azure). You will analyze large datasets, "
        "build dashboards, and present insights to stakeholders."
    ),
    "ML Engineer": (
        "Machine Learning Engineer needed to develop and deploy production ML "
        "models. You will build end-to-end ML pipelines, run experiments with "
        "MLflow, and deploy models via REST APIs. Required: Python, PyTorch or "
        "TensorFlow, scikit-learn, feature engineering. Nice to have: LLMs, "
        "NLP, time-series forecasting, Docker, Airflow."
    ),
    "Software Developer": (
        "We are looking for a Software Developer to join our backend team. "
        "You will design and build REST APIs, write clean maintainable code, "
        "and collaborate in an agile environment. Required: Python or Java, "
        "SQL, Git, CI/CD. Nice to have: Flask/Django, Docker, cloud experience. "
        "Strong problem-solving skills and CS fundamentals expected."
    ),
    "Data Scientist": (
        "We are hiring a Data Scientist to work on predictive analytics and "
        "statistical modeling. You will design experiments, build ML models, "
        "and communicate findings to stakeholders. Required: Python, R or SQL, "
        "statistical analysis, machine learning (scikit-learn, XGBoost). "
        "Nice to have: deep learning, NLP, cloud platforms, data visualization."
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

CATEGORY_CSS = {
    "work": "cat-work",
    "project": "cat-project",
    "education": "cat-education",
    "other": "cat-other",
}


def render_skills(skills: list) -> str:
    """Render a list of skills as HTML tags."""
    return " ".join(f'<span class="skill-tag">{s}</span>' for s in skills)


def render_category(category: str) -> str:
    """Render a category badge."""
    css = CATEGORY_CSS.get(category, "cat-other")
    return f'<span class="category-badge {css}">{category}</span>'


def extract_jd_skills(job_description: str) -> set:
    """Extract likely skill keywords from a job description."""
    known_skills = {
        "python",
        "java",
        "javascript",
        "typescript",
        "sql",
        "r",
        "c++",
        "go",
        "rust",
        "scala",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "flask",
        "django",
        "fastapi",
        "react",
        "angular",
        "vue",
        "node.js",
        "spring",
        "express",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "xgboost",
        "keras",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "git",
        "ci/cd",
        "jenkins",
        "terraform",
        "ansible",
        "tableau",
        "powerbi",
        "power bi",
        "excel",
        "jupyter",
        "nlp",
        "llm",
        "llms",
        "ml",
        "machine learning",
        "deep learning",
        "computer vision",
        "etl",
        "data visualization",
        "data engineering",
        "rest api",
        "rest apis",
        "api",
        "apis",
        "microservices",
        "airflow",
        "spark",
        "hadoop",
        "kafka",
        "redis",
        "mongodb",
        "postgresql",
        "mysql",
        "sqlite",
        "faiss",
        "agile",
        "scrum",
        "jira",
        "html",
        "css",
        "latex",
        "mlflow",
        "wandb",
        "weights & biases",
        "a/b testing",
        "statistical analysis",
        "feature engineering",
        "time-series",
        "forecasting",
        "predictive modeling",
        "streamlit",
        "gradio",
    }
    jd_lower = job_description.lower()
    found = set()
    for skill in known_skills:
        if skill in jd_lower:
            found.add(skill)
    return found


def render_skill_overlap(exp_skills: list, jd_skills: set) -> str:
    """Render matched (green) and missing (red) JD skills only."""
    exp_skills_lower = {s.lower() for s in exp_skills}
    matched = jd_skills & exp_skills_lower
    missing = jd_skills - exp_skills_lower

    html_parts = []
    for skill in sorted(matched):
        html_parts.append(f'<span class="skill-match">{skill}</span>')
    for skill in sorted(missing):
        html_parts.append(f'<span class="skill-miss">{skill}</span>')

    return " ".join(html_parts)


def format_results_for_export(result: dict) -> str:
    """Format all results as a plain-text export."""
    lines = []
    lines.append("=" * 70)
    lines.append("AI-POWERED EXPERIENCE MATCHER — RESULTS")
    lines.append("=" * 70)

    if result.get("fit_analysis"):
        lines.append("\nFIT ANALYSIS:")
        lines.append(result["fit_analysis"])

    for match in result["matched_experiences"]:
        exp = match["experience"]
        score_pct = int(match["relevance_score"] * 100)
        lines.append(f"\n{'─' * 70}")
        lines.append(
            f"#{match['rank']}  {exp['title']} @ {exp['company']}  "
            f"(Relevance: {score_pct}%)"
        )
        lines.append(f"Duration: {exp['duration']}")
        lines.append(f"Category: {exp['category']}")
        lines.append(f"Skills: {', '.join(exp['skills'])}")
        lines.append(f"\nOriginal:\n{exp['description']}")
        if match.get("tailored_description"):
            lines.append(f"\nAI-Optimized:\n{match['tailored_description']}")
        lines.append(f"\nAchievements:")
        for a in exp["achievements"]:
            lines.append(f"  - {a}")

    meta = result["metadata"]
    lines.append(f"\n{'=' * 70}")
    lines.append("METADATA")
    lines.append(f"  Model: {meta['model_used']}")
    if meta.get("query_time_seconds"):
        lines.append(f"  Time: {meta['query_time_seconds']}s")
        lines.append(
            f"  Tokens: {meta['total_input_tokens']}+{meta['total_output_tokens']}"
        )
        lines.append(f"  Cost: ${meta['estimated_cost']:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("\U0001f3af AI-Powered Experience Matcher")
st.caption(
    "Paste a job description to find your most relevant experiences "
    "and get AI-optimized bullet points."
)

if IS_CLOUD:
    st.caption("\u2139\ufe0f Running on Streamlit Cloud")

# ---------------------------------------------------------------------------
# Experience overview (collapsible)
# ---------------------------------------------------------------------------

with st.expander("\U0001f4da Your Loaded Experiences"):
    try:
        with open("data/experiences.json", "r", encoding="utf-8") as f:
            loaded_experiences = json.load(f)

        st.caption(f"**{len(loaded_experiences)}** experiences loaded")

        overview_data = []
        for exp in loaded_experiences:
            overview_data.append(
                {
                    "Title": exp["title"],
                    "Company": exp["company"],
                    "Category": exp["category"].capitalize(),
                }
            )

        st.dataframe(
            pd.DataFrame(overview_data),
            width="stretch",
            hide_index=True,
        )
    except FileNotFoundError:
        st.warning("No experiences file found at data/experiences.json")

# ---------------------------------------------------------------------------
# Sample JD selector + input
# ---------------------------------------------------------------------------

st.text("")

sample_choice = st.selectbox(
    "Try a sample JD",
    options=list(SAMPLE_JDS.keys()),
    index=0,
    help="Select a sample job description or choose 'Custom' to write your own.",
)

if sample_choice != "Custom":
    default_jd = SAMPLE_JDS[sample_choice]
else:
    default_jd = ""

job_description = st.text_area(
    "Job Description",
    value=default_jd,
    height=250,
    placeholder=SAMPLE_JDS["Data Analyst"],
    label_visibility="collapsed",
)

col_count, col_btn = st.columns([3, 1])
with col_count:
    if job_description:
        char_count = len(job_description)
        word_count = len(job_description.split())
        info = f"{char_count} characters | ~{word_count} words"
        if char_count > 5000:
            st.caption(f":warning: {info} (very long — consider trimming)")
        else:
            st.caption(info)
with col_btn:
    analyze_clicked = st.button(
        "\U0001f50d Analyze", type="primary", use_container_width=True
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if analyze_clicked:
    resolved_key = _resolve_api_key(api_key_input)
    if not resolved_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    if not job_description or not job_description.strip():
        st.warning("Please paste a job description above.")
        st.stop()

    if len(job_description) > 10000:
        st.warning(
            "Job description is very long (>10,000 chars). "
            "Consider trimming for better results."
        )

    try:
        with st.spinner("\U0001f916 Analyzing your experiences..."):
            matcher = get_matcher(resolved_key)

            if generate_ai:
                result = matcher.match_and_generate(job_description, top_k=top_k)
            else:
                search_results = matcher.search(job_description, top_k=top_k)
                result = {
                    "matched_experiences": [
                        {
                            "rank": r["rank"],
                            "relevance_score": r["relevance_score"],
                            "experience": r["experience"],
                            "tailored_description": None,
                        }
                        for r in search_results
                    ],
                    "fit_analysis": None,
                    "metadata": {
                        "total_experiences_searched": len(matcher.experiences),
                        "model_used": LLM_MODEL,
                        "query_time_seconds": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "estimated_cost": 0,
                    },
                }

        st.session_state["result"] = result
        st.session_state["job_description"] = job_description

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            st.error(
                "**API key error.** Please check that your OpenAI API key is valid "
                "and has sufficient credits."
            )
        elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
            st.error(
                "**Rate limit reached.** OpenAI is throttling requests. "
                "Please wait a moment and try again."
            )
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            st.error(
                "**Request timed out.** The API took too long to respond. "
                "Please try again."
            )
        else:
            st.error(f"**Something went wrong:** {error_msg}")
        st.stop()

# Display stored results
if "result" in st.session_state:
    result = st.session_state["result"]
    meta = result["metadata"]
    jd_for_skills = st.session_state.get("job_description", "")
    jd_skills = extract_jd_skills(jd_for_skills)

    if meta.get("query_time_seconds"):
        st.success(
            f"Found {len(result['matched_experiences'])} matches "
            f"in {meta['query_time_seconds']}s"
        )

    # Export button
    export_text = format_results_for_export(result)
    st.download_button(
        label="\U0001f4e5 Export All Results",
        data=export_text,
        file_name="experience_matcher_results.txt",
        mime="text/plain",
    )

    # Fit analysis
    if result.get("fit_analysis"):
        st.markdown(
            f'<div class="fit-box">'
            f"<strong>\U0001f4ca Fit Analysis</strong><br><br>"
            f"{result['fit_analysis']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Matched experiences
    for match in result["matched_experiences"]:
        exp = match["experience"]
        score = match["relevance_score"]
        score_pct = int(score * 100)
        cat_html = render_category(exp["category"])

        st.markdown("---")

        # Header row
        header_col, score_col = st.columns([4, 1])
        with header_col:
            st.markdown(
                f'<span class="rank-badge">#{match["rank"]}</span> '
                f"**{exp['title']}** @ {exp['company']} {cat_html}",
                unsafe_allow_html=True,
            )
            st.caption(f"\U0001f4c5 {exp['duration']}")
        with score_col:
            st.metric("Relevance", f"{score_pct}%")

        # Score progress bar
        st.progress(float(min(score, 1.0)))

        # Skill overlap visualization
        if jd_skills:
            st.markdown("**Skill Overlap**")
            overlap_html = render_skill_overlap(exp["skills"], jd_skills)
            st.markdown(overlap_html, unsafe_allow_html=True)
            exp_lower = {s.lower() for s in exp["skills"]}
            matched_count = len(exp_lower & jd_skills)
            st.caption(
                f"{matched_count} of {len(jd_skills)} JD skills matched "
                f"| {len(exp['skills'])} experience skills total"
            )

        # Tabs for original vs AI-optimized
        if match.get("tailored_description"):
            tab_original, tab_ai = st.tabs(
                ["\U0001f4c4 Original", "\u2728 AI-Optimized"]
            )
        else:
            tab_original = st.container()
            tab_ai = None

        with tab_original:
            st.write(exp["description"])
            st.markdown(render_skills(exp["skills"]), unsafe_allow_html=True)

        if tab_ai is not None:
            with tab_ai:
                st.markdown(match["tailored_description"])
                st.caption("Copy-ready version:")
                st.code(match["tailored_description"], language=None, wrap_lines=True)

        # Expandable details
        with st.expander("Details"):
            detail_left, detail_right = st.columns(2)
            with detail_left:
                st.markdown("**Achievements**")
                for a in exp["achievements"]:
                    st.markdown(f"- {a}")
            with detail_right:
                st.markdown("**Keywords**")
                st.markdown(render_skills(exp["keywords"]), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Comparison summary table
    # ------------------------------------------------------------------
    if len(result["matched_experiences"]) > 1:
        st.markdown("---")
        st.subheader("\U0001f4ca Comparison Summary")

        comparison_rows = []
        for match in result["matched_experiences"]:
            exp = match["experience"]
            exp_lower = {s.lower() for s in exp["skills"]}
            top_matched = (
                sorted(exp_lower & jd_skills) if jd_skills else exp["skills"][:3]
            )
            top_skills_str = ", ".join(s.capitalize() for s in top_matched[:3])
            if not top_skills_str:
                top_skills_str = ", ".join(exp["skills"][:3])

            comparison_rows.append(
                {
                    "Rank": f"#{match['rank']}",
                    "Experience": exp["title"],
                    "Score": f"{int(match['relevance_score'] * 100)}%",
                    "Category": exp["category"].capitalize(),
                    "Top Skills": top_skills_str,
                }
            )

        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Rank"),
            width="stretch",
        )

    # Query metadata footer
    if meta.get("query_time_seconds"):
        st.markdown(
            f'<div class="meta-footer">'
            f"Completed in {meta['query_time_seconds']}s | "
            f"Model: {meta['model_used']} | "
            f"Tokens: {meta['total_input_tokens']}+{meta['total_output_tokens']} | "
            f"Cost: ${meta['estimated_cost']:.4f}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# App footer (always visible)
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="app-footer">'
    "Built with Python, LangChain, FAISS &amp; Streamlit<br>"
    '<a href="https://github.com/SmoKerYM/AI_Powered_Experience_Matcher">'
    "GitHub</a> &middot; "
    "&copy; 2026 Mingwei Yan"
    "</div>",
    unsafe_allow_html=True,
)
