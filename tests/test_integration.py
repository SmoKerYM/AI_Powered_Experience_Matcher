"""Layer 4: End-to-end integration tests — full pipeline via match_and_generate."""

import pytest

pytestmark = pytest.mark.api

_SAMPLE_JD = (
    "We are seeking a Data Analyst with 1-3 years of experience. "
    "Required: Python, SQL, data visualization (Tableau or PowerBI). "
    "Nice to have: ETL pipeline experience, machine learning basics, "
    "cloud platforms (AWS/Azure). You will analyze large datasets, "
    "build dashboards, and present insights to stakeholders."
)


@pytest.fixture(scope="module")
def pipeline_result(matcher_with_vectorstore):
    """Run the full pipeline once and reuse across all integration tests."""
    return matcher_with_vectorstore.match_and_generate(_SAMPLE_JD)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_match_and_generate_structure(pipeline_result):
    """Result dict contains all four top-level keys."""
    assert "job_description" in pipeline_result
    assert "matched_experiences" in pipeline_result
    assert "fit_analysis" in pipeline_result
    assert "metadata" in pipeline_result


def test_match_and_generate_returns_correct_count(pipeline_result):
    """Pipeline returns exactly 3 matched experiences (top_k=3)."""
    assert len(pipeline_result["matched_experiences"]) == 3


def test_match_and_generate_experience_structure(pipeline_result):
    """Each matched experience has rank, score, experience dict, and tailored description."""
    for match in pipeline_result["matched_experiences"]:
        assert "rank" in match
        assert "relevance_score" in match
        assert "experience" in match
        assert "tailored_description" in match


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_match_and_generate_metadata(pipeline_result):
    """Metadata contains valid model info, timing, token counts, and cost."""
    meta = pipeline_result["metadata"]
    assert meta["total_experiences_searched"] > 0
    assert meta["model_used"] == "gpt-4o-mini"
    assert meta["query_time_seconds"] > 0
    assert meta["total_input_tokens"] > 0
    assert meta["total_output_tokens"] > 0
    assert meta["estimated_cost"] > 0


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------


def test_match_and_generate_tailored_descriptions_nonempty(pipeline_result):
    """Every tailored description is non-empty and contains bullet points."""
    for match in pipeline_result["matched_experiences"]:
        td = match["tailored_description"]
        assert isinstance(td, str) and len(td.strip()) > 0
        assert td.count("-") >= 2 or td.count("•") >= 2, (
            f"No bullet points in tailored description for {match['experience']['title']}"
        )


def test_match_and_generate_fit_analysis_has_assessment(pipeline_result):
    """Fit analysis includes a Strong/Moderate/Weak assessment."""
    fa = pipeline_result["fit_analysis"].lower()
    assert any(w in fa for w in ["strong", "moderate", "weak"]), (
        "Fit analysis missing strength assessment"
    )


def test_match_and_generate_scores_ordered(pipeline_result):
    """Relevance scores are sorted in descending order."""
    scores = [m["relevance_score"] for m in pipeline_result["matched_experiences"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_match_and_generate_short_jd(matcher_with_vectorstore):
    """A minimal 3-word JD still returns valid results without errors."""
    result = matcher_with_vectorstore.match_and_generate("Data analyst Python")
    assert len(result["matched_experiences"]) == 3
    assert len(result["fit_analysis"]) > 20
