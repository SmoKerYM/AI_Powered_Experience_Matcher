"""Layer 4: End-to-end integration tests — full pipeline via match_and_generate."""

import pytest

pytestmark = pytest.mark.api


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_match_and_generate_structure(matcher_with_vectorstore, sample_jd):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    assert "job_description" in result
    assert "matched_experiences" in result
    assert "fit_analysis" in result
    assert "metadata" in result


def test_match_and_generate_returns_correct_count(matcher_with_vectorstore, sample_jd):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    assert len(result["matched_experiences"]) == 3


def test_match_and_generate_experience_structure(matcher_with_vectorstore, sample_jd):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    for match in result["matched_experiences"]:
        assert "rank" in match
        assert "relevance_score" in match
        assert "experience" in match
        assert "tailored_description" in match


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_match_and_generate_metadata(matcher_with_vectorstore, sample_jd):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    meta = result["metadata"]
    assert meta["total_experiences_searched"] > 0
    assert meta["model_used"] == "gpt-4o-mini"
    assert meta["query_time_seconds"] > 0
    assert meta["total_input_tokens"] > 0
    assert meta["total_output_tokens"] > 0
    assert meta["estimated_cost"] > 0


# ---------------------------------------------------------------------------
# Output quality
# ---------------------------------------------------------------------------


def test_match_and_generate_tailored_descriptions_nonempty(
    matcher_with_vectorstore, sample_jd
):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    for match in result["matched_experiences"]:
        td = match["tailored_description"]
        assert isinstance(td, str) and len(td.strip()) > 0
        assert td.count("-") >= 2 or td.count("•") >= 2, (
            f"No bullet points in tailored description for {match['experience']['title']}"
        )


def test_match_and_generate_fit_analysis_has_assessment(
    matcher_with_vectorstore, sample_jd
):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    fa = result["fit_analysis"].lower()
    assert any(w in fa for w in ["strong", "moderate", "weak"]), (
        "Fit analysis missing strength assessment"
    )


def test_match_and_generate_scores_ordered(matcher_with_vectorstore, sample_jd):
    result = matcher_with_vectorstore.match_and_generate(sample_jd)
    scores = [m["relevance_score"] for m in result["matched_experiences"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_match_and_generate_short_jd(matcher_with_vectorstore):
    result = matcher_with_vectorstore.match_and_generate("Data analyst Python")
    assert len(result["matched_experiences"]) == 3
    assert len(result["fit_analysis"]) > 20
