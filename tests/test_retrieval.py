"""Layer 2: Retrieval quality — requires OpenAI embedding API."""

import pytest

pytestmark = pytest.mark.api


@pytest.fixture(scope="module")
def data_analyst_search_results(matcher_with_vectorstore):
    """Run the data analyst search query once and reuse across tests."""
    return matcher_with_vectorstore.search("data analyst Python SQL", top_k=3)


# ---------------------------------------------------------------------------
# Basic search behaviour (reuse single search call)
# ---------------------------------------------------------------------------


def test_search_returns_correct_count(data_analyst_search_results):
    """Search returns exactly 3 results for top_k=3."""
    assert len(data_analyst_search_results) == 3


def test_search_result_structure(data_analyst_search_results):
    """Each result dict has experience, relevance_score, and rank keys."""
    for r in data_analyst_search_results:
        assert "experience" in r
        assert "relevance_score" in r
        assert "rank" in r


def test_search_scores_in_range(data_analyst_search_results):
    """All relevance scores fall within (0.0, 1.0]."""
    for r in data_analyst_search_results:
        assert 0.0 < r["relevance_score"] <= 1.0, (
            f"Score {r['relevance_score']} out of range"
        )


def test_search_results_ranked_descending(data_analyst_search_results):
    """Scores are in descending order and ranks are 1, 2, 3."""
    scores = [r["relevance_score"] for r in data_analyst_search_results]
    assert scores == sorted(scores, reverse=True)
    ranks = [r["rank"] for r in data_analyst_search_results]
    assert ranks == [1, 2, 3]


def test_search_experience_has_required_fields(data_analyst_search_results):
    """Each returned experience contains all 9 required fields."""
    required = {"id", "title", "company", "duration", "description", "skills", "achievements", "keywords", "category"}
    for r in data_analyst_search_results:
        assert required.issubset(r["experience"].keys())


# ---------------------------------------------------------------------------
# Relevance tests — known queries with expected top results
# ---------------------------------------------------------------------------


def test_search_relevance_data_analyst(matcher_with_vectorstore):
    """A data analyst query surfaces a data-related experience as the top result."""
    results = matcher_with_vectorstore.search("Python data analyst SQL ETL Tableau")
    top_title = results[0]["experience"]["title"].lower()
    assert "data" in top_title or "analyst" in top_title, (
        f"Expected data-related top result, got: {results[0]['experience']['title']}"
    )


def test_search_relevance_ml_engineer(matcher_with_vectorstore):
    """An ML query surfaces an ML/DL experience as the top result."""
    results = matcher_with_vectorstore.search(
        "machine learning PyTorch deep learning model training"
    )
    top_keywords = " ".join(results[0]["experience"]["keywords"]).lower()
    top_skills = " ".join(results[0]["experience"]["skills"]).lower()
    combined = top_keywords + " " + top_skills
    assert any(term in combined for term in ["machine learning", "pytorch", "deep learning", "ml"]), (
        f"Expected ML-related top result, got: {results[0]['experience']['title']}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_search_top_k_parameter(matcher_with_vectorstore):
    """top_k=1 returns 1 result and top_k=5 returns 5 results."""
    results_1 = matcher_with_vectorstore.search("Python developer", top_k=1)
    assert len(results_1) == 1

    results_5 = matcher_with_vectorstore.search("Python developer", top_k=5)
    assert len(results_5) == 5


def test_search_short_query(matcher_with_vectorstore):
    """A single-word query returns results without errors."""
    results = matcher_with_vectorstore.search("Python")
    assert len(results) == 3
    for r in results:
        assert r["relevance_score"] > 0
