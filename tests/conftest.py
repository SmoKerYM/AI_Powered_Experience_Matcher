"""Shared fixtures for the Experience Matcher test suite."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.matcher import ExperienceMatcher


# ---------------------------------------------------------------------------
# Auto-skip for API-dependent tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def skip_without_api_key(request):
    """Skip tests marked with @pytest.mark.api when OPENAI_API_KEY is absent."""
    if request.node.get_closest_marker("api"):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_experience():
    """A single hardcoded experience dict with all required fields."""
    return {
        "id": "exp_test",
        "title": "Data Analyst Intern",
        "company": "Test Corp",
        "duration": "Jan 2024 - Jun 2024",
        "description": "Analysed sales data using Python and SQL. Built Tableau dashboards for stakeholders.",
        "skills": ["Python", "SQL", "Tableau", "pandas", "Excel"],
        "achievements": [
            "Reduced reporting time by 40% through automated dashboards",
            "Processed 500K+ records daily using pandas ETL pipeline",
        ],
        "keywords": ["data analysis", "dashboards", "ETL", "reporting"],
        "category": "work",
    }


@pytest.fixture
def sample_jd():
    """Data Analyst job description for testing."""
    return (
        "We are seeking a Data Analyst with 1-3 years of experience. "
        "Required: Python, SQL, data visualization (Tableau or PowerBI). "
        "Nice to have: ETL pipeline experience, machine learning basics, "
        "cloud platforms (AWS/Azure). You will analyze large datasets, "
        "build dashboards, and present insights to stakeholders."
    )


@pytest.fixture
def sample_jd_software_engineer():
    """Software Engineer job description for testing."""
    return (
        "We are looking for a Software Engineer to join our backend team. "
        "You will design and build REST APIs, write clean maintainable code, "
        "and collaborate in an agile environment. Required: Python or Java, "
        "SQL, Git, CI/CD. Nice to have: Flask/Django, Docker, cloud experience. "
        "Strong problem-solving skills and CS fundamentals expected."
    )


@pytest.fixture
def experiences_json_path():
    """Path to the real experiences JSON file."""
    return "data/experiences.json"


# ---------------------------------------------------------------------------
# Matcher fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def matcher_with_vectorstore():
    """Session-scoped matcher with real embeddings and FAISS vector store.

    Built once per test session to minimise embedding API calls.
    """
    matcher = ExperienceMatcher()
    matcher.load_experiences("data/experiences.json")
    matcher.create_vector_store()
    return matcher


@pytest.fixture
def mock_matcher():
    """Matcher with mocked OpenAI clients — no API calls made.

    Useful for testing data loading, document conversion, and structure
    validation without any API cost.
    """
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key-for-testing"}),
        patch("src.matcher.OpenAIEmbeddings") as mock_embeddings,
        patch("src.matcher.ChatOpenAI") as mock_llm,
    ):
        mock_embeddings.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        matcher = ExperienceMatcher()
        matcher.load_experiences("data/experiences.json")
        yield matcher
