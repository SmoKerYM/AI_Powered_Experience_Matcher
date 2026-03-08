"""Layer 1: Data sanity checks — zero API calls."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.config import EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RESULTS
from src.matcher import REQUIRED_FIELDS
from src.prompts import ANALYSIS_CHAT_PROMPT, MATCH_CHAT_PROMPT

# ---------------------------------------------------------------------------
# Load experiences once for parametrized tests
# ---------------------------------------------------------------------------

EXPERIENCES_PATH = Path("data/experiences.json")

with open(EXPERIENCES_PATH, "r", encoding="utf-8") as _f:
    _ALL_EXPERIENCES = json.load(_f)


# ---------------------------------------------------------------------------
# File-level tests
# ---------------------------------------------------------------------------


def test_experiences_file_exists():
    """The experiences JSON file exists on disk."""
    assert EXPERIENCES_PATH.exists()


def test_experiences_is_valid_json():
    """The file parses as a valid JSON list."""
    assert isinstance(_ALL_EXPERIENCES, list)


def test_experiences_count():
    """There are exactly 14 experiences in the dataset."""
    assert len(_ALL_EXPERIENCES) == 14


# ---------------------------------------------------------------------------
# Per-experience validation (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "experience",
    _ALL_EXPERIENCES,
    ids=[exp["id"] for exp in _ALL_EXPERIENCES],
)
def test_all_required_fields_present(experience):
    """Every experience has all 9 required fields."""
    for field in REQUIRED_FIELDS:
        assert field in experience, f"{experience['id']} missing field: {field}"


def test_unique_ids():
    """No duplicate experience IDs exist."""
    ids = [exp["id"] for exp in _ALL_EXPERIENCES]
    assert len(ids) == len(set(ids)), "Duplicate experience IDs found"


def test_skills_are_lists():
    """Skills, achievements, and keywords are all list types."""
    for exp in _ALL_EXPERIENCES:
        assert isinstance(exp["skills"], list), f"{exp['id']}: skills is not a list"
        assert isinstance(exp["achievements"], list), f"{exp['id']}: achievements is not a list"
        assert isinstance(exp["keywords"], list), f"{exp['id']}: keywords is not a list"


def test_nonempty_text_fields():
    """Title, company, duration, and description are non-empty strings."""
    for exp in _ALL_EXPERIENCES:
        for field in ("title", "company", "duration", "description"):
            assert isinstance(exp[field], str) and len(exp[field].strip()) > 0, (
                f"{exp['id']}: {field} is empty or not a string"
            )


# ---------------------------------------------------------------------------
# Matcher data-loading tests (mocked, no API)
# ---------------------------------------------------------------------------


def test_load_experiences_returns_correct_count(mock_matcher):
    """load_experiences() returns all 14 experiences from the JSON file."""
    result = mock_matcher.load_experiences("data/experiences.json")
    assert len(result) == 14


def test_load_experiences_missing_file(mock_matcher):
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        mock_matcher.load_experiences("nonexistent_file.json")


def test_experience_to_document(mock_matcher, sample_experience):
    """Conversion produces a LangChain Document with correct content and metadata."""
    doc = mock_matcher._experience_to_document(sample_experience)
    assert isinstance(doc, Document)
    assert sample_experience["title"] in doc.page_content
    assert sample_experience["company"] in doc.page_content
    for key in ("id", "title", "company", "category", "skills"):
        assert key in doc.metadata


# ---------------------------------------------------------------------------
# Config & prompt validation
# ---------------------------------------------------------------------------


def test_config_values():
    """Config constants match expected model names and parameters."""
    assert EMBEDDING_MODEL == "text-embedding-3-small"
    assert LLM_MODEL == "gpt-4o-mini"
    assert LLM_TEMPERATURE == 0.3
    assert TOP_K_RESULTS == 3


def test_prompt_templates_defined():
    """Both prompt templates are valid ChatPromptTemplate instances."""
    assert isinstance(MATCH_CHAT_PROMPT, ChatPromptTemplate)
    assert isinstance(ANALYSIS_CHAT_PROMPT, ChatPromptTemplate)
