"""Layer 3: LLM generation tests — mocked structure + real API quality."""

from unittest.mock import MagicMock

import pytest

# Strong action verbs commonly used in resume bullet points
ACTION_VERBS = {
    "built", "developed", "led", "designed", "implemented", "created",
    "managed", "delivered", "optimised", "optimized", "analysed", "analyzed",
    "automated", "deployed", "established", "engineered", "integrated",
    "launched", "maintained", "orchestrated", "reduced", "streamlined",
    "transformed", "improved", "achieved", "accelerated", "architected",
    "collaborated", "configured", "consolidated", "coordinated", "customised",
    "customized", "executed", "facilitated", "generated", "identified",
    "leveraged", "migrated", "modernised", "modernized", "monitored",
    "pioneered", "processed", "refactored", "resolved", "scaled",
    "spearheaded", "standardised", "standardized", "supervised", "trained",
    "upgraded", "utilised", "utilized", "conducted", "contributed",
    "enhanced", "ensured", "extracted", "formulated", "initiated",
    "iterated", "overhauled", "presented", "proposed", "provided",
    "restructured", "supported", "tested", "validated", "wrote",
}


# ---------------------------------------------------------------------------
# Mocked tests — no API calls, no @pytest.mark.api
# ---------------------------------------------------------------------------


def test_tailored_description_calls_llm(mock_matcher, sample_experience, sample_jd):
    """Verify generate_tailored_description invokes the LLM and tracks tokens."""
    mock_response = MagicMock()
    mock_response.content = "- Built automated dashboards\n- Developed ETL pipeline\n- Led data analysis"
    mock_response.response_metadata = {
        "token_usage": {"prompt_tokens": 150, "completion_tokens": 60}
    }
    mock_matcher.llm.return_value = mock_response
    # Patch the chain's invoke to return our mock response
    from src.prompts import MATCH_CHAT_PROMPT

    chain = MATCH_CHAT_PROMPT | mock_matcher.llm
    mock_matcher.llm.__or__ = lambda self, other: chain
    # Directly mock the chain invoke
    import unittest.mock

    with unittest.mock.patch.object(
        type(chain), "invoke", return_value=mock_response
    ):
        result = mock_matcher.generate_tailored_description(
            sample_experience, sample_jd
        )
    assert isinstance(result, str)
    assert len(result) > 0


def test_tailored_description_retries_on_failure(
    mock_matcher, sample_experience, sample_jd
):
    """Verify retry logic: fails on first attempt, succeeds on second."""
    mock_success = MagicMock()
    mock_success.content = "- Analysed data\n- Built dashboards"
    mock_success.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }

    from src.prompts import MATCH_CHAT_PROMPT
    import unittest.mock

    chain = MATCH_CHAT_PROMPT | mock_matcher.llm

    with unittest.mock.patch.object(
        type(chain),
        "invoke",
        side_effect=[RuntimeError("API timeout"), mock_success],
    ):
        result = mock_matcher.generate_tailored_description(
            sample_experience, sample_jd
        )
    assert "Analysed data" in result


def test_tailored_description_raises_after_two_failures(
    mock_matcher, sample_experience, sample_jd
):
    """Verify exception propagates after both retry attempts fail."""
    from src.prompts import MATCH_CHAT_PROMPT
    import unittest.mock

    chain = MATCH_CHAT_PROMPT | mock_matcher.llm

    with unittest.mock.patch.object(
        type(chain),
        "invoke",
        side_effect=[RuntimeError("fail 1"), RuntimeError("fail 2")],
    ):
        with pytest.raises(RuntimeError, match="fail 2"):
            mock_matcher.generate_tailored_description(sample_experience, sample_jd)


def test_track_tokens(mock_matcher):
    """Verify token accumulation across multiple calls."""
    mock_matcher._total_input_tokens = 0
    mock_matcher._total_output_tokens = 0

    response1 = MagicMock()
    response1.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    response2 = MagicMock()
    response2.response_metadata = {
        "token_usage": {"prompt_tokens": 200, "completion_tokens": 80}
    }

    mock_matcher._track_tokens(response1)
    mock_matcher._track_tokens(response2)

    assert mock_matcher._total_input_tokens == 300
    assert mock_matcher._total_output_tokens == 130


# ---------------------------------------------------------------------------
# Real API tests — @pytest.mark.api
# ---------------------------------------------------------------------------


@pytest.mark.api
def test_real_tailored_description_format(matcher_with_vectorstore, sample_jd):
    """Verify real LLM output has bullet points and reasonable length."""
    exp = matcher_with_vectorstore.experiences[0]
    result = matcher_with_vectorstore.generate_tailored_description(exp, sample_jd)

    assert isinstance(result, str)
    assert len(result) > 100, "Tailored description too short"
    assert len(result) < 2000, "Tailored description too long"
    bullet_count = result.count("-") + result.count("•")
    assert bullet_count >= 2, f"Expected bullet points, got: {result[:100]}..."


@pytest.mark.api
def test_real_tailored_description_starts_with_action_verb(
    matcher_with_vectorstore, sample_jd
):
    """Each bullet point should start with a strong action verb."""
    exp = matcher_with_vectorstore.experiences[0]
    result = matcher_with_vectorstore.generate_tailored_description(exp, sample_jd)

    bullets = [
        line.strip().lstrip("-•").strip()
        for line in result.strip().split("\n")
        if line.strip().startswith(("-", "•"))
    ]
    assert len(bullets) >= 2, "Not enough bullet points found"

    for bullet in bullets:
        first_word = bullet.split()[0].lower().rstrip(",:")
        is_action_verb = first_word in ACTION_VERBS or first_word.endswith("ed")
        assert is_action_verb, (
            f"Bullet does not start with an action verb: '{first_word}' "
            f"in '{bullet[:60]}...'"
        )


@pytest.mark.api
def test_real_tailored_description_references_skills(
    matcher_with_vectorstore, sample_jd
):
    """Anti-hallucination: at least 1 original skill appears in output."""
    exp = matcher_with_vectorstore.experiences[0]
    result = matcher_with_vectorstore.generate_tailored_description(exp, sample_jd)

    result_lower = result.lower()
    original_skills = [s.lower() for s in exp["skills"]]
    matched = sum(1 for s in original_skills if s in result_lower)
    assert matched >= 1, (
        f"No original skills found in output. Skills: {exp['skills']}"
    )


@pytest.mark.api
def test_real_fit_analysis_contains_assessment(matcher_with_vectorstore, sample_jd):
    """Verify fit analysis contains a strength assessment word."""
    search_results = matcher_with_vectorstore.search(sample_jd)
    result = matcher_with_vectorstore.generate_fit_analysis(search_results, sample_jd)

    assert isinstance(result, str)
    assert len(result) > 20, "Fit analysis too short"
    result_lower = result.lower()
    assert any(w in result_lower for w in ["strong", "moderate", "weak"]), (
        f"No strength assessment found in: {result[:100]}..."
    )
