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
#
# Search + generate tailored descriptions once, then validate the results.
# The Zendo experience (exp_001) is known to appear in the top 3 for the
# sample Data Analyst JD.
# ---------------------------------------------------------------------------


_SAMPLE_JD = (
    "We are seeking a Data Analyst with 1-3 years of experience. "
    "Required: Python, SQL, data visualization (Tableau or PowerBI). "
    "Nice to have: ETL pipeline experience, machine learning basics, "
    "cloud platforms (AWS/Azure). You will analyze large datasets, "
    "build dashboards, and present insights to stakeholders."
)


@pytest.fixture(scope="module")
def generation_results(matcher_with_vectorstore):
    """Search and generate tailored descriptions once for all generation tests."""
    search_results = matcher_with_vectorstore.search(_SAMPLE_JD)
    tailored = {}
    for r in search_results:
        exp = r["experience"]
        desc = matcher_with_vectorstore.generate_tailored_description(exp, _SAMPLE_JD)
        tailored[exp["company"]] = {
            "experience": exp,
            "tailored_description": desc,
        }
    return {
        "search_results": search_results,
        "tailored": tailored,
    }


@pytest.mark.api
def test_zendo_in_search_results(generation_results):
    """Zendo Technologies should appear in the top 3 for the Data Analyst JD."""
    assert "Zendo Technologies" in generation_results["tailored"], (
        f"Zendo not in results. Got: {list(generation_results['tailored'].keys())}"
    )


@pytest.mark.api
def test_zendo_bullet_count(generation_results):
    """Zendo tailored description should have 3-4 bullet points."""
    desc = generation_results["tailored"]["Zendo Technologies"]["tailored_description"]
    bullets = [
        line.strip()
        for line in desc.strip().split("\n")
        if line.strip().startswith(("-", "•"))
    ]
    assert 3 <= len(bullets) <= 4, (
        f"Expected 3-4 bullet points, got {len(bullets)}"
    )


@pytest.mark.api
def test_zendo_bullet_length(generation_results):
    """Each bullet point should be 1-2 lines (30-200 chars)."""
    desc = generation_results["tailored"]["Zendo Technologies"]["tailored_description"]
    bullets = [
        line.strip().lstrip("-•").strip()
        for line in desc.strip().split("\n")
        if line.strip().startswith(("-", "•"))
    ]
    for bullet in bullets:
        assert 30 <= len(bullet) <= 200, (
            f"Bullet length {len(bullet)} out of range: '{bullet[:60]}...'"
        )


@pytest.mark.api
def test_zendo_bullets_start_with_action_verb(generation_results):
    """Each Zendo bullet should start with a strong action verb."""
    desc = generation_results["tailored"]["Zendo Technologies"]["tailored_description"]
    bullets = [
        line.strip().lstrip("-•").strip()
        for line in desc.strip().split("\n")
        if line.strip().startswith(("-", "•"))
    ]
    for bullet in bullets:
        first_word = bullet.split()[0].lower().rstrip(",:")
        is_action_verb = first_word in ACTION_VERBS or first_word.endswith("ed")
        assert is_action_verb, (
            f"Bullet does not start with an action verb: '{first_word}' "
            f"in '{bullet[:60]}...'"
        )


@pytest.mark.api
def test_zendo_references_jd_keywords(generation_results):
    """At least one keyword from the JD should appear in the tailored output."""
    desc = generation_results["tailored"]["Zendo Technologies"]["tailored_description"].lower()
    jd_keywords = ["python", "sql", "data", "tableau", "etl", "dashboard", "visualization"]
    matched = [kw for kw in jd_keywords if kw in desc]
    assert len(matched) >= 1, (
        f"No JD keywords found in output. Checked: {jd_keywords}"
    )


@pytest.mark.api
def test_zendo_references_original_skills(generation_results):
    """Anti-hallucination: at least 1 original Zendo skill appears in output."""
    entry = generation_results["tailored"]["Zendo Technologies"]
    desc_lower = entry["tailored_description"].lower()
    original_skills = [s.lower() for s in entry["experience"]["skills"]]
    matched = sum(1 for s in original_skills if s in desc_lower)
    assert matched >= 1, (
        f"No original skills found in output. Skills: {entry['experience']['skills']}"
    )


@pytest.mark.api
def test_real_fit_analysis_contains_assessment(matcher_with_vectorstore):
    """Verify fit analysis contains a strength assessment word."""
    search_results = matcher_with_vectorstore.search(_SAMPLE_JD)
    result = matcher_with_vectorstore.generate_fit_analysis(search_results, _SAMPLE_JD)

    assert isinstance(result, str)
    assert len(result) > 20, "Fit analysis too short"
    result_lower = result.lower()
    assert any(w in result_lower for w in ["strong", "moderate", "weak"]), (
        f"No strength assessment found in: {result[:100]}..."
    )


@pytest.mark.api
def test_llm_as_judge_relevance_and_specificity(generation_results):
    """LLM-as-a-judge: GPT-4o-mini scores the best-match tailored output >= 3/5."""
    # Find the best-match experience (first in search results)
    best_company = generation_results["search_results"][0]["experience"]["company"]
    best_entry = generation_results["tailored"][best_company]
    tailored_desc = best_entry["tailored_description"]

    judge_prompt = f"""You are an expert resume reviewer. Score the following tailored bullet points
on two criteria, given the job description they were written for.

Job Description:
{_SAMPLE_JD}

Tailored Bullet Points:
{tailored_desc}

Score each criterion from 1 (poor) to 5 (excellent):
1. Relevance: How well do the bullet points address the specific requirements in the JD?
2. Specificity: Do the bullet points include concrete details, metrics, and action verbs rather than vague claims?

Respond with ONLY two lines in this exact format:
Relevance: <integer_score>
Specificity: <integer_score>"""

    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)
    response = judge.invoke(judge_prompt)
    output = response.content.strip()
    print(f"LLM Judge Output:\n{output}")

    # Parse scores
    scores = {}
    for line in output.split("\n"):
        line = line.strip()
        for criterion in ("Relevance", "Specificity"):
            if line.startswith(criterion):
                score_str = line.split(":")[-1].strip()
                # Handle formats like "4", "4/5", "4.0"
                score_str = score_str.split("/")[0].strip()
                scores[criterion] = float(score_str)

    assert "Relevance" in scores, f"Could not parse Relevance score from: {output}"
    assert "Specificity" in scores, f"Could not parse Specificity score from: {output}"
    assert scores["Relevance"] >= 3, (
        f"Relevance score {scores['Relevance']}/5 below threshold. Output: {tailored_desc[:100]}..."
    )
    assert scores["Specificity"] >= 3, (
        f"Specificity score {scores['Specificity']}/5 below threshold. Output: {tailored_desc[:100]}..."
    )
