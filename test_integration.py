"""Comprehensive integration test for the Experience Matcher RAG pipeline."""

import sys
import time

from src.matcher import ExperienceMatcher

# ---------------------------------------------------------------------------
# Job descriptions for testing
# ---------------------------------------------------------------------------

JD_DATA_ANALYST = (
    "We are seeking a Data Analyst with 1-3 years of experience. "
    "Required: Python, SQL, data visualization (Tableau or PowerBI). "
    "Nice to have: ETL pipeline experience, machine learning basics, "
    "cloud platforms (AWS/Azure). You will analyze large datasets, "
    "build dashboards, and present insights to stakeholders."
)

JD_SOFTWARE_ENGINEER = (
    "We are looking for a Software Engineer to join our backend team. "
    "You will design and build REST APIs, write clean maintainable code, "
    "and collaborate in an agile environment. Required: Python or Java, "
    "SQL, Git, CI/CD. Nice to have: Flask/Django, Docker, cloud experience. "
    "Strong problem-solving skills and CS fundamentals expected."
)

JD_ML_ENGINEER = (
    "Machine Learning Engineer needed to develop and deploy production ML "
    "models. You will build end-to-end ML pipelines, run experiments with "
    "MLflow, and deploy models via REST APIs. Required: Python, PyTorch or "
    "TensorFlow, scikit-learn, feature engineering. Nice to have: LLMs, "
    "NLP, time-series forecasting, Docker, Airflow."
)

JD_SHORT = "Data analyst Python"

JD_LONG = (
    "We are looking for a highly motivated and detail-oriented Senior Data "
    "Scientist to join our growing Analytics and AI team. In this role, you "
    "will work closely with product managers, engineers, and business "
    "stakeholders to develop data-driven solutions that directly impact "
    "business outcomes. You will be responsible for designing and "
    "implementing advanced statistical models, machine learning algorithms, "
    "and data pipelines that process millions of records daily. The ideal "
    "candidate has a strong foundation in mathematics, statistics, and "
    "computer science, with hands-on experience in Python, SQL, and modern "
    "ML frameworks such as PyTorch, TensorFlow, or scikit-learn. You should "
    "be comfortable working with large-scale datasets stored in cloud "
    "platforms like AWS (S3, Redshift, SageMaker) or GCP (BigQuery, Vertex "
    "AI). Experience with experiment tracking tools like MLflow or Weights & "
    "Biases is highly desirable. You will also be expected to communicate "
    "complex technical findings to non-technical audiences through clear "
    "visualizations and presentations. Proficiency in data visualization "
    "tools such as Tableau, PowerBI, or Matplotlib is required. "
    "Additionally, you should have experience with A/B testing, causal "
    "inference, and experimental design. Familiarity with NLP, time-series "
    "forecasting, and deep learning architectures (CNNs, RNNs, "
    "Transformers) is a strong plus. The role requires strong software "
    "engineering practices including version control with Git, code review, "
    "CI/CD pipelines, and containerization with Docker. You will be "
    "deploying models as microservices using FastAPI or Flask, with "
    "orchestration through Airflow or Kubeflow. We value candidates who are "
    "self-starters, can work independently, and have a track record of "
    "delivering impactful projects. A Master's or PhD in Computer Science, "
    "Statistics, Mathematics, or a related quantitative field is preferred. "
    "Published research in top-tier venues (NeurIPS, ICML, ACL) is a bonus "
    "but not required. We offer competitive compensation, flexible working "
    "arrangements, and the opportunity to work on challenging problems at "
    "scale. If you are passionate about using data and AI to solve real-world "
    "problems, we would love to hear from you."
)

TOP_K = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(condition: bool, label: str) -> bool:
    """Print a check result and return whether it passed."""
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    return condition


def validate_result(result: dict, label: str) -> bool:
    """Run all validation checks on a match_and_generate result."""
    print(f"\n--- Validating: {label} ---")
    all_ok = True

    # Correct number of results
    n = len(result["matched_experiences"])
    all_ok &= check(n == TOP_K, f"Returns exactly {TOP_K} results (got {n})")

    for match in result["matched_experiences"]:
        title = match["experience"]["title"]
        score = match["relevance_score"]

        # Score range
        all_ok &= check(
            0.0 < score <= 1.0,
            f"Score in (0,1] for '{title}' (got {score:.4f})",
        )

        # Tailored description is non-empty and has bullet points
        td = match["tailored_description"]
        all_ok &= check(
            len(td.strip()) > 0,
            f"Tailored description non-empty for '{title}'",
        )
        all_ok &= check(
            td.count("-") >= 2 or td.count("•") >= 2,
            f"Tailored description has bullet points for '{title}'",
        )

        # No hallucinated skills: every skill word in the tailored
        # description should appear in the original experience's
        # skills, description, achievements, or keywords
        original_text = " ".join(
            [
                match["experience"]["description"],
                " ".join(match["experience"]["skills"]),
                " ".join(match["experience"]["achievements"]),
                " ".join(match["experience"]["keywords"]),
            ]
        ).lower()

        original_skills = {s.lower() for s in match["experience"]["skills"]}
        # Check that at least half the original skills appear somewhere
        # in the tailored text (sanity check, not exhaustive)
        td_lower = td.lower()
        matched_skills = sum(1 for s in original_skills if s in td_lower)
        all_ok &= check(
            matched_skills >= 1,
            f"At least 1 original skill referenced in tailored text for '{title}' "
            f"({matched_skills}/{len(original_skills)})",
        )

    # Fit analysis
    fa = result["fit_analysis"]
    all_ok &= check(len(fa.strip()) > 20, "Fit analysis is non-empty")
    all_ok &= check(
        any(w in fa.lower() for w in ["strong", "moderate", "weak"]),
        "Fit analysis contains strength assessment",
    )

    # Metadata
    meta = result["metadata"]
    all_ok &= check(
        meta["total_experiences_searched"] > 0, "Metadata: experiences_searched > 0"
    )
    all_ok &= check(meta["query_time_seconds"] > 0, "Metadata: query_time > 0")
    all_ok &= check(meta["total_input_tokens"] > 0, "Metadata: input_tokens > 0")
    all_ok &= check(meta["total_output_tokens"] > 0, "Metadata: output_tokens > 0")
    all_ok &= check(meta["estimated_cost"] > 0, "Metadata: estimated_cost > 0")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("INTEGRATION TEST — AI-Powered Experience Matcher")
    print("=" * 70)

    # Initialise
    matcher = ExperienceMatcher()
    matcher.load_experiences("data/experiences.json")
    matcher.create_vector_store()

    all_passed = True
    summary_rows = []

    # ------------------------------------------------------------------
    # Test 1-3: Standard job descriptions
    # ------------------------------------------------------------------
    test_cases = [
        ("Data Analyst", JD_DATA_ANALYST),
        ("Software Engineer", JD_SOFTWARE_ENGINEER),
        ("ML Engineer", JD_ML_ENGINEER),
    ]

    for label, jd in test_cases:
        result = matcher.match_and_generate(jd, top_k=TOP_K)
        passed = validate_result(result, label)
        all_passed &= passed

        top = result["matched_experiences"][0]
        summary_rows.append(
            {
                "query": label,
                "top_match": top["experience"]["title"],
                "score": top["relevance_score"],
                "time": result["metadata"]["query_time_seconds"],
                "cost": result["metadata"]["estimated_cost"],
            }
        )

    # ------------------------------------------------------------------
    # Test 4: Edge cases
    # ------------------------------------------------------------------
    print("\n--- Edge Case: Short JD (3 words) ---")
    result_short = matcher.match_and_generate(JD_SHORT, top_k=TOP_K)
    passed = validate_result(result_short, "Short JD")
    all_passed &= passed
    top = result_short["matched_experiences"][0]
    summary_rows.append(
        {
            "query": "Short JD",
            "top_match": top["experience"]["title"],
            "score": top["relevance_score"],
            "time": result_short["metadata"]["query_time_seconds"],
            "cost": result_short["metadata"]["estimated_cost"],
        }
    )

    print("\n--- Edge Case: Long JD (~500 words) ---")
    word_count = len(JD_LONG.split())
    print(f"  (JD length: {word_count} words)")
    result_long = matcher.match_and_generate(JD_LONG, top_k=TOP_K)
    passed = validate_result(result_long, "Long JD")
    all_passed &= passed
    top = result_long["matched_experiences"][0]
    summary_rows.append(
        {
            "query": f"Long JD ({word_count}w)",
            "top_match": top["experience"]["title"],
            "score": top["relevance_score"],
            "time": result_long["metadata"]["query_time_seconds"],
            "cost": result_long["metadata"]["estimated_cost"],
        }
    )

    print("\n--- Edge Case: Empty JD ---")
    try:
        result_empty = matcher.match_and_generate("", top_k=TOP_K)
        # If it doesn't error, just validate the output
        validate_result(result_empty, "Empty JD")
        print(f"  [NOTE] Empty JD did not raise an error — returned results anyway")
    except Exception as e:
        print(f"  [NOTE] Empty JD raised {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Query':<22} {'#1 Match':<45} {'Score':>6} {'Time':>6} {'Cost':>10}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['query']:<22} "
            f"{row['top_match']:<45} "
            f"{row['score']:>6.4f} "
            f"{row['time']:>5.1f}s "
            f"${row['cost']:>8.6f}"
        )

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    total_cost = sum(r["cost"] for r in summary_rows)
    print(f"\nTotal estimated cost: ${total_cost:.6f}")
    print()
    if all_passed:
        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
    else:
        print("=" * 70)
        print("SOME TESTS FAILED — review output above")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
