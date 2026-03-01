"""Test script for LLM-powered experience matching."""

from src.matcher import ExperienceMatcher

JOB_DESCRIPTION = """\
We are seeking a Data Analyst with 1-3 years of experience. \
Required: Python, SQL, data visualization (Tableau or PowerBI). \
Nice to have: ETL pipeline experience, machine learning basics, \
cloud platforms (AWS/Azure). You will analyze large datasets, \
build dashboards, and present insights to stakeholders."""


def main():
    matcher = ExperienceMatcher()
    matcher.load_experiences("data/experiences.json")
    matcher.load_vector_store()

    print("=" * 70)
    print("JOB DESCRIPTION:")
    print(JOB_DESCRIPTION)
    print("=" * 70)

    result = matcher.match_and_generate(JOB_DESCRIPTION)

    for match in result["matched_experiences"]:
        exp = match["experience"]
        print(f"\n{'─' * 70}")
        print(
            f"#{match['rank']}  {exp['title']} @ {exp['company']}  "
            f"(score: {match['relevance_score']:.4f})"
        )
        print(f"{'─' * 70}")

        print("\nORIGINAL:")
        print(
            exp["description"][:300] + "..."
            if len(exp["description"]) > 300
            else exp["description"]
        )

        print("\nTAILORED:")
        print(match["tailored_description"])

    print(f"\n{'=' * 70}")
    print("FIT ANALYSIS:")
    print(result["fit_analysis"])

    print(f"\n{'=' * 70}")
    print("METADATA:")
    meta = result["metadata"]
    print(f"  Experiences searched: {meta['total_experiences_searched']}")
    print(f"  Model: {meta['model_used']}")
    print(f"  Time: {meta['query_time_seconds']}s")
    print(
        f"  Tokens: {meta['total_input_tokens']} in / {meta['total_output_tokens']} out"
    )
    print(f"  Estimated cost: ${meta['estimated_cost']:.6f}")


if __name__ == "__main__":
    main()
