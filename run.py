"""CLI tool to match experiences against a job description."""

import argparse
import sys

from src.matcher import ExperienceMatcher


def main():
    parser = argparse.ArgumentParser(
        description="Match your experiences to a job description using RAG."
    )
    parser.add_argument(
        "job_description",
        nargs="?",
        help="Job description text. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=3, help="Number of matches to return (default: 3)"
    )
    args = parser.parse_args()

    # Get job description from arg or stdin
    if args.job_description:
        jd = args.job_description
    elif not sys.stdin.isatty():
        jd = sys.stdin.read().strip()
    else:
        print("Paste your job description below (press Ctrl+D when done):\n")
        jd = sys.stdin.read().strip()

    if not jd:
        print("Error: No job description provided.")
        sys.exit(1)

    # Run pipeline
    matcher = ExperienceMatcher()
    matcher.load_experiences("data/experiences.json")
    matcher.load_vector_store()

    print(f"\n{'=' * 70}")
    print("JOB DESCRIPTION:")
    print(jd[:300] + "..." if len(jd) > 300 else jd)
    print(f"{'=' * 70}")

    result = matcher.match_and_generate(jd, top_k=args.top_k)

    for match in result["matched_experiences"]:
        exp = match["experience"]
        print(f"\n{'─' * 70}")
        print(
            f"#{match['rank']}  {exp['title']} @ {exp['company']}  "
            f"(score: {match['relevance_score']:.4f})"
        )
        print(f"{'─' * 70}")
        print(match["tailored_description"])

    print(f"\n{'=' * 70}")
    print("FIT ANALYSIS:")
    print(result["fit_analysis"])

    meta = result["metadata"]
    print(f"\n--- {meta['query_time_seconds']}s | "
          f"{meta['total_input_tokens']}+{meta['total_output_tokens']} tokens | "
          f"${meta['estimated_cost']:.4f} ---")


if __name__ == "__main__":
    main()
