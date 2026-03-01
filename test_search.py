"""Test script for ExperienceMatcher vector search."""

from src.matcher import ExperienceMatcher


def main():
    matcher = ExperienceMatcher()
    experiences = matcher.load_experiences("data/experiences.json")
    print(f"\nLoaded {len(experiences)} experiences\n")

    print("Creating vector store...")
    matcher.create_vector_store()
    print("Vector store created!\n")

    queries = [
        "Python data analyst with SQL and ETL experience",
        "Machine learning engineer with predictive modeling",
        "Full-stack developer with API experience",
    ]

    print("=" * 70)
    for query in queries:
        print(f'\nQuery: "{query}"')
        print("-" * 70)
        results = matcher.search(query, top_k=3)
        for r in results:
            exp = r["experience"]
            print(
                f"  #{r['rank']}  {exp['title']} @ {exp['company']}"
                f"  (score: {r['relevance_score']:.4f})"
            )
        print()


if __name__ == "__main__":
    main()
