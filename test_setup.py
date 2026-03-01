import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED_FIELDS = [
    "id",
    "title",
    "company",
    "duration",
    "description",
    "skills",
    "achievements",
    "keywords",
    "category",
]


def main():
    errors = []

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        errors.append("OPENAI_API_KEY not set in .env")
    else:
        print(f"OpenAI API key found: {api_key[:8]}...{api_key[-4:]}")

        # Test API connectivity
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hello' in one word."}],
                max_tokens=5,
            )
            print(f"API test successful: {response.choices[0].message.content.strip()}")
        except Exception as e:
            errors.append(f"OpenAI API call failed: {e}")

    # Load and validate experiences.json
    try:
        with open("data/experiences.json", "r") as f:
            experiences = json.load(f)

        for exp in experiences:
            missing = [field for field in REQUIRED_FIELDS if field not in exp]
            if missing:
                errors.append(
                    f"Experience '{exp.get('id', '?')}' missing fields: {missing}"
                )

        print(f"Loaded {len(experiences)} experiences from data/experiences.json")
    except Exception as e:
        errors.append(f"Failed to load experiences.json: {e}")

    # Summary
    if errors:
        print("\nErrors found:")
        for err in errors:
            print(f"  ❌ {err}")
        sys.exit(1)
    else:
        print(f"\n✅ Setup complete! {len(experiences)} experiences loaded.")


if __name__ == "__main__":
    main()
