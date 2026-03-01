"""Core RAG logic for the AI-Powered Experience Matcher."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K_RESULTS,
    VECTOR_STORE_PATH,
)
from src.prompts import ANALYSIS_CHAT_PROMPT, MATCH_CHAT_PROMPT

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

# Approximate pricing for gpt-4o-mini (per 1M tokens)
_INPUT_COST_PER_M = 0.15
_OUTPUT_COST_PER_M = 0.60


class ExperienceMatcher:
    """Manages experience embeddings, semantic search, and LLM generation."""

    def __init__(self) -> None:
        """Load API key, initialise embeddings and LLM, set up empty state."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL, openai_api_key=api_key
        )
        self.vectorstore: Optional[FAISS] = None
        self.experiences: list[dict] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Also init llm model
        self._init_llm()
        logger.info(
            "ExperienceMatcher initialised (embedding=%s, llm=%s)",
            EMBEDDING_MODEL,
            LLM_MODEL,
        )

    def _init_llm(self) -> None:
        """Initialise the ChatOpenAI LLM from config settings."""
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=500,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_experiences(self, json_path: str) -> list[dict]:
        """Load and validate experiences from a JSON file.

        Args:
            json_path: Path to the experiences JSON file.

        Returns:
            List of validated experience dicts.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Experiences file not found: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for exp in data:
            missing = [field for field in REQUIRED_FIELDS if field not in exp]
            if missing:
                raise ValueError(
                    f"Experience '{exp.get('id', '?')}' missing fields: {missing}"
                )

        self.experiences = data

        categories: dict[str, int] = {}
        for exp in data:
            cat = exp["category"]
            categories[cat] = categories.get(cat, 0) + 1

        logger.info(
            "Loaded %d experiences: %s",
            len(data),
            ", ".join(f"{v} {k}" for k, v in categories.items()),
        )
        return data

    # ------------------------------------------------------------------
    # Document conversion & vector store
    # ------------------------------------------------------------------

    def _experience_to_document(self, exp: dict) -> Document:
        """Convert a single experience dict into a LangChain Document.

        Combines title, description, skills, and achievements into rich
        text for embedding. Stores key fields as metadata for retrieval.
        """
        skills_str = ", ".join(exp["skills"])
        achievements_str = "\n".join(f"- {a}" for a in exp["achievements"])
        keywords_str = ", ".join(exp["keywords"])

        page_content = (
            f"Title: {exp['title']} at {exp['company']}\n"
            f"Duration: {exp['duration']}\n"
            f"Category: {exp['category']}\n\n"
            f"Description:\n{exp['description']}\n\n"
            f"Skills: {skills_str}\n\n"
            f"Achievements:\n{achievements_str}\n\n"
            f"Keywords: {keywords_str}"
        )

        metadata = {
            "id": exp["id"],
            "title": exp["title"],
            "company": exp["company"],
            "category": exp["category"],
            "skills": skills_str,
        }

        return Document(page_content=page_content, metadata=metadata)

    def create_vector_store(self) -> FAISS:
        """Create a FAISS vector store from all loaded experiences.

        Returns:
            The created FAISS vectorstore.
        """
        if not self.experiences:
            raise ValueError("No experiences loaded. Call load_experiences() first.")

        documents = [self._experience_to_document(exp) for exp in self.experiences]
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save_local(VECTOR_STORE_PATH)

        logger.info(
            "Created vector store with %d vectors, saved to %s",
            len(documents),
            VECTOR_STORE_PATH,
        )
        return self.vectorstore

    def load_vector_store(self) -> FAISS:
        """Load a previously saved FAISS vector store from disk.

        Falls back to creating a new store if none exists.

        Returns:
            The loaded (or newly created) FAISS vectorstore.
        """
        try:
            self.vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded vector store from %s", VECTOR_STORE_PATH)
        except (FileNotFoundError, RuntimeError):
            logger.warning(
                "No vector store found at %s — creating new one", VECTOR_STORE_PATH
            )
            if not self.experiences:
                self.load_experiences("data/experiences.json")
            self.create_vector_store()

        return self.vectorstore

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Run semantic similarity search against the vector store.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: experience, relevance_score, rank.
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store not initialised. Call create_vector_store() or load_vector_store() first."
            )

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        exp_lookup = {exp["id"]: exp for exp in self.experiences}

        output = []
        for doc, distance in results:
            exp_id = doc.metadata["id"]
            relevance = 1.0 / (1.0 + distance)
            full_exp = exp_lookup.get(exp_id, doc.metadata)
            output.append(
                {
                    "experience": full_exp,
                    "relevance_score": round(relevance, 4),
                }
            )

        output.sort(key=lambda x: x["relevance_score"], reverse=True)
        for i, item in enumerate(output, 1):
            item["rank"] = i

        return output

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def _track_tokens(self, response) -> None:
        """Accumulate token usage from an LLM response."""
        usage = response.response_metadata.get("token_usage", {})
        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)

    def generate_tailored_description(
        self, experience: dict, job_description: str
    ) -> str:
        """Generate a tailored description for one experience using the LLM.

        Args:
            experience: Full experience dict from experiences.json.
            job_description: The target job description text.

        Returns:
            LLM-generated bullet points tailored to the job.
        """
        chain = MATCH_CHAT_PROMPT | self.llm

        invoke_kwargs = {
            "job_description": job_description,
            "title": experience["title"],
            "company": experience["company"],
            "duration": experience["duration"],
            "description": experience["description"],
            "skills": ", ".join(experience["skills"]),
            "achievements": "\n".join(f"- {a}" for a in experience["achievements"]),
        }

        # Try up to 2 times
        for attempt in range(2):
            try:
                response = chain.invoke(invoke_kwargs)
                self._track_tokens(response)
                logger.info(
                    "Generated tailored description for '%s' (attempt %d)",
                    experience["title"],
                    attempt + 1,
                )
                return response.content
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        "LLM call failed for '%s', retrying: %s", experience["title"], e
                    )
                else:
                    logger.error(
                        "LLM call failed for '%s' after retry: %s",
                        experience["title"],
                        e,
                    )
                    raise
        return ""  # unreachable, but satisfies type checker

    def generate_fit_analysis(
        self, matched_results: list[dict], job_description: str
    ) -> str:
        """Generate an overall fit analysis from matched search results.

        Args:
            matched_results: Output from search() — list of ranked matches.
            job_description: The target job description text.

        Returns:
            LLM-generated fit analysis string.
        """
        summary_parts = []
        for r in matched_results:
            exp = r["experience"]
            summary_parts.append(
                f"#{r['rank']} (score: {r['relevance_score']:.2f}) — "
                f"{exp['title']} at {exp['company']}\n"
                f"   Skills: {', '.join(exp['skills'])}\n"
                f"   Achievements: {'; '.join(exp['achievements'])}"
            )

        chain = ANALYSIS_CHAT_PROMPT | self.llm
        response = chain.invoke(
            {
                "job_description": job_description,
                "experiences_summary": "\n\n".join(summary_parts),
            }
        )
        self._track_tokens(response)
        return response.content

    def match_and_generate(
        self, job_description: str, top_k: int = TOP_K_RESULTS
    ) -> dict:
        """Main entry point: search, tailor descriptions, and analyse fit.

        Args:
            job_description: The target job description text.
            top_k: Number of experiences to match.

        Returns:
            Structured dict with matched_experiences, fit_analysis, and metadata.
        """
        start = time.time()
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Step 1: Semantic search
        search_results = self.search(job_description, top_k=top_k)

        # Step 2: Generate tailored descriptions for each match
        for result in search_results:
            result["tailored_description"] = self.generate_tailored_description(
                result["experience"],
                job_description,
            )

        # Step 3: Overall fit analysis
        fit_analysis = self.generate_fit_analysis(search_results, job_description)

        elapsed = time.time() - start

        # Estimate cost
        estimated_cost = (
            self._total_input_tokens * _INPUT_COST_PER_M / 1_000_000
            + self._total_output_tokens * _OUTPUT_COST_PER_M / 1_000_000
        )

        return {
            "job_description": job_description,
            "matched_experiences": [
                {
                    "rank": r["rank"],
                    "relevance_score": r["relevance_score"],
                    "experience": r["experience"],
                    "tailored_description": r["tailored_description"],
                }
                for r in search_results
            ],
            "fit_analysis": fit_analysis,
            "metadata": {
                "total_experiences_searched": len(self.experiences),
                "model_used": LLM_MODEL,
                "query_time_seconds": round(elapsed, 2),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "estimated_cost": round(estimated_cost, 6),
            },
        }
