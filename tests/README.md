# Test Suite

4-layer pytest suite for the AI Experience Matcher RAG pipeline.

## Run

```bash
# Layer 1 only — no API key needed
pytest tests/test_data_integrity.py -v

# All layers — requires OPENAI_API_KEY
pytest tests/ -v
```

## Test Layers

### Layer 1: Data Integrity (`test_data_integrity.py`) — 0 API calls

| Test | What it checks |
|---|---|
| `test_experiences_file_exists` | `data/experiences.json` exists on disk |
| `test_experiences_is_valid_json` | File parses as a valid JSON list |
| `test_experiences_count` | Exactly 14 experiences in the dataset |
| `test_all_required_fields_present` | All 9 required fields present (parametrized x14) |
| `test_unique_ids` | No duplicate experience IDs |
| `test_skills_are_lists` | `skills`, `achievements`, `keywords` are lists |
| `test_nonempty_text_fields` | `title`, `company`, `duration`, `description` are non-empty |
| `test_load_experiences_returns_correct_count` | `load_experiences()` returns 14 (mocked) |
| `test_load_experiences_missing_file` | Missing file raises `FileNotFoundError` |
| `test_experience_to_document` | Conversion to LangChain `Document` has correct content and metadata |
| `test_config_values` | Model names, temperature, top_k match expected constants |
| `test_prompt_templates_defined` | Both prompt templates are valid `ChatPromptTemplate` instances |

### Layer 2: Retrieval Quality (`test_retrieval.py`) — Embedding API calls

| Test | What it checks |
|---|---|
| `test_search_returns_correct_count` | `top_k=3` returns exactly 3 results |
| `test_search_result_structure` | Each result has `experience`, `relevance_score`, `rank` |
| `test_search_scores_in_range` | All scores fall within (0.0, 1.0] |
| `test_search_results_ranked_descending` | Scores descend, ranks are 1, 2, 3 |
| `test_search_experience_has_required_fields` | Returned experiences have all 9 fields |
| `test_search_relevance_data_analyst` | Data analyst query → data-related top result |
| `test_search_relevance_ml_engineer` | ML query → ML-related top result |
| `test_search_top_k_parameter` | `top_k=1` returns 1, `top_k=5` returns 5 |
| `test_search_short_query` | Single-word query returns results without error |

### Layer 3: Generation Quality (`test_generation.py`) — LLM API calls

**Mocked tests (no API):**

| Test | What it checks |
|---|---|
| `test_tailored_description_calls_llm` | LLM is invoked and returns content |
| `test_tailored_description_retries_on_failure` | Retry succeeds after first failure |
| `test_tailored_description_raises_after_two_failures` | Exception propagates after 2 failures |
| `test_track_tokens` | Token counts accumulate correctly |

**Real API tests:**

| Test | What it checks |
|---|---|
| `test_zendo_in_search_results` | Zendo Technologies appears in top 3 for Data Analyst JD |
| `test_zendo_bullet_count` | Tailored description has 3-4 bullet points |
| `test_zendo_bullet_length` | Each bullet is 30-200 characters |
| `test_zendo_bullets_start_with_action_verb` | Each bullet starts with a strong action verb |
| `test_zendo_references_jd_keywords` | At least 1 JD keyword appears in output |
| `test_zendo_references_original_skills` | Anti-hallucination: original skills appear in output |
| `test_real_fit_analysis_contains_assessment` | Fit analysis contains Strong/Moderate/Weak |
| `test_llm_as_judge_relevance_and_specificity` | Separate GPT-4o-mini judge scores relevance and specificity >= 3/5 |

### Layer 4: End-to-End Integration (`test_integration.py`) — Full pipeline

| Test | What it checks |
|---|---|
| `test_match_and_generate_structure` | Result has all 4 top-level keys (JD, EXP, fit, metadata) |
| `test_match_and_generate_returns_correct_count` | Returns exactly 3 matched experiences |
| `test_match_and_generate_experience_structure` | Each match has rank, score, experience, tailored description |
| `test_match_and_generate_metadata` | Metadata has model info, timing, token counts, cost |
| `test_match_and_generate_tailored_descriptions_nonempty` | All descriptions are non-empty with bullet points |
| `test_match_and_generate_fit_analysis_has_assessment` | Fit analysis includes strength assessment |
| `test_match_and_generate_scores_ordered` | Relevance scores in descending order |
| `test_match_and_generate_short_jd` | 3-word JD completes without error |

## Shared Fixtures (`conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `skip_without_api_key` | function (autouse) | Auto-skips `@pytest.mark.api` tests when `OPENAI_API_KEY` is absent |
| `sample_experience` | function | Hardcoded experience dict with all 9 required fields |
| `sample_jd` | function | Data Analyst job description string |
| `matcher_with_vectorstore` | session | Real matcher with FAISS — built once per test run to minimise API cost |
| `mock_matcher` | function | Matcher with mocked OpenAI clients — zero API calls |
