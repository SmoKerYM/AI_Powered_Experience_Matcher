# AI-Powered Experience Matcher

**Semantically match your professional experiences to any job description using RAG.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

![Demo](demo/screenshot.png)

---

## What It Does

AI Experience Matcher takes a job description and finds the most relevant experiences from your professional history using semantic vector search. It then uses GPT-4o-mini to rewrite each matched experience with tailored bullet points optimised for the target role. The result is a ranked list of your best-fit experiences with AI-generated descriptions ready to paste into your resume.

---

## Architecture

```
Job Description
      |
      v
 [ OpenAI Embedding ]  ──>  1536-dim vector
      |
      v
 [ FAISS Vector Search ]  ──>  Top-K nearest experiences
      |
      v
 [ GPT-4o-mini ]  ──>  Tailored bullet points + Fit analysis
      |
      v
 Optimised Output (Streamlit UI / CLI)
```

The pipeline follows a **Retrieval-Augmented Generation (RAG)** pattern:
1. **Embed** the job description using `text-embedding-3-small`
2. **Retrieve** the top-K most similar experiences via FAISS L2 search
3. **Generate** tailored descriptions and fit analysis using GPT-4o-mini

---

## Tech Stack

| Component         | Technology                          |
| ----------------- | ----------------------------------- |
| Embeddings        | OpenAI `text-embedding-3-small`     |
| Vector Store      | FAISS (Facebook AI Similarity Search) |
| LLM               | OpenAI `gpt-4o-mini`               |
| Orchestration     | LangChain                          |
| Frontend          | Streamlit                           |
| Language          | Python 3.9+                        |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/MingweiYan/AI-Powered-Experience-Matcher.git
cd AI-Powered-Experience-Matcher
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 5. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python run.py "Your job description here"
# or pipe from a file
cat job_description.txt | python run.py
```

---

## Project Structure

```
AI-Powered-Experience-Matcher/
├── app.py                  # Streamlit web UI
├── run.py                  # CLI tool
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── src/
│   ├── __init__.py
│   ├── config.py           # Model and path constants
│   ├── matcher.py          # Core RAG pipeline (ExperienceMatcher)
│   └── prompts.py          # LLM prompt templates
├── data/
│   └── experiences.json    # Your professional experiences
├── demo/                   # Screenshots and demo media
├── test_setup.py           # Environment validation
├── test_search.py          # Vector search tests
├── test_llm.py             # LLM generation tests
└── test_integration.py     # Full pipeline integration tests
```

---

## How It Works

Traditional keyword matching misses the nuance of job descriptions. "Data analysis" and "business intelligence" mean similar things but share no words.

This tool solves that by converting both your experiences and the job description into **embedding vectors** — numerical representations that capture semantic meaning. Experiences are stored in a FAISS index for fast nearest-neighbour lookup.

When you submit a job description:
1. It gets embedded into the same vector space as your experiences
2. FAISS finds the closest matches by L2 distance (equivalent to cosine similarity for normalised vectors)
3. Each matched experience is sent to GPT-4o-mini along with the job description
4. The LLM rewrites bullet points to emphasise relevant skills while preserving your actual achievements
5. A fit analysis summarises your overall alignment with the role

---

## Performance

| Metric              | Value             |
| ------------------- | ----------------- |
| Query time          | ~5-8s (with LLM)  |
| Cost per query      | ~$0.001           |
| Embedding model     | text-embedding-3-small (1536 dims) |
| LLM model           | gpt-4o-mini       |
| Vector search       | < 10ms (FAISS)    |

---

## Future Enhancements

- **Cover letter generation** — Auto-generate a cover letter from matched experiences
- **Skill gap analysis** — Identify missing skills and suggest learning resources
- **Multi-resume support** — Load multiple experience profiles for different career paths
- **PDF export** — Download results as a formatted PDF
- **Batch processing** — Analyse multiple job descriptions at once
- **Fine-tuned embeddings** — Train domain-specific embeddings for better matching

---

## Author

**Mingwei Yan**

[![GitHub](https://img.shields.io/badge/GitHub-MingweiYan-181717?logo=github)](https://github.com/MingweiYan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mingwei_Yan-0A66C2?logo=linkedin)](https://www.linkedin.com/in/mingwei-yan/)

---

## Acknowledgements

Built with assistance from [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
