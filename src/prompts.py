"""LLM prompt templates for the Experience Matcher."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an expert resume writer and career coach. Your job is to \
rewrite a candidate's experience to be highly relevant to a specific \
job description.

Rules:
- Write exactly 3-4 bullet points
- Start each bullet with a strong action verb (Built, Developed, Led, etc.)
- Preserve ALL quantifiable metrics from the original (numbers, percentages)
- Highlight skills that match the job requirements
- Do NOT invent achievements or skills not present in the original
- Keep each bullet point to 1-2 lines
- Use professional, concise language"""

MATCH_PROMPT = """Job Description:
{job_description}

Candidate's Experience:
Title: {title}
Company: {company}
Duration: {duration}
Original Description: {description}
Skills Used: {skills}
Key Achievements: {achievements}

Rewrite this experience to maximize relevance to the job description above.
Output ONLY the bullet points, nothing else."""

ANALYSIS_PROMPT = """Based on the following matched experiences and job description, \
provide a brief 2-3 sentence analysis of the candidate's overall fit.

Job Description: {job_description}

Matched Experiences (ranked by relevance):
{experiences_summary}

Provide:
1. Overall fit assessment (Strong/Moderate/Weak match)
2. Key strengths that align with this role
3. Any notable gaps the candidate should address"""

MATCH_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", MATCH_PROMPT),
    ]
)

ANALYSIS_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a career advisor providing concise, honest fit assessments.",
        ),
        ("human", ANALYSIS_PROMPT),
    ]
)
