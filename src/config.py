import os
import tempfile

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.3
TOP_K_RESULTS = 3

# Streamlit Cloud has a read-only filesystem except /tmp
if os.environ.get("STREAMLIT_SHARING"):
    VECTOR_STORE_PATH = os.path.join(tempfile.gettempdir(), "experience_vectorstore")
else:
    VECTOR_STORE_PATH = "experience_vectorstore"
