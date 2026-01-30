import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
_project_root = Path(__file__).parent
_env_path = _project_root / ".env"
load_dotenv(_env_path, override=True)

# API Keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY", "")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# Set environment variables for compatibility with code using os.environ directly
_env_vars = {
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "OPENAI_API_BASE": OPENAI_BASE_URL,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "XAI_API_KEY": XAI_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
}
for key, value in _env_vars.items():
    if value:
        os.environ[key] = value

# Project configuration
CHINESE_FONT_PATH = 'simhei.ttf'

LLM_ANSWER_TYPE = [
    'LLM_Response',
    'LLM_NaiveRAG_Response',
    'LLM_HypotheticalRAG_Response',
    'LLM_HierarchicalIndexRAG_Response',
    "LLM_LightRAG_Hybrid_Distillation_Response",
]

LLM_RESPONSE_TEMPERATURE = 0

EXP_CACHE_DIR = "RESULTS/eye_rag/exp_cache"
os.makedirs(EXP_CACHE_DIR, exist_ok=True)

RAG_DB_DIR = "Data/RAG"
MEDICAL_GUIDE_SOURCE = 'AAO_PPP'
if MEDICAL_GUIDE_SOURCE == 'AAO_PPP':
    WORKING_DIR = os.path.join(RAG_DB_DIR, "AAO_PPP/OphthaKG_AAO_PPP")
    RAG_MD_DIR = os.path.join(RAG_DB_DIR, "AAO_PPP/medical_guide_markdown")
    RAG_MD_TMP_DATA_PKL_PATH = os.path.join(RAG_DB_DIR, "AAO_PPP/medical_guide_md_data.pkl")
    RAG_FAISS_INDEX_DIR = os.path.join(RAG_DB_DIR, "AAO_PPP/faiss_indexes")
elif MEDICAL_GUIDE_SOURCE == 'COS':
    WORKING_DIR = os.path.join(RAG_DB_DIR, "COS/OphthaKG_COS")
    RAG_MD_DIR = os.path.join(RAG_DB_DIR, "COS/medical_guide_markdown")
    RAG_MD_TMP_DATA_PKL_PATH = os.path.join(RAG_DB_DIR, "COS/medical_guide_md_data.pkl")
    RAG_FAISS_INDEX_DIR = os.path.join(RAG_DB_DIR, "COS/faiss_indexes")
else:
    raise ValueError(f"Unknown MEDICAL_GUIDE_SOURCE: {MEDICAL_GUIDE_SOURCE}")

# LightRAG configuration
LIGHTRAG_QUERY_REWRITE_QUESTION = True
DEFAULT_LIGHTRAG_MODE = "hybrid"
DEFAULT_CHUNK_SEARCH_TOP_K = 10
DEFAULT_CHUNK_RERANK_TOP_K = 3
DEFAULT_KG_SEARCH_TOP_K = 10
DEFAULT_KG_RERANK_TOP_K = 3

# Naive RAG, HypotheticalRAG, HierarchicalIndexRAG configuration
QUERY_TOP_K = 10
RERANK_TOP_N = 3

# HierarchicalIndexRAG configuration
SUMMARY_TOP_K = 5

# Maximum number of records in whole context
NUM_RECORD_FOR_WHOLE_CONTEXT = 20
