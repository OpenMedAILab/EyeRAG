# EyeRAG - Ophthalmology RAG System
from config import (
    LLM_ANSWER_TYPE,
    LLM_RESPONSE_TEMPERATURE,
    WORKING_DIR,
    DEFAULT_LIGHTRAG_MODE,
)

from .llm import (
    LLMModelName,
    get_chat_llm,
)

from eye_rag.qa.patient_data import (
    get_all_question_ids,
    init_patient_data,
    get_clinical_data_for_query,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "LLM_ANSWER_TYPE",
    "LLM_RESPONSE_TEMPERATURE",
    "WORKING_DIR",
    "DEFAULT_LIGHTRAG_MODE",
    # LLM
    "LLMModelName",
    "get_chat_llm",
    # Patient data
    "get_all_question_ids",
    "init_patient_data",
    "get_clinical_data_for_query",
]
