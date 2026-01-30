# State and utilities
from ._state import EyeRAGGraphState, execute_agent_graph, generate_graph_image

# Graph nodes
from .rewrite_question import rewrite_question_for_retrieval, rewrite_question

from .answer_question import (
    # Independent generation functions
    generate_response_with_clinical_data,
    generate_response_with_context,
    # Graph node wrappers (with state & cache)
    answer_question_with_context,
    answer_question_with_clinical_data,
)

# LightRAG retrieval (kept for backward compatibility)
from .lightrag import (
    retrieve_medical_guide,
    retrieve_medical_guide_hypothetical_question,
    retrieve_medical_guide_rewritten_questions,
    initialize_rag_sync,
    cleanup_rag,
)

# FAISS-based RAG retrieval (recommended, default)
from .naive_rag import (
    retrieve_medical_guide as faiss_retrieve_medical_guide,
    retrieve_medical_guide_hypothetical_question as faiss_retrieve_hypothetical,
    retrieve_medical_guide_rewritten_questions as faiss_retrieve_rewritten,
)

from .hi_rag import (
    retrieve_medical_guide as faiss_hi_retrieve_medical_guide,
    retrieve_medical_guide_hypothetical_question as faiss_hi_retrieve_hypothetical,
    retrieve_medical_guide_rewritten_questions as faiss_hi_retrieve_rewritten,
)

# Distillation
from .distill_retrieval import distill_context

__all__ = [
    # State
    "EyeRAGGraphState",
    "execute_agent_graph",
    "generate_graph_image",
    # Nodes
    "rewrite_question_for_retrieval",
    "rewrite_question",
    # Independent generation functions
    "generate_response_with_clinical_data",
    "generate_response_with_context",
    # Graph node wrappers
    "answer_question_with_context",
    "answer_question_with_clinical_data",
    # LightRAG (backward compatibility)
    "retrieve_medical_guide",
    "retrieve_medical_guide_hypothetical_question",
    "retrieve_medical_guide_rewritten_questions",
    "initialize_rag_sync",
    "cleanup_rag",
    # FAISS naive RAG (recommended)
    "faiss_retrieve_medical_guide",
    "faiss_retrieve_hypothetical",
    "faiss_retrieve_rewritten",
    # FAISS hierarchical RAG (recommended)
    "faiss_hi_retrieve_medical_guide",
    "faiss_hi_retrieve_hypothetical",
    "faiss_hi_retrieve_rewritten",
    # Distillation
    "distill_context",
]
