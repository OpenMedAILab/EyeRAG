"""
Hierarchical RAG graph node using FAISS.

This module provides graph nodes for hierarchical retrieval using FAISS.
The Pinecone version is deprecated - use this FAISS version by default.
"""

import asyncio

from eye_rag.graph_node._state import EyeRAGGraphState
from config import SUMMARY_TOP_K, QUERY_TOP_K, RERANK_TOP_N

from eye_rag.rag.faiss.hi_rag import FAISSHierarchicalRAG

_rag_instance = None


def _get_rag_instance():
    """Lazy initialization of the FAISSHierarchicalRAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FAISSHierarchicalRAG()
    return _rag_instance


async def _retrieval_mg(query: str, question_id: str = ''):
    """Async retrieval using hierarchical RAG."""
    rag = _get_rag_instance()
    context = await rag.hi_retrieval(
        query=query,
        summary_top_k=SUMMARY_TOP_K,
        chunk_top_k=QUERY_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        question_id=question_id,
    )
    return context


def _retrieve_medical_guide(state: EyeRAGGraphState, questions) -> dict:
    """Batch retrieval synchronous version."""
    if isinstance(questions, list) and len(questions) == 1:
        questions = questions[0]

    question_id = state['question_id']
    result = []
    if isinstance(questions, list):
        for q in questions:
            assert isinstance(q, str), "All questions must be strings"
            res = asyncio.run(_retrieval_mg(q, question_id=question_id))
            result.append(res)
    else:
        assert isinstance(questions, str), "Question must be a string"
        result = asyncio.run(_retrieval_mg(questions, question_id=question_id))

    state["context"] = result
    return state


def retrieve_medical_guide_rewritten_questions(state: EyeRAGGraphState) -> dict:
    """Retrieve medical guide using rewritten questions."""
    state["curr_state"] = "faiss_hi_rag: retrieve_medical_guide_rewritten_questions"
    questions = state["rewritten_questions"]
    return _retrieve_medical_guide(state, questions)


def retrieve_medical_guide(state: EyeRAGGraphState) -> dict:
    """Retrieve medical guide using original question."""
    state["curr_state"] = "faiss_hi_rag: retrieve_medical_guide"
    return _retrieve_medical_guide(state, state["question"])


def retrieve_medical_guide_hypothetical_question(state: EyeRAGGraphState) -> dict:
    """Retrieve medical guide using hypothetical question."""
    state["curr_state"] = "faiss_hi_rag: retrieve_medical_guide_hypothetical_question"
    hypothetical_question = state["response"]
    return _retrieve_medical_guide(state, hypothetical_question)
