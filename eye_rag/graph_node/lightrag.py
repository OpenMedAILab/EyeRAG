"""
LightRAG graph node for retrieval operations.

This module provides graph nodes for LightRAG-based retrieval.
Uses a cost-free BM25 reranker instead of paid API services.
"""

import asyncio
import os
from typing import List, Optional
from multiprocessing import Pool

from config import WORKING_DIR, DEFAULT_LIGHTRAG_MODE, EXP_CACHE_DIR
from config import (
    DEFAULT_KG_RERANK_TOP_K,
    DEFAULT_CHUNK_RERANK_TOP_K,
    DEFAULT_CHUNK_SEARCH_TOP_K,
    DEFAULT_KG_SEARCH_TOP_K,
)
from eye_rag.eye_rag_utils import get_catch_file_path, load_cache_file, save_cache_file
from eye_rag.graph_node._state import EyeRAGGraphState

from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG, QueryParam


USE_CACHE_FILE = True


# ============================================================================
# Cost-free BM25 Reranker
# ============================================================================

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank_bm25 not installed. Using no-rerank fallback. "
          "Install with: pip install rank-bm25")


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    import re
    # Simple word tokenization, lowercase
    return re.findall(r'\w+', text.lower())


async def rerank_func(
    query: str,
    documents: list,
    top_n: int = None,
    **kwargs
) -> list:
    """
    Cost-free rerank function using BM25.

    Falls back to no reranking if rank_bm25 is not installed.

    Args:
        query: The query string to rerank against
        documents: List of document dicts with 'content' field
        top_n: Number of top documents to return

    Returns:
        Reranked list of documents
    """
    if not documents:
        return documents

    if not HAS_BM25:
        # Fallback: return documents as-is (limited to top_n if specified)
        return documents[:top_n] if top_n else documents

    try:
        # Extract content from documents
        if isinstance(documents[0], dict):
            contents = [doc.get('content', str(doc)) for doc in documents]
        else:
            contents = [str(doc) for doc in documents]

        # Tokenize documents and query
        tokenized_docs = [_tokenize(content) for content in contents]
        tokenized_query = _tokenize(query)

        # Create BM25 index and get scores
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)

        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_n documents
        reranked = [doc for doc, _ in scored_docs]
        return reranked[:top_n] if top_n else reranked

    except Exception as e:
        print(f"BM25 rerank failed: {e}, returning original order")
        return documents[:top_n] if top_n else documents


# ============================================================================
# Event Loop Management
# ============================================================================

_rag_instance = None
_initialization_complete = False
_event_loop = None


def get_event_loop():
    """Get or create a single event loop."""
    global _event_loop

    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)

    return _event_loop


def run_in_shared_loop(coro):
    """Run coroutine in the shared event loop."""
    loop = get_event_loop()

    try:
        running_loop = asyncio.get_running_loop()
        if running_loop == loop:
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass

    if not loop.is_running():
        return loop.run_until_complete(coro)
    else:
        task = loop.create_task(coro)
        return loop.run_until_complete(task)


# ============================================================================
# RAG Initialization
# ============================================================================

def initialize_rag_sync():
    """Synchronously initialize RAG instance."""
    global _rag_instance, _initialization_complete

    if _rag_instance is not None and _initialization_complete:
        return _rag_instance

    print("Initializing RAG instance...")

    assert os.path.exists(WORKING_DIR), f"Working directory {WORKING_DIR} does not exist."
    # Ensure required files exist under WORKING_DIR
    required_files = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_full_relations.json",
        "kv_store_text_chunks.json",
        "kv_store_doc_status.json",
        "vdb_chunks.json",
        "kv_store_full_docs.json",
        "vdb_entities.json",
        "kv_store_full_entities.json",
        "vdb_relationships.json",
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(WORKING_DIR, f))]
    assert not missing, f"Missing required files under {WORKING_DIR}: {', '.join(missing)}"

    try:
        _rag_instance = LightRAG(
            working_dir=WORKING_DIR,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
            chunk_token_size=1000,
            chunk_overlap_token_size=100,
            rerank_model_func=rerank_func,
        )

        async def initialize_all():
            await _rag_instance.initialize_storages()
            await initialize_pipeline_status()

        run_in_shared_loop(initialize_all())

        _initialization_complete = True
        print("RAG instance initialized successfully")

    except Exception as e:
        print(f"RAG initialization failed: {e}")
        _rag_instance = None
        _initialization_complete = False
        raise

    return _rag_instance


def cleanup_rag():
    """Clean up RAG instance and event loop."""
    global _rag_instance, _initialization_complete, _event_loop

    if _rag_instance:
        _rag_instance = None
        _initialization_complete = False

    if _event_loop and not _event_loop.is_closed():
        _event_loop.close()
        _event_loop = None


# ============================================================================
# Retrieval Functions
# ============================================================================

def retrieval_sync(
    query: str,
    lightrag_mode: str,
    question_id: Optional[str] = ""
) -> str:
    """Synchronous retrieval function with caching."""
    mode = lightrag_mode
    assert mode in ["naive", "global", "local", "hybrid", "mix"]

    print(f"\n=====================\nLightRAG Query mode: {mode}")

    # Create query parameters
    if mode == "naive":
        param = QueryParam(
            mode=mode,
            enable_rerank=True,
            rerank_top_n=DEFAULT_CHUNK_RERANK_TOP_K,
            chunk_top_k=DEFAULT_CHUNK_SEARCH_TOP_K,
            only_need_context=True,
        )
    else:
        param = QueryParam(
            mode=mode,
            enable_rerank=True,
            rerank_top_n=DEFAULT_KG_RERANK_TOP_K,
            top_k=DEFAULT_KG_SEARCH_TOP_K,
            only_need_context=True,
        )

    # Generate cache file path
    param_dict = vars(param)
    cache_file_path = get_catch_file_path(
        question_id=question_id,
        cache_dir=os.path.join(EXP_CACHE_DIR, 'lightrag_context_cache'),
        question=f'LightRAG_{lightrag_mode}|{query}',
        param_dict=param_dict
    )

    # Try to load from cache
    context = load_cache_file(
        use_cache_file=USE_CACHE_FILE,
        cache_file_path=cache_file_path,
        key="context"
    )
    if context:
        print("Loaded from cache, skipping RAG initialization")
        return context

    # Initialize RAG and query
    print("Cache miss, initializing RAG instance...")
    rag = initialize_rag_sync()

    async def async_query():
        ctx = await rag.aquery(query, param=param)

        if USE_CACHE_FILE:
            save_cache_file(
                cache_file_path=cache_file_path,
                data_dict_to_save={
                    'question_id': question_id,
                    'question': query,
                    'context': ctx
                },
                param_dict=param_dict
            )

        return ctx

    return run_in_shared_loop(async_query())


def _batch_retrieve_medical_guide_sync(
    state: EyeRAGGraphState,
    questions,
    question_id: str = ''
) -> dict:
    """Synchronous batch retrieval."""
    lightrag_mode = state.get("lightrag_mode", DEFAULT_LIGHTRAG_MODE)

    if isinstance(questions, list) and len(questions) == 1:
        questions = questions[0]

    if isinstance(questions, list):
        result = []
        for q in questions:
            assert isinstance(q, str), "All questions must be strings"
            res = retrieval_sync(q, lightrag_mode, question_id=question_id)
            result.append(res)
    else:
        assert isinstance(questions, str), "Question must be a string"
        result = retrieval_sync(questions, lightrag_mode, question_id=question_id)

    state["context"] = result
    return state


# ============================================================================
# Graph Node Functions
# ============================================================================

def retrieve_medical_guide_rewritten_questions(state: EyeRAGGraphState) -> dict:
    """Retrieve medical guide using rewritten questions."""
    state["curr_state"] = "lightrag: retrieve_medical_guide_rewritten_questions"
    questions = state["rewritten_questions"]
    question_id = state.get("question_id", "")

    # Single question: direct processing
    if isinstance(questions, str) or (isinstance(questions, list) and len(questions) == 1):
        return _batch_retrieve_medical_guide_sync(state, questions, question_id=question_id)

    # Multiple questions: parallel processing
    if isinstance(questions, list) and len(questions) > 1:
        lightrag_mode = state.get("lightrag_mode", DEFAULT_LIGHTRAG_MODE)
        tasks = [(q, lightrag_mode, question_id) for q in questions]

        with Pool(processes=min(len(questions), os.cpu_count())) as pool:
            results = pool.starmap(retrieval_sync, tasks)

        state["context"] = results
        return state

    # Fallback
    return _batch_retrieve_medical_guide_sync(state, questions)


def retrieve_medical_guide(state: EyeRAGGraphState) -> dict:
    """Retrieve medical guide using original question."""
    state["curr_state"] = "lightrag: retrieve_medical_guide"
    question_id = state.get("question_id", "")
    return _batch_retrieve_medical_guide_sync(state, state["question"], question_id=question_id)


def retrieve_medical_guide_hypothetical_question(state: EyeRAGGraphState) -> dict:
    """Retrieve medical guide using hypothetical question (from response)."""
    state["curr_state"] = "lightrag: retrieve_medical_guide_hypothetical_question"
    hypothetical_question = state["response"]
    question_id = state.get("question_id", "")
    return _batch_retrieve_medical_guide_sync(state, hypothetical_question, question_id=question_id)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Initialization
    "initialize_rag_sync",
    "cleanup_rag",
    # Retrieval
    "retrieval_sync",
    "retrieve_medical_guide",
    "retrieve_medical_guide_rewritten_questions",
    "retrieve_medical_guide_hypothetical_question",
    # Reranker
    "rerank_func",
]


