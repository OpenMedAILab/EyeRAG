"""
FAISS-based Hierarchical RAG implementation.

This module provides a two-stage hierarchical RAG system using FAISS:
1. First stage: Retrieve relevant documents using document summaries
2. Second stage: Retrieve specific chunks from the relevant documents

Default configuration:
- Embedding: OpenAI text-embedding-3-small (best quality)
- Reranking: BM25 (cost-free lexical matching)

Automatic fallback chain if OpenAI embedding fails:
- OpenAI -> ONNX sentence-transformers -> TF-IDF

Note: Summary generation uses OpenAI (gpt-4o-mini) for document summarization.

Parallel Retrieval:
This module supports parallel retrieval using multiprocessing. Both the chunk
index and summary index are built once during initialization. Use
`get_shared_hi_rag_instance()` for thread-safe singleton access, or
`parallel_hi_retrieval()` for multiprocess-safe parallel retrieval.
"""

import asyncio
import os
import pickle
import threading
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm

from config import EXP_CACHE_DIR, RAG_MD_DIR, RAG_MD_TMP_DATA_PKL_PATH, RAG_FAISS_INDEX_DIR
from eye_rag.eye_rag_utils import get_catch_file_path, load_cache_file, save_cache_file
from eye_rag.rag.faiss.naive_rag import (
    FAISSRAGMedicalGuide,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    retrieval_result_to_dict_list,
    create_reranker,
)

# Constants
HIERARCHICAL_INDEX_NAME_SUMMARY = "md-medical-guide-faiss-summary"
HIERARCHICAL_INDEX_NAME_CHUNKS = "md-medical-guide-faiss-chunks"

USE_CACHE_FILE = True


def get_chunk_key(filename: str) -> str:
    """Extract chunk key from filename."""
    return os.path.basename(filename).replace(".md", '')


class FAISSHierarchicalRAG(FAISSRAGMedicalGuide):
    """
    Hierarchical RAG with document summaries and chunk retrieval using FAISS.

    This class implements a two-stage retrieval process:
    1. Summary-level retrieval to identify relevant documents
    2. Chunk-level retrieval from the identified documents

    Default: OpenAI embeddings + BM25 reranking.
    Fallback: ONNX -> TF-IDF embeddings if OpenAI fails.
    """

    def __init__(
        self,
        index_name_summary: str = HIERARCHICAL_INDEX_NAME_SUMMARY,
        index_name_chunks: str = HIERARCHICAL_INDEX_NAME_CHUNKS,
        md_dir: str = RAG_MD_DIR,
        tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        index_dir: str = RAG_FAISS_INDEX_DIR,
    ):
        """
        Initialize the Hierarchical FAISS RAG system.

        Args:
            index_name_summary: Name for the summary FAISS index
            index_name_chunks: Name for the chunks FAISS index
            md_dir: Directory containing markdown files
            tmp_md_data_pkl_path: Path to pickle cache file
            embed_model: Embedding model spec (e.g., "onnx:all-MiniLM-L6-v2", "tfidf", "openai:text-embedding-3-small")
            rerank_model: Reranker model spec (e.g., "bm25", "pinecone:cohere-rerank-3.5")
            index_dir: Directory to store FAISS indexes
        """
        # Initialize parent class for chunks index
        super().__init__(
            index_name=index_name_chunks,
            md_dir=md_dir,
            tmp_md_data_pkl_path=tmp_md_data_pkl_path,
            embed_model=embed_model,
            rerank_model=rerank_model,
            index_dir=index_dir,
        )

        self.index_name_summary = index_name_summary
        self.index_name_chunks = index_name_chunks

        # Paths for summary index persistence
        self.summary_index_path = os.path.join(index_dir, f"{index_name_summary}.index")
        self.summary_metadata_path = os.path.join(index_dir, f"{index_name_summary}_metadata.pkl")

        # Summary generation LLM
        self.summary_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=4000)
        self.summary_chain = load_summarize_chain(self.summary_llm, chain_type="map_reduce")

        # Summary index storage
        self.summary_index = None
        self.summary_metadata = []

        # Reference to chunks index from parent
        self.chunks_index = self.index

        self.cache_file_dir = os.path.join(EXP_CACHE_DIR, 'faiss_md_hi_retrieval')
        os.makedirs(self.cache_file_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)


    async def create_document_summaries(self) -> List[Dict]:
        """
        Create summaries for each document and save to text files.

        Returns:
            List of summary records with metadata
        """
        if self.guide_data is None:
            self.guide_data = self.load_medical_guide_data()

        summary_dir = self.md_dir + '_summary'
        os.makedirs(summary_dir, exist_ok=True)

        summaries = []

        for idx, record in enumerate(tqdm(self.guide_data, desc="Creating document summaries")):
            try:
                md_basename = os.path.splitext(record['filename'])[0]
                summary_filename = f"{md_basename}_summary.txt"
                summary_path = os.path.join(summary_dir, summary_filename)

                if os.path.exists(summary_path):
                    print(f"Loading existing summary from: {summary_path}")
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary_text = f.read().strip()
                else:
                    doc = Document(page_content=record['text'], metadata=record)
                    print(f"Creating summary for document {idx}: {record['filename']}")
                    summary_output = await self.summary_chain.ainvoke([doc])
                    summary_text = summary_output['output_text']

                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(summary_text)
                    print(f"Saved summary to: {summary_path}")

                summary_record = {
                    'id': f"summary_{idx}",
                    'summary_text': summary_text,
                    'filename': record['filename'],
                    'reference': record['reference'],
                }
                summaries.append(summary_record)

                if not os.path.exists(summary_path):
                    await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error creating summary for {record['filename']}: {e}")
                # Use fallback summary (first 500 chars)
                fallback_summary = record['text'][:500] + "..."
                summary_record = {
                    'id': f"summary_{idx}",
                    'summary_text': fallback_summary,
                    'filename': record['filename'],
                    'reference': record['reference'],
                }
                summaries.append(summary_record)

                try:
                    md_basename = os.path.splitext(record['filename'])[0]
                    summary_filename = f"{md_basename}_summary.txt"
                    summary_path = os.path.join(summary_dir, summary_filename)

                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(f"Source: {record['filename']}\n")
                        f.write(f"Reference: {record['reference']}\n")
                        f.write("=" * 50 + "\n")
                        f.write("(Fallback Summary - LLM summarization failed)\n\n")
                        f.write(fallback_summary)
                    print(f"Saved fallback summary to: {summary_path}")
                except Exception as save_error:
                    print(f"Error saving fallback summary for {record['filename']}: {save_error}")

        return summaries

    async def create_summary_index(self) -> faiss.Index:
        """
        Create FAISS index for document summaries.

        Returns:
            FAISS index for summaries
        """
        if self.guide_data is None:
            self.guide_data = self.load_medical_guide_data()

        print(f"Building summary FAISS index from {len(self.guide_data)} markdown files...")

        # Create summaries
        summaries = await self.create_document_summaries()
        self.summary_metadata = summaries

        # Generate embeddings
        print("Generating summary embeddings...")
        summary_texts = [s['summary_text'] for s in summaries]

        batch_size = 100
        all_embeddings = []
        for i in tqdm(range(0, len(summary_texts), batch_size), desc="Embedding summaries"):
            batch_texts = summary_texts[i:i + batch_size]
            batch_embeddings = self.embed_model.encode(batch_texts, batch_size=batch_size)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        # Save index and metadata
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(index, self.summary_index_path)
        with open(self.summary_metadata_path, 'wb') as f:
            pickle.dump(self.summary_metadata, f)

        print(f"Saved summary FAISS index with {index.ntotal} vectors")
        return index

    async def load_or_build_summary_index_async(self) -> faiss.Index:
        """
        Load existing summary index or build a new one asynchronously.

        Returns:
            FAISS index for summaries
        """
        if os.path.exists(self.summary_index_path) and os.path.exists(self.summary_metadata_path):
            print(f"Loading existing summary index: {self.index_name_summary}")
            index = faiss.read_index(self.summary_index_path)
            with open(self.summary_metadata_path, 'rb') as f:
                self.summary_metadata = pickle.load(f)
            print(f"Loaded summary index with {index.ntotal} vectors")
            return index
        else:
            print(f"Summary index '{self.index_name_summary}' does not exist. Creating...")
            index = await self.create_summary_index()
            print(f"Created summary index: {self.index_name_summary}")
            return index

    async def initialize_summary_index(self):
        """Initialize summary index asynchronously if not already loaded."""
        if self.summary_index is None:
            self.summary_index = await self.load_or_build_summary_index_async()

    def _search_summaries(self, query: str, top_k: int) -> List[Dict]:
        """
        Search summary index for relevant documents.

        Args:
            query: Search query string
            top_k: Number of summaries to retrieve

        Returns:
            List of summary results with scores
        """
        # Generate query embedding
        query_embedding = self.embed_model.encode_query(query)
        faiss.normalize_L2(query_embedding)

        # Search summary index
        scores, indices = self.summary_index.search(query_embedding, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.summary_metadata):
                result = self.summary_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)

        return results

    def _rerank_summaries(self, query: str, results: List[Dict], top_n: int) -> List[Dict]:
        """
        Rerank summary results using the configured reranker.

        Args:
            query: Original query string
            results: List of summary results
            top_n: Number of results after reranking

        Returns:
            Reranked list of summary results
        """
        if not results:
            return []

        documents = [r['summary_text'] for r in results]
        rerank_results = self.rerank_model.rerank(query, documents, top_n)

        reranked = []
        for item in rerank_results:
            result = results[item['index']].copy()
            result['rerank_score'] = item['score']
            reranked.append(result)

        return reranked

    async def _retrieve_relevant_documents(
        self,
        query: str,
        summary_top_k: int
    ) -> List[Dict]:
        """
        Retrieve relevant documents using summary search.

        Args:
            query: Search query string
            summary_top_k: Number of documents to retrieve

        Returns:
            List of relevant document summary results
        """
        await self.initialize_summary_index()

        # Initial search with extra candidates for reranking
        initial_results = self._search_summaries(query, summary_top_k + 2)

        # Rerank summaries
        reranked_results = self._rerank_summaries(query, initial_results, summary_top_k)

        return reranked_results

    def _search_chunks_filtered(
        self,
        query: str,
        relevant_docs: List[str],
        top_k: int,
    ) -> List[Dict]:
        """
        Search chunks index and filter by relevant documents.

        Args:
            query: Search query string
            relevant_docs: List of relevant document filenames
            top_k: Number of chunks to retrieve

        Returns:
            List of chunk results from relevant documents only
        """
        # Generate query embedding
        query_embedding = self.embed_model.encode_query(query)
        faiss.normalize_L2(query_embedding)

        # Search with more candidates to allow for filtering
        search_k = min(top_k * 5, self.chunks_index.ntotal)
        scores, indices = self.chunks_index.search(query_embedding, search_k)

        # Filter results by relevant documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx]
                if chunk.get('filename') in relevant_docs:
                    result = chunk.copy()
                    result['score'] = float(score)
                    results.append(result)
                    if len(results) >= top_k:
                        break

        return results

    async def hi_retrieval(
        self,
        query: str,
        summary_top_k: int,
        chunk_top_k: int,
        rerank_top_n: int,
        question_id: str = '',
    ) -> List[Dict]:
        """
        Hierarchical retrieval: first find relevant docs, then retrieve chunks.

        This implements a two-stage retrieval process:
        1. Use document summaries to identify relevant documents
        2. Retrieve chunks only from those relevant documents

        Args:
            query: Search query string
            summary_top_k: Number of documents to retrieve in first stage
            chunk_top_k: Number of chunks to retrieve in second stage
            rerank_top_n: Number of final results after reranking
            question_id: Optional question identifier for caching

        Returns:
            List of context dictionaries
        """
        # Cache parameters
        param_dict = dict(
            summary_top_k=summary_top_k,
            chunk_top_k=chunk_top_k,
            rerank_top_n=rerank_top_n,
            query=query,
            index_name_summary=self.index_name_summary,
            index_name_chunks=self.index_name_chunks,
            rerank_model=self.rerank_model_name,
            embed_model=self.embed_model_name,
        )
        cache_file_path = get_catch_file_path(
            cache_dir=self.cache_file_dir,
            question_id=question_id,
            question=query,
            param_dict=param_dict,
        )

        # Check cache
        context = load_cache_file(
            use_cache_file=USE_CACHE_FILE,
            cache_file_path=cache_file_path,
            key="context",
        )
        if context:
            print("Loading from cache, skipping hierarchical retrieval")
            return context

        # Stage 1: Retrieve relevant documents using summaries
        summary_results = await self._retrieve_relevant_documents(query, summary_top_k)
        relevant_docs = [
            r['filename']
            for r in summary_results
            if 'filename' in r
        ]

        print(f"Found {len(relevant_docs)} relevant documents from summary search")
        for k, doc in enumerate(relevant_docs):
            print(f"{k + 1}. {doc}")

        if not relevant_docs:
            return []

        # Stage 2: Retrieve chunks from relevant documents
        chunk_results = self._search_chunks_filtered(query, relevant_docs, chunk_top_k)

        # Rerank chunks
        reranked_chunks = self._rerank(query, chunk_results, rerank_top_n)

        # Format output
        context = retrieval_result_to_dict_list(reranked_chunks)

        # Save to cache
        if USE_CACHE_FILE:
            save_cache_file(
                cache_file_path=cache_file_path,
                data_dict_to_save={'question_id': question_id, "context": context},
                param_dict=param_dict,
            )

        return context

    async def test(self):
        """Test hierarchical retrieval functionality."""
        query = "What is the long-term prognosis for primary acute angle-closure glaucoma?"

        # Test naive retrieval from parent class
        print("\n=== Testing Naive Retrieval ===")
        naive_context = self.retrieval(query=query, query_top_k=10, rerank_top_n=3)
        print(f"Naive retrieval results: {len(naive_context)} contexts")

        # Test hierarchical retrieval
        print("\n=== Testing Hierarchical Retrieval ===")
        hi_context = await self.hi_retrieval(
            query=query,
            summary_top_k=5,
            chunk_top_k=10,
            rerank_top_n=3,
        )
        print(f"Hierarchical retrieval results: {len(hi_context)} contexts")


def rerank(query: str, documents: list, top_n: int, rerank_model: str = DEFAULT_RERANK_MODEL) -> list:
    """
    Rerank documents using the configured reranker.

    Args:
        query: Query string for reranking
        documents: List of Document objects with page_content
        top_n: Number of top results to return
        rerank_model: Reranker model specification

    Returns:
        Reranked list of documents
    """
    reranker = create_reranker(rerank_model)

    doc_texts = [doc.page_content for doc in documents]
    rerank_results = reranker.rerank(query, doc_texts, top_n)

    # Return reranked documents
    return [documents[item['index']] for item in rerank_results]


# ============================================================================
# Singleton Instance Management for Parallel Retrieval
# ============================================================================

# Module-level singleton instance and lock for thread-safe initialization
_hi_rag_instance: Optional['FAISSHierarchicalRAG'] = None
_hi_rag_instance_lock = threading.Lock()

# Worker-level instance for multiprocessing
_worker_hi_rag_instance: Optional['FAISSHierarchicalRAG'] = None


def get_shared_hi_rag_instance(
    index_name_summary: str = HIERARCHICAL_INDEX_NAME_SUMMARY,
    index_name_chunks: str = HIERARCHICAL_INDEX_NAME_CHUNKS,
    md_dir: str = RAG_MD_DIR,
    tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    index_dir: str = RAG_FAISS_INDEX_DIR,
    force_new: bool = False,
) -> 'FAISSHierarchicalRAG':
    """
    Get or create a singleton Hierarchical RAG instance.

    This function ensures both the summary index and chunk index are built
    only once, even when called from multiple threads.

    Args:
        index_name_summary: Name for the summary FAISS index
        index_name_chunks: Name for the chunks FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes
        force_new: If True, create a new instance even if one exists

    Returns:
        Shared FAISSHierarchicalRAG instance
    """
    global _hi_rag_instance

    with _hi_rag_instance_lock:
        if _hi_rag_instance is None or force_new:
            print("Initializing shared Hierarchical RAG instance...")
            _hi_rag_instance = FAISSHierarchicalRAG(
                index_name_summary=index_name_summary,
                index_name_chunks=index_name_chunks,
                md_dir=md_dir,
                tmp_md_data_pkl_path=tmp_md_data_pkl_path,
                embed_model=embed_model,
                rerank_model=rerank_model,
                index_dir=index_dir,
            )
            print("Shared Hierarchical RAG instance initialized successfully.")

    return _hi_rag_instance


async def ensure_summary_index_built(
    index_name_summary: str = HIERARCHICAL_INDEX_NAME_SUMMARY,
    index_name_chunks: str = HIERARCHICAL_INDEX_NAME_CHUNKS,
    md_dir: str = RAG_MD_DIR,
    tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    index_dir: str = RAG_FAISS_INDEX_DIR,
) -> 'FAISSHierarchicalRAG':
    """
    Ensure both chunk and summary indexes are built before parallel retrieval.

    This async function initializes the Hierarchical RAG and builds the
    summary index if it doesn't exist. Call this before parallel retrieval
    to ensure all indexes are ready.

    Args:
        index_name_summary: Name for the summary FAISS index
        index_name_chunks: Name for the chunks FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes

    Returns:
        Initialized FAISSHierarchicalRAG instance with summary index loaded
    """
    rag = get_shared_hi_rag_instance(
        index_name_summary=index_name_summary,
        index_name_chunks=index_name_chunks,
        md_dir=md_dir,
        tmp_md_data_pkl_path=tmp_md_data_pkl_path,
        embed_model=embed_model,
        rerank_model=rerank_model,
        index_dir=index_dir,
    )

    # Ensure summary index is built
    await rag.initialize_summary_index()

    return rag


def init_worker_hi_rag(
    index_name_summary: str,
    index_name_chunks: str,
    md_dir: str,
    tmp_md_data_pkl_path: str,
    embed_model: str,
    rerank_model: str,
    index_dir: str,
):
    """
    Initialize Hierarchical RAG instance in multiprocessing worker.

    This function should be passed as the `initializer` argument to
    `multiprocessing.Pool()`. The FAISS indexes are loaded from disk
    (not rebuilt) since they were already created during initialization.

    Args:
        index_name_summary: Name for the summary FAISS index
        index_name_chunks: Name for the chunks FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes
    """
    global _worker_hi_rag_instance

    print(f"Worker {os.getpid()}: Initializing Hierarchical RAG instance...")
    _worker_hi_rag_instance = FAISSHierarchicalRAG(
        index_name_summary=index_name_summary,
        index_name_chunks=index_name_chunks,
        md_dir=md_dir,
        tmp_md_data_pkl_path=tmp_md_data_pkl_path,
        embed_model=embed_model,
        rerank_model=rerank_model,
        index_dir=index_dir,
    )

    # Load summary index synchronously in worker
    summary_index_path = os.path.join(index_dir, f"{index_name_summary}.index")
    summary_metadata_path = os.path.join(index_dir, f"{index_name_summary}_metadata.pkl")

    if os.path.exists(summary_index_path) and os.path.exists(summary_metadata_path):
        _worker_hi_rag_instance.summary_index = faiss.read_index(summary_index_path)
        with open(summary_metadata_path, 'rb') as f:
            _worker_hi_rag_instance.summary_metadata = pickle.load(f)
        print(f"Worker {os.getpid()}: Loaded summary index with "
              f"{_worker_hi_rag_instance.summary_index.ntotal} vectors")
    else:
        raise RuntimeError(f"Summary index not found at {summary_index_path}. "
                           "Run ensure_summary_index_built() first.")

    print(f"Worker {os.getpid()}: Hierarchical RAG instance initialized.")


def _worker_hi_retrieval_sync(args: Tuple) -> List[Dict]:
    """
    Perform hierarchical retrieval synchronously in worker process.

    This is a synchronous wrapper around the async hi_retrieval method,
    designed to be called via `pool.map()`.

    Args:
        args: Tuple of (query, summary_top_k, chunk_top_k, rerank_top_n, question_id)

    Returns:
        List of context dictionaries
    """
    global _worker_hi_rag_instance

    if _worker_hi_rag_instance is None:
        raise RuntimeError("Worker Hierarchical RAG instance not initialized. "
                           "Use init_worker_hi_rag as Pool initializer.")

    query, summary_top_k, chunk_top_k, rerank_top_n, question_id = args

    # Run async method in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _worker_hi_rag_instance.hi_retrieval(
                query=query,
                summary_top_k=summary_top_k,
                chunk_top_k=chunk_top_k,
                rerank_top_n=rerank_top_n,
                question_id=question_id,
            )
        )
        return result
    finally:
        loop.close()


def parallel_hi_retrieval(
    queries: List[Tuple[str, int, int, int, str]],
    num_workers: int = 4,
    index_name_summary: str = HIERARCHICAL_INDEX_NAME_SUMMARY,
    index_name_chunks: str = HIERARCHICAL_INDEX_NAME_CHUNKS,
    md_dir: str = RAG_MD_DIR,
    tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    index_dir: str = RAG_FAISS_INDEX_DIR,
) -> List[List[Dict]]:
    """
    Perform parallel hierarchical retrieval using multiprocessing.

    This function first ensures both the chunk and summary indexes exist,
    then distributes retrieval across multiple worker processes.

    Args:
        queries: List of tuples (query, summary_top_k, chunk_top_k, rerank_top_n, question_id)
        num_workers: Number of parallel worker processes
        index_name_summary: Name for the summary FAISS index
        index_name_chunks: Name for the chunks FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes

    Returns:
        List of retrieval results (one per query)

    Example:
        >>> import asyncio
        >>>
        >>> # First, ensure indexes are built (done once)
        >>> asyncio.run(ensure_summary_index_built())
        >>>
        >>> # Define queries: (query, summary_top_k, chunk_top_k, rerank_top_n, question_id)
        >>> queries = [
        ...     ("What causes glaucoma?", 5, 10, 3, "q1"),
        ...     ("Treatment for cataracts?", 5, 10, 3, "q2"),
        ...     ("Symptoms of macular degeneration?", 5, 10, 3, "q3"),
        ... ]
        >>>
        >>> # Run parallel retrieval
        >>> results = parallel_hi_retrieval(queries, num_workers=4)
    """
    import multiprocessing as mp

    # Ensure indexes are built before spawning workers
    print("Ensuring FAISS indexes are built before parallel hierarchical retrieval...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            ensure_summary_index_built(
                index_name_summary=index_name_summary,
                index_name_chunks=index_name_chunks,
                md_dir=md_dir,
                tmp_md_data_pkl_path=tmp_md_data_pkl_path,
                embed_model=embed_model,
                rerank_model=rerank_model,
                index_dir=index_dir,
            )
        )
    finally:
        loop.close()

    print(f"Starting parallel hierarchical retrieval with {num_workers} workers "
          f"for {len(queries)} queries...")

    # Use spawn context for better compatibility
    ctx = mp.get_context('spawn')

    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker_hi_rag,
        initargs=(index_name_summary, index_name_chunks, md_dir,
                  tmp_md_data_pkl_path, embed_model, rerank_model, index_dir),
    ) as pool:
        results = pool.map(_worker_hi_retrieval_sync, queries)

    print(f"Parallel hierarchical retrieval complete. Processed {len(results)} queries.")
    return results


