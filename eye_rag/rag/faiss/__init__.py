"""
FAISS-based RAG implementations.

Default configuration:
- Embedding: OpenAI text-embedding-3-small (best quality)
- Reranking: BM25 (cost-free lexical matching)

Automatic fallback chain if OpenAI fails:
- OpenAI -> ONNX sentence-transformers -> TF-IDF

Example usage:
    # Default (OpenAI embedding + BM25 reranking)
    from eye_rag.rag.faiss import FAISSRAGMedicalGuide
    rag = FAISSRAGMedicalGuide()

    # With ONNX embeddings (cost-free, no PyTorch)
    rag = FAISSRAGMedicalGuide(embed_model="onnx:all-MiniLM-L6-v2")

    # With TF-IDF embeddings (cost-free, no dependencies)
    rag = FAISSRAGMedicalGuide(embed_model="tfidf")
"""

from eye_rag.rag.faiss.naive_rag import (
    # Main RAG classes
    FAISSRAGMedicalGuide,
    FAISSRAGENMedicalGuide,
    # Embedding models
    ONNXEmbedding,
    TFIDFEmbedding,
    OpenAIEmbedding,
    create_embedding_model,
    # Reranking models
    BM25Reranker,
    NoReranker,
    PineconeReranker,
    create_reranker,
    # Configuration
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
)
from eye_rag.rag.faiss.hi_rag import FAISSHierarchicalRAG

__all__ = [
    # Main RAG classes
    'FAISSRAGMedicalGuide',
    'FAISSRAGENMedicalGuide',
    'FAISSHierarchicalRAG',
    # Embedding models
    'ONNXEmbedding',
    'TFIDFEmbedding',
    'OpenAIEmbedding',
    'create_embedding_model',
    # Reranking models
    'BM25Reranker',
    'NoReranker',
    'PineconeReranker',
    'create_reranker',
    # Configuration
    'DEFAULT_EMBED_MODEL',
    'DEFAULT_RERANK_MODEL',
]
