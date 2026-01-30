"""
DEPRECATED: Pinecone-based RAG implementations.

This module is deprecated. Use eye_rag.rag.faiss (FAISS-based) instead.
These classes are kept for backward compatibility only.

Usage (deprecated):
    from eye_rag.rag.pinecone import RAGENMedicalGuide, HierarchicalRAG

Recommended usage:
    from eye_rag.rag import FAISSRAGMedicalGuide, FAISSHierarchicalRAG
"""

import warnings

warnings.warn(
    "eye_rag.rag.pinecone is deprecated. "
    "Use eye_rag.rag.faiss (FAISS-based) instead.",
    DeprecationWarning,
    stacklevel=2
)

from eye_rag.rag.pinecone.naive_rag import RAGMedicalGuide, RAGENMedicalGuide
from eye_rag.rag.pinecone.hi_rag import HierarchicalRAG

__all__ = [
    'RAGMedicalGuide',
    'RAGENMedicalGuide',
    'HierarchicalRAG',
]
