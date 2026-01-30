"""
RAG (Retrieval-Augmented Generation) implementations.

This module provides RAG implementations using different vector stores:
- FAISS (recommended, default): Local vector store, no external API dependency
- Pinecone (deprecated): Cloud-based vector store, requires Pinecone API key

Usage:
    # Default FAISS-based RAG (recommended)
    from eye_rag.rag import FAISSRAGMedicalGuide, FAISSHierarchicalRAG

    # Or use the aliases
    from eye_rag.rag import RAGMedicalGuide, HierarchicalRAG

    # Pinecone-based RAG (deprecated, for backward compatibility)
    from eye_rag.rag.pinecone import RAGENMedicalGuide, HierarchicalRAG as PineconeHierarchicalRAG
"""

# FAISS-based RAG (recommended, default)
from eye_rag.rag.faiss import FAISSRAGMedicalGuide, FAISSHierarchicalRAG

# Default aliases - use FAISS implementations
RAGMedicalGuide = FAISSRAGMedicalGuide
HierarchicalRAG = FAISSHierarchicalRAG

# Utilities
from eye_rag.rag.rag_util import (
    load_markdown_files,
    _split_and_format_chunks,
)

__all__ = [
    # FAISS (default)
    "FAISSRAGMedicalGuide",
    "FAISSHierarchicalRAG",
    # Default aliases
    "RAGMedicalGuide",
    "HierarchicalRAG",
    # Utilities
    "load_markdown_files",
]
