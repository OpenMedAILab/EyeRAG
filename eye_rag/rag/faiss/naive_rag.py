"""
FAISS-based Naive RAG implementation.

This module provides a RAG system using FAISS for vector storage.

Embedding options (in order of preference):
1. OpenAI embeddings (default, API-based, best quality)
2. ONNX sentence-transformers (fallback, cost-free, no PyTorch)
3. TF-IDF + SVD (fallback, cost-free, no dependencies)

Reranking options:
1. BM25 (default, cost-free, no dependencies)
2. None (no reranking, use FAISS scores directly)
3. Pinecone hosted rerankers (optional, API-based, costs money)

Parallel Retrieval:
This module supports parallel retrieval using multiprocessing. The database
is built once during initialization, and retrieval can be executed in parallel
without rebuilding. Use `get_shared_rag_instance()` to get a singleton instance,
or use `parallel_retrieval()` for multiprocess-safe retrieval.
"""

import os
import pickle
import re
import threading
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from tqdm.auto import tqdm

from config import EXP_CACHE_DIR, RAG_MD_DIR, RAG_MD_TMP_DATA_PKL_PATH, RAG_FAISS_INDEX_DIR
from eye_rag.eye_rag_utils import get_catch_file_path, load_cache_file, save_cache_file
from eye_rag.rag.rag_util import load_markdown_files, _split_and_format_chunks

USE_CACHE_FILE = True

# Default configuration
RAG_MD_INDEX_NAME = 'md-medical-guide-faiss-chunks'

# ============================================================================
# Embedding Models Configuration
# ============================================================================
# Default (best quality):
# - "openai:text-embedding-3-small" (1536 dim, multilingual) [DEFAULT]
#
# Cost-free fallbacks (if OpenAI fails):
# - "onnx:all-MiniLM-L6-v2" (ONNX runtime, fast, 384 dim)
# - "tfidf" (TF-IDF + SVD, no dependencies, configurable dim)

DEFAULT_EMBED_MODEL = "openai:text-embedding-3-small"
FALLBACK_EMBED_MODEL = "onnx:all-MiniLM-L6-v2"
FALLBACK_EMBED_MODEL_2 = "tfidf"

# ============================================================================
# Reranking Models Configuration
# ============================================================================
# Cost-free options (default):
# - "bm25" (lexical matching, fast, no dependencies) [DEFAULT]
# - "none" (no reranking, use FAISS scores directly)
#
# Optional API-based (costs money):
# - "pinecone:cohere-rerank-3.5" (Cohere via Pinecone)

DEFAULT_RERANK_MODEL = "bm25"
FALLBACK_RERANK_MODEL = "none"


def load_or_cache_medical_guide_data(md_dir: str, tmp_md_data_pkl_path: str) -> List[Dict]:
    """
    Load markdown data from cache or process from files.

    Args:
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file

    Returns:
        List of medical guide data dictionaries
    """
    if os.path.exists(tmp_md_data_pkl_path):
        print("Loading cached markdown data...")
        with open(tmp_md_data_pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Loading and processing markdown files for the first time...")
        data = load_markdown_files(md_dir)
        os.makedirs(os.path.dirname(tmp_md_data_pkl_path), exist_ok=True)
        with open(tmp_md_data_pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached markdown data to {tmp_md_data_pkl_path}")
    return data


def format_retrieval_contexts(results: List[Dict]) -> str:
    """
    Format retrieval results into a readable context string.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted context string with numbered contexts
    """
    context_str = ""
    if results:
        cnt = 1
        for result in results:
            chunk_text = result.get('chunk_text', '')
            if chunk_text:
                filename = result.get('filename', '')
                filename_str = f" (from {filename})" if filename else ""
                context_str += f"Context {cnt}{filename_str}:\n{chunk_text}\n\n"
                cnt += 1

    print(f"Retrieved contexts [{len(results)} total]: {context_str[:150]}...")
    return context_str


def retrieval_result_to_dict_list(results: List[Dict]) -> List[Dict]:
    """
    Convert retrieval results to a list of dictionaries.

    Args:
        results: List of search result dictionaries

    Returns:
        List of formatted result dictionaries
    """
    result_list = []
    for i, result in enumerate(results):
        chunk_text = result.get('chunk_text', '')
        if chunk_text:
            filename = result.get('filename', '')
            source = ''.join(os.path.basename(filename).split('.')[:-1]) if filename else ''
            result_list.append({
                'id': i,
                'source': source,
                'content': chunk_text
            })

    print(f"Retrieved contexts [{len(result_list)} total]")
    if result_list:
        data_to_show = {k: str(v)[:200] for k, v in result_list[0].items()}
        print(f"First context: {data_to_show}")
    return result_list


# ============================================================================
# Embedding Models (Cost-Free, No PyTorch)
# ============================================================================

class ONNXEmbedding:
    """
    ONNX-based sentence embedding model.
    Cost-free, no PyTorch required - uses onnxruntime.
    """

    # Model configurations
    MODEL_CONFIGS = {
        "all-MiniLM-L6-v2": {
            "repo": "sentence-transformers/all-MiniLM-L6-v2",
            "dim": 384,
        },
        "all-mpnet-base-v2": {
            "repo": "sentence-transformers/all-mpnet-base-v2",
            "dim": 768,
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dim": 384,
        },
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ONNX embedding model.

        Args:
            model_name: Model name (without 'onnx:' prefix)
        """
        self.model_name = model_name
        config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["all-MiniLM-L6-v2"])
        self.embedding_dim = config["dim"]

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config["repo"])
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                config["repo"],
                export=True,  # Export to ONNX if not already
            )
            self._available = True
            print(f"Initialized ONNX embedding: {model_name} (dim={self.embedding_dim})")

        except ImportError as e:
            print(f"ONNX dependencies not available: {e}")
            print("Install with: pip install optimum[onnxruntime] transformers")
            self._available = False
            raise

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1),
            token_embeddings.shape
        ).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        if not self._available:
            raise RuntimeError("ONNX model not available")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype('float32')

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query text

        Returns:
            Numpy array of query embedding
        """
        return self.encode([query])


class TFIDFEmbedding:
    """
    TF-IDF + SVD based embedding model.
    Completely cost-free, no PyTorch, no API calls.
    Uses scikit-learn.
    """

    def __init__(self, embedding_dim: int = 384, model_path: str = None):
        """
        Initialize TF-IDF embedding model.

        Args:
            embedding_dim: Output embedding dimension
            model_path: Path to save/load the fitted model
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        self.embedding_dim = embedding_dim
        self.model_path = model_path or os.path.join(RAG_FAISS_INDEX_DIR, "tfidf_model.pkl")
        self._fitted = False

        # Create pipeline
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('svd', self.svd),
        ])

        # Try to load existing model
        if os.path.exists(self.model_path):
            self._load_model()

        print(f"Initialized TF-IDF embedding (dim={self.embedding_dim})")

    def _save_model(self):
        """Save the fitted model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'svd': self.svd,
                'pipeline': self.pipeline,
            }, f)
        print(f"Saved TF-IDF model to {self.model_path}")

    def _load_model(self):
        """Load a fitted model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.svd = data['svd']
                self.pipeline = data['pipeline']
                self._fitted = True
            print(f"Loaded TF-IDF model from {self.model_path}")
        except Exception as e:
            print(f"Failed to load TF-IDF model: {e}")

    def fit(self, texts: List[str]):
        """
        Fit the TF-IDF + SVD model on corpus.

        Args:
            texts: List of texts to fit on
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        print("Fitting TF-IDF + SVD model...")

        # First fit TF-IDF to get number of features
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_features = tfidf_matrix.shape[1]

        # Adjust SVD components if necessary
        actual_components = min(self.embedding_dim, n_features - 1, len(texts) - 1)
        if actual_components < 1:
            actual_components = 1

        if actual_components != self.svd.n_components:
            print(f"Adjusting SVD components from {self.embedding_dim} to {actual_components} (based on corpus size)")
            self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
            self.embedding_dim = actual_components
            self.pipeline = Pipeline([
                ('tfidf', self.vectorizer),
                ('svd', self.svd),
            ])

        # Fit SVD on the already-fitted TF-IDF matrix
        self.svd.fit(tfidf_matrix)
        self._fitted = True
        self._save_model()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Not used, kept for API compatibility

        Returns:
            Numpy array of embeddings
        """
        if not self._fitted:
            # Auto-fit on the input texts
            self.fit(texts)

        # Transform: TF-IDF then SVD
        tfidf_matrix = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf_matrix)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings.astype('float32')

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query text

        Returns:
            Numpy array of query embedding
        """
        if not self._fitted:
            raise RuntimeError("TF-IDF model not fitted. Call encode() with corpus first.")
        return self.encode([query])


class OpenAIEmbedding:
    """
    OpenAI embedding model wrapper using langchain_openai.
    API-based, costs money. Use only if explicitly requested.
    """

    DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding model.

        Args:
            model_name: OpenAI embedding model name
        """
        from langchain_openai import OpenAIEmbeddings
        self.model_name = model_name
        self.model = OpenAIEmbeddings(model=model_name)
        self.embedding_dim = self.DIM_MAP.get(model_name, 1536)
        print(f"Initialized OpenAI embedding: {model_name} (dim={self.embedding_dim})")

    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for API calls

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings, dtype='float32')

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query text

        Returns:
            Numpy array of query embedding
        """
        embedding = self.model.embed_query(query)
        return np.array([embedding], dtype='float32')


def create_embedding_model(model_spec: str):
    """
    Factory function to create the appropriate embedding model.

    Args:
        model_spec: Embedding model specification:
            - "openai:<model>" - OpenAI embedding (default, best quality)
            - "onnx:<model>" - ONNX model (fallback, no PyTorch)
            - "tfidf" - TF-IDF + SVD (fallback, no dependencies)

    Returns:
        Embedding model instance
    """
    if model_spec.startswith("openai:"):
        model_name = model_spec.replace("openai:", "")
        try:
            return OpenAIEmbedding(model_name)
        except Exception as e:
            print(f"OpenAI embedding not available: {e}")
            print("Falling back to ONNX embedding...")
            return _create_fallback_embedding()

    elif model_spec.startswith("onnx:"):
        model_name = model_spec.replace("onnx:", "")
        try:
            return ONNXEmbedding(model_name)
        except (ImportError, Exception) as e:
            print(f"ONNX embedding not available: {e}")
            print("Falling back to TF-IDF embedding...")
            return TFIDFEmbedding()

    elif model_spec == "tfidf":
        return TFIDFEmbedding()

    else:
        # Default: try ONNX, fallback to TF-IDF
        try:
            return ONNXEmbedding(model_spec)
        except (ImportError, Exception):
            print(f"Unknown model '{model_spec}', falling back to TF-IDF")
            return TFIDFEmbedding()


def _create_fallback_embedding():
    """Create fallback embedding model (ONNX -> TF-IDF)."""
    try:
        return ONNXEmbedding("all-MiniLM-L6-v2")
    except (ImportError, Exception) as e:
        print(f"ONNX embedding also not available: {e}")
        print("Falling back to TF-IDF embedding...")
        return TFIDFEmbedding()


# ============================================================================
# Reranking Models (Cost-Free)
# ============================================================================

class BM25Reranker:
    """
    BM25-based reranker using rank_bm25 library.
    Cost-free, no heavy dependencies.
    """

    def __init__(self, model_name: str = "bm25"):
        self.model_name = model_name
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_class = BM25Okapi
            self._available = True
            print(f"Initialized BM25 reranker (cost-free)")
        except ImportError:
            self._available = False
            print(f"Warning: rank_bm25 not installed. Install with: pip install rank-bm25")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        return re.findall(r'\w+', text.lower())

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict]:
        """
        Rerank documents using BM25 scoring.

        Args:
            query: Query string
            documents: List of document texts
            top_n: Number of top results to return

        Returns:
            List of reranked results with index and score
        """
        if not documents:
            return []

        if not self._available:
            # Fallback: return original order
            return [{'index': i, 'score': 1.0 - i * 0.01} for i in range(min(len(documents), top_n))]

        try:
            # Tokenize documents and query
            tokenized_docs = [self._tokenize(doc) for doc in documents]
            tokenized_query = self._tokenize(query)

            # Create BM25 index and get scores
            bm25 = self._bm25_class(tokenized_docs)
            scores = bm25.get_scores(tokenized_query)

            # Sort by score (descending) and return top_n
            scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            results = []
            for idx, score in scored_indices[:top_n]:
                results.append({
                    'index': idx,
                    'score': float(score),
                })
            return results

        except Exception as e:
            print(f"BM25 rerank failed: {e}, returning original order")
            return [{'index': i, 'score': 1.0 - i * 0.01} for i in range(min(len(documents), top_n))]


class NoReranker:
    """
    No-op reranker that returns documents in original order.
    """

    def __init__(self, model_name: str = "none"):
        self.model_name = model_name
        print(f"Initialized NoReranker (no reranking, using FAISS scores)")

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict]:
        """Return documents in original order (by FAISS score)."""
        results = []
        for i in range(min(len(documents), top_n)):
            results.append({
                'index': i,
                'score': 1.0 - i * 0.01,  # Preserve order
            })
        return results


class PineconeReranker:
    """
    Pinecone reranking API wrapper.
    API-based, costs money. Use only if explicitly requested.
    """

    def __init__(self, model_name: str):
        """
        Initialize Pinecone reranker.

        Args:
            model_name: Pinecone rerank model name (e.g., "cohere-rerank-3.5")
        """
        from pinecone import Pinecone
        from config import PINECONE_API_KEY
        self.model_name = model_name
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"Initialized Pinecone reranker: {model_name}")

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict]:
        """
        Rerank documents using Pinecone inference API.

        Args:
            query: Query string
            documents: List of document texts
            top_n: Number of top results to return

        Returns:
            List of reranked results with index and score
        """
        if not documents:
            return []

        # Format documents for Pinecone rerank API
        rerank_documents = [
            {"id": str(i), "content": doc}
            for i, doc in enumerate(documents)
        ]

        response = self.pc.inference.rerank(
            model=self.model_name,
            query=query,
            documents=rerank_documents,
            rank_fields=["content"],
            top_n=top_n,
            return_documents=True,
        )

        results = []
        for item in response.data:
            results.append({
                'index': int(item.index),
                'score': item.score,
            })
        return results


def create_reranker(model_spec: str):
    """
    Factory function to create the appropriate reranker.

    Args:
        model_spec: Reranker model specification:
            - "bm25" - BM25 lexical reranker (default, cost-free)
            - "none" - No reranking
            - "pinecone:<model>" - Pinecone hosted reranker (costs money)

    Returns:
        Reranker instance
    """
    if model_spec == "bm25":
        return BM25Reranker(model_spec)
    elif model_spec == "none":
        return NoReranker(model_spec)
    elif model_spec.startswith("pinecone:"):
        actual_model = model_spec.replace("pinecone:", "")
        return PineconeReranker(actual_model)
    else:
        # Default to BM25 for unknown models
        print(f"Unknown reranker model: {model_spec}, falling back to BM25")
        return BM25Reranker("bm25")


# ============================================================================
# Main RAG Class
# ============================================================================

class FAISSRAGMedicalGuide:
    """
    FAISS-based RAG system for medical guide retrieval.

    This class provides vector search functionality using FAISS for efficient
    similarity search, with configurable embedding and reranking models.

    Default configuration is completely cost-free (no PyTorch, no OpenAI):
    - Embedding: ONNX sentence-transformers (or TF-IDF fallback)
    - Reranking: BM25 (lexical matching)

    Optional API-based alternatives available:
    - Embedding: OpenAI text-embedding-3-small
    - Reranking: Pinecone hosted Cohere reranker
    """

    def __init__(
        self,
        index_name: str = RAG_MD_INDEX_NAME,
        md_dir: str = RAG_MD_DIR,
        tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        index_dir: str = RAG_FAISS_INDEX_DIR,
    ):
        """
        Initialize the FAISS RAG system.

        Args:
            index_name: Name identifier for the FAISS index
            md_dir: Directory containing markdown files
            tmp_md_data_pkl_path: Path to pickle cache file for markdown data
            embed_model: Embedding model specification (see module docstring)
            rerank_model: Reranker model specification (see module docstring)
            index_dir: Directory to store FAISS indexes
        """
        self.index_name = index_name
        self.md_dir = md_dir
        self.tmp_md_data_pkl_path = tmp_md_data_pkl_path
        self.index_dir = index_dir
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model

        # Paths for index persistence
        self.index_path = os.path.join(index_dir, f"{index_name}.index")
        self.metadata_path = os.path.join(index_dir, f"{index_name}_metadata.pkl")

        # Initialize models
        print(f"Loading embedding model: {embed_model}")
        self.embed_model = create_embedding_model(embed_model)
        self.embedding_dim = self.embed_model.embedding_dim

        print(f"Loading rerank model: {rerank_model}")
        self.rerank_model = create_reranker(rerank_model)

        # Data storage
        self.guide_data = None
        self.chunks_metadata = []  # Store chunk metadata (text, filename, etc.)
        self.index = None  # FAISS index

        # Load or build the index
        self.index = self.load_or_build_vector_database()

        self.cache_file_dir = os.path.join(EXP_CACHE_DIR, 'faiss_md_retrieval')
        os.makedirs(self.cache_file_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

    def load_medical_guide_data(self) -> List[Dict]:
        """Load markdown data from cache or process from files."""
        return load_or_cache_medical_guide_data(self.md_dir, self.tmp_md_data_pkl_path)

    def load_or_build_vector_database(self) -> faiss.Index:
        """
        Load existing FAISS index or build a new one.

        Returns:
            FAISS index object
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print(f"Loading existing FAISS index: {self.index_name}")
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            print(f"Loaded index with {index.ntotal} vectors")
            return index
        else:
            print(f"Index '{self.index_name}' does not exist. Creating a new one...")
            index = self.create_index()
            print(f"Created and loaded index: {self.index_name}")
            return index

    def create_index(self) -> faiss.Index:
        """
        Create and populate the FAISS index from markdown files.

        Returns:
            FAISS index object
        """
        if self.guide_data is None:
            self.guide_data = self.load_medical_guide_data()

        print(f"Building FAISS vector database from {len(self.guide_data)} markdown files...")

        # Split documents into chunks
        chunks_to_upsert = _split_and_format_chunks(self.guide_data)
        print(f"Total chunks to index: {len(chunks_to_upsert)}")

        # Store metadata
        self.chunks_metadata = chunks_to_upsert

        # Generate embeddings in batches
        print("Generating embeddings...")
        chunk_texts = [chunk['chunk_text'] for chunk in chunks_to_upsert]

        batch_size = 32
        all_embeddings = []
        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding chunks"):
            batch_texts = chunk_texts[i:i + batch_size]
            batch_embeddings = self.embed_model.encode(batch_texts, batch_size=batch_size)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype('float32')

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index (using Inner Product for normalized vectors = cosine similarity)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)

        # Save index and metadata
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)

        print(f"Saved FAISS index with {index.ntotal} vectors to {self.index_path}")
        return index

    def _search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform vector similarity search.

        Args:
            query: Search query string
            top_k: Number of results to retrieve

        Returns:
            List of result dictionaries with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embed_model.encode_query(query)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks_metadata):
                result = self.chunks_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)

        return results

    def _rerank(self, query: str, results: List[Dict], top_n: int) -> List[Dict]:
        """
        Rerank results using the configured reranker.

        Args:
            query: Original query string
            results: List of search results to rerank
            top_n: Number of top results to return after reranking

        Returns:
            Reranked list of results
        """
        if not results:
            return []

        # Get document texts for reranking
        documents = [r['chunk_text'] for r in results]

        # Call reranker
        rerank_results = self.rerank_model.rerank(query, documents, top_n)

        # Build reranked results
        reranked = []
        for item in rerank_results:
            result = results[item['index']].copy()
            result['rerank_score'] = item['score']
            reranked.append(result)

        return reranked

    def retrieval(
        self,
        query: str,
        query_top_k: int,
        rerank_top_n: int,
        question_id: str = '',
    ) -> List[Dict]:
        """
        Retrieve relevant context from the RAG system.

        Args:
            query: Search query string
            query_top_k: Number of initial results to retrieve
            rerank_top_n: Number of results after reranking
            question_id: Optional question identifier for caching

        Returns:
            List of context dictionaries
        """
        # Generate cache file path
        param_dict = dict(
            query_top_k=query_top_k,
            rerank_top_n=rerank_top_n,
            query=query,
            index_name=self.index_name,
            rerank_model=self.rerank_model_name,
            embed_model=self.embed_model_name,
        )
        cache_file_path = get_catch_file_path(
            cache_dir=self.cache_file_dir,
            question_id=question_id,
            question=query,
            param_dict=param_dict
        )

        # Check cache
        context = load_cache_file(
            use_cache_file=USE_CACHE_FILE,
            cache_file_path=cache_file_path,
            key="context"
        )
        if context:
            print("Loaded from cache, skipping RAG retrieval")
            return context

        # Perform search
        search_results = self._search(query, query_top_k)

        # Rerank results
        reranked_results = self._rerank(query, search_results, rerank_top_n)

        # Format output
        context = retrieval_result_to_dict_list(reranked_results)

        # Save to cache
        if USE_CACHE_FILE:
            save_cache_file(
                cache_file_path=cache_file_path,
                data_dict_to_save={'question_id': question_id, "context": context},
                param_dict=param_dict
            )

        return context

    def test(self):
        """Test retrieval functionality."""
        query = "What is the long-term prognosis for primary acute angle-closure glaucoma?"
        context = self.retrieval(query=query, query_top_k=10, rerank_top_n=3)
        print(context)


# ============================================================================
# Singleton Instance Management for Parallel Retrieval
# ============================================================================

# Module-level singleton instance and lock for thread-safe initialization
_rag_instance: Optional['FAISSRAGMedicalGuide'] = None
_rag_instance_lock = threading.Lock()

# Worker-level instance for multiprocessing (set via initializer)
_worker_rag_instance: Optional['FAISSRAGMedicalGuide'] = None


def get_shared_rag_instance(
    index_name: str = RAG_MD_INDEX_NAME,
    md_dir: str = RAG_MD_DIR,
    tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    index_dir: str = RAG_FAISS_INDEX_DIR,
    force_new: bool = False,
) -> 'FAISSRAGMedicalGuide':
    """
    Get or create a singleton RAG instance.

    This function ensures the FAISS database is built only once, even when
    called from multiple threads. Use this for thread-safe access to a shared
    RAG instance.

    Args:
        index_name: Name identifier for the FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes
        force_new: If True, create a new instance even if one exists

    Returns:
        Shared FAISSRAGMedicalGuide instance
    """
    global _rag_instance

    with _rag_instance_lock:
        if _rag_instance is None or force_new:
            print("Initializing shared RAG instance...")
            _rag_instance = FAISSRAGMedicalGuide(
                index_name=index_name,
                md_dir=md_dir,
                tmp_md_data_pkl_path=tmp_md_data_pkl_path,
                embed_model=embed_model,
                rerank_model=rerank_model,
                index_dir=index_dir,
            )
            print("Shared RAG instance initialized successfully.")

    return _rag_instance


def init_worker_rag(
    index_name: str,
    md_dir: str,
    tmp_md_data_pkl_path: str,
    embed_model: str,
    rerank_model: str,
    index_dir: str,
):
    """
    Initialize RAG instance in multiprocessing worker.

    This function should be passed as the `initializer` argument to
    `multiprocessing.Pool()` to set up each worker with its own RAG instance.
    The FAISS index is loaded from disk (not rebuilt) since it was already
    created during the main process initialization.

    Args:
        index_name: Name identifier for the FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes
    """
    global _worker_rag_instance

    print(f"Worker {os.getpid()}: Initializing RAG instance...")
    _worker_rag_instance = FAISSRAGMedicalGuide(
        index_name=index_name,
        md_dir=md_dir,
        tmp_md_data_pkl_path=tmp_md_data_pkl_path,
        embed_model=embed_model,
        rerank_model=rerank_model,
        index_dir=index_dir,
    )
    print(f"Worker {os.getpid()}: RAG instance initialized.")


def worker_retrieval(args: Tuple) -> List[Dict]:
    """
    Perform retrieval in worker process.

    This function is designed to be called via `pool.map()` or `pool.starmap()`.
    It uses the worker-level RAG instance that was initialized via `init_worker_rag()`.

    Args:
        args: Tuple of (query, query_top_k, rerank_top_n, question_id)

    Returns:
        List of context dictionaries
    """
    global _worker_rag_instance

    if _worker_rag_instance is None:
        raise RuntimeError("Worker RAG instance not initialized. "
                           "Use init_worker_rag as Pool initializer.")

    query, query_top_k, rerank_top_n, question_id = args
    return _worker_rag_instance.retrieval(
        query=query,
        query_top_k=query_top_k,
        rerank_top_n=rerank_top_n,
        question_id=question_id,
    )


def parallel_retrieval(
    queries: List[Tuple[str, int, int, str]],
    num_workers: int = 4,
    index_name: str = RAG_MD_INDEX_NAME,
    md_dir: str = RAG_MD_DIR,
    tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    index_dir: str = RAG_FAISS_INDEX_DIR,
) -> List[List[Dict]]:
    """
    Perform parallel retrieval across multiple queries using multiprocessing.

    This function first ensures the FAISS index exists (building it if needed),
    then distributes retrieval across multiple worker processes. Each worker
    loads the pre-built index, so the database is never rebuilt multiple times.

    Args:
        queries: List of tuples (query, query_top_k, rerank_top_n, question_id)
        num_workers: Number of parallel worker processes
        index_name: Name identifier for the FAISS index
        md_dir: Directory containing markdown files
        tmp_md_data_pkl_path: Path to pickle cache file
        embed_model: Embedding model specification
        rerank_model: Reranker model specification
        index_dir: Directory to store FAISS indexes

    Returns:
        List of retrieval results (one per query)

    Example:
        >>> # First, ensure the database is built (done once)
        >>> rag = get_shared_rag_instance()
        >>>
        >>> # Define queries: (query_text, top_k, rerank_n, question_id)
        >>> queries = [
        ...     ("What causes glaucoma?", 10, 3, "q1"),
        ...     ("Treatment for cataracts?", 10, 3, "q2"),
        ...     ("Symptoms of macular degeneration?", 10, 3, "q3"),
        ... ]
        >>>
        >>> # Run parallel retrieval
        >>> results = parallel_retrieval(queries, num_workers=4)
    """
    import multiprocessing as mp

    # Ensure database is built before spawning workers
    print("Ensuring FAISS index is built before parallel retrieval...")
    _ = get_shared_rag_instance(
        index_name=index_name,
        md_dir=md_dir,
        tmp_md_data_pkl_path=tmp_md_data_pkl_path,
        embed_model=embed_model,
        rerank_model=rerank_model,
        index_dir=index_dir,
    )

    print(f"Starting parallel retrieval with {num_workers} workers for {len(queries)} queries...")

    # Use spawn context for better compatibility across platforms
    ctx = mp.get_context('spawn')

    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker_rag,
        initargs=(index_name, md_dir, tmp_md_data_pkl_path,
                  embed_model, rerank_model, index_dir),
    ) as pool:
        results = pool.map(worker_retrieval, queries)

    print(f"Parallel retrieval complete. Processed {len(results)} queries.")
    return results


class FAISSRAGENMedicalGuide(FAISSRAGMedicalGuide):
    """Convenience class with default configuration for English medical guide."""

    def __init__(
        self,
        index_name: str = RAG_MD_INDEX_NAME,
        md_dir: str = RAG_MD_DIR,
        tmp_md_data_pkl_path: str = RAG_MD_TMP_DATA_PKL_PATH,
    ):
        super().__init__(
            index_name=index_name,
            md_dir=md_dir,
            tmp_md_data_pkl_path=tmp_md_data_pkl_path,
        )


