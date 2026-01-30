import os
import pickle
from pinecone import Pinecone
from tqdm.auto import tqdm

from config import EXP_CACHE_DIR
from eye_rag.eye_rag_utils import get_catch_file_path, load_cache_file, save_cache_file
from eye_rag.rag.rag_util import load_markdown_files, _split_and_format_chunks

USE_CACHE_FILE = True
RAG_MD_INDEX_NAME = 'md-medical-guide-llama-text-embed-1000-v3'
RAG_MD_DIR = "Data/RAG/medical_guide_markdown"
RAG_MD_TMP_DATA_PKL_PATH = "Data/RAG/medical_guide_md_data20250830.pkl"


def load_or_cache_medical_guide_data(md_dir, tmp_md_data_pkl_path):
    """
    Load markdown data from cache or process from files.

    Args:
        md_dir (str): Directory containing markdown files
        tmp_md_data_pkl_path (str): Path to pickle cache file

    Returns:
        list: List of medical guide data dictionaries
    """
    # Check if cached data exists
    if os.path.exists(tmp_md_data_pkl_path):
        print("Loading cached markdown data...")
        with open(tmp_md_data_pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Loading and processing markdown files for the first time...")
        data = load_markdown_files(md_dir)
        # Save to cache
        os.makedirs(os.path.dirname(tmp_md_data_pkl_path), exist_ok=True)
        with open(tmp_md_data_pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached markdown data to {tmp_md_data_pkl_path}")

    return data


def format_retrieval_contexts(reranked_results):
    """
    Format retrieval results into a readable context string.

    Args:
        reranked_results: Search results from Pinecone index

    Returns:
        str: Formatted context string with numbered contexts
    """
    contexts = [hit['fields']['chunk_text'] for hit in reranked_results['result']['hits']]
    context_str = ""

    if contexts:
        cnt = 1
        for i, context in enumerate(contexts):
            if context:
                # Get filename if available
                filename = ""
                if 'filename' in reranked_results['result']['hits'][i]['fields']:
                    filename = f" (from {reranked_results['result']['hits'][i]['fields']['filename']})"
                context_str += f"Context {cnt}{filename}:\n{context}\n\n"
                cnt += 1

    print(f"Retrieved contexts [{len(contexts)} total]: {context_str[:150]}...")
    return context_str


def retrieval_result_to_dict_list(reranked_results):
    """
    Format retrieval results into a readable context string.

    Args:
        reranked_results: Search results from Pinecone index

    Returns:
        str: Formatted context string with numbered contexts
    """
    contexts = [hit['fields']['chunk_text'] for hit in reranked_results['result']['hits']]
    result = []

    if contexts:
        for i, context in enumerate(contexts):
            if context:
                # Get filename if available
                filename = ""
                if 'filename' in reranked_results['result']['hits'][i]['fields']:
                    filename = f"{reranked_results['result']['hits'][i]['fields']['filename']}"
                result.append({
                    'id': i,
                    'source': ''.join(os.path.basename(filename).split('.')[:-1]),
                    'content': context
                })

    print(f"Retrieved contexts [{len(contexts)} total]")
    if len(result):
        data_to_show = {k: str(v)[:200] for k, v in result[0].items()}
        print(f"The first context is {data_to_show}")
    return result


class RAGMedicalGuide:
    def __init__(self,
                 index_name,
                 md_dir,
                 tmp_md_data_pkl_path,
                 embed_model="llama-text-embed-v2",
                 ):
        from config import PINECONE_API_KEY
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            environment="gcp-starter"
        )
        self.index_name = index_name
        self.embed_model = embed_model
        self.guide_data = None
        self.md_dir = md_dir
        self.tmp_md_data_pkl_path = tmp_md_data_pkl_path  # Renamed variable

        self.index = self.load_or_build_vector_database()
        self.reranking_embed_model = "cohere-rerank-3.5"

        self.cache_file_dir = os.path.join(EXP_CACHE_DIR, 'pinecone_md_retrieval')

    def load_medical_guide_data(self):
        """Load markdown data from cache or process from files."""
        return load_or_cache_medical_guide_data(self.md_dir, self.tmp_md_data_pkl_path)

    def load_or_build_vector_database(self):
        # Initialize Pinecone connection
        index_name = self.index_name

        # Check if the index exists
        if self.pc.list_indexes() and index_name in [x["name"] for x in self.pc.list_indexes().indexes]:
            # Load existing index
            index = self.pc.Index(index_name)
            print(f"Loaded existing index: {index_name}")
        else:
            # Code block to create the index
            print(f"Index '{index_name}' does not exist. Creating a new one...")
            index = self.create_index()
            print(f"Created and loaded index: {index_name}")
        return index

    def create_index(self):
        if self.guide_data is None:
            self.guide_data = self.load_medical_guide_data()

        print(f"Building vector database from {len(self.guide_data)} markdown files...")
        index_name = self.index_name

        # Call the appropriate chunk processing method
        chunks_to_upsert = _split_and_format_chunks(self.guide_data)

        pc = self.pc
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": self.embed_model,
                    "field_map": {"text": "chunk_text"}
                }
            )

        index = pc.Index(index_name)

        batch_size = 5
        for i in tqdm(range(0, len(chunks_to_upsert), batch_size), desc="Upserting chunks"):
            i_end = min(len(chunks_to_upsert), i + batch_size)
            meta_batch = chunks_to_upsert[i:i_end]
            index.upsert_records("ns1", meta_batch)

        index.describe_index_stats()
        return index

    def retrieval(self, query, query_top_k, rerank_top_n, question_id=''):
        """
        Retrieve relevant context from markdown-based RAG system.
        """
        param_dict = dict(
            query_top_k=query_top_k, rerank_top_n=rerank_top_n,
            query=query,index_name=self.index_name,
            rerank_model=self.reranking_embed_model,
            embed_model=self.embed_model,
        )
        cache_file_path = get_catch_file_path(
            cache_dir=self.cache_file_dir, question_id=question_id,
            question=query, param_dict=param_dict)

        context = load_cache_file(use_cache_file=USE_CACHE_FILE, cache_file_path=cache_file_path, key="context")
        if context:
            print("Loading from cache, skipping RAG initialization")
            return context

        reranked_results = self.index.search(
            namespace="ns1",
            query={
                "top_k": query_top_k,
                "inputs": {
                    'text': query
                }
            },
            rerank={
                "model": self.reranking_embed_model,
                "top_n": rerank_top_n,
                "rank_fields": ["chunk_text"]
            },
            fields=["chunk_text", "filename"]  # Include filename in results
        )

        # context = format_retrieval_contexts(reranked_results)
        context = retrieval_result_to_dict_list(reranked_results)

        if USE_CACHE_FILE:
            save_cache_file(cache_file_path=cache_file_path,
                            data_dict_to_save={'question_id': question_id, "context": context},
                             param_dict=param_dict)
        return context


class RAGENMedicalGuide(RAGMedicalGuide):
    def __init__(self,
                 index_name=RAG_MD_INDEX_NAME,
                 md_dir=RAG_MD_DIR,
                 tmp_md_data_pkl_path=RAG_MD_TMP_DATA_PKL_PATH,
                 ):
        super().__init__(
            index_name=index_name,
            md_dir=md_dir,
            tmp_md_data_pkl_path=tmp_md_data_pkl_path,
        )

    def test(self):
        query = "What is the long-term prognosis for primary acute angle-closure glaucoma?"
        context = self.retrieval(query=query, query_top_k=10, rerank_top_n=3, )
        print(context)

