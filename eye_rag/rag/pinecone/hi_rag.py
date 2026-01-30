import asyncio
import os

from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from tqdm.auto import tqdm

from config import EXP_CACHE_DIR, PINECONE_API_KEY
from eye_rag.eye_rag_utils import get_catch_file_path, load_cache_file, save_cache_file
from eye_rag.rag.pinecone.naive_rag import (
    RAGMedicalGuide,
    RAG_MD_DIR,
    RAG_MD_TMP_DATA_PKL_PATH,
    retrieval_result_to_dict_list,
)
from eye_rag.rag.rag_util import _split_and_format_chunks

# Constants
HIERARCHICAL_INDEX_NAME_SUMMARY = "md-medical-guide-summary"
HIERARCHICAL_INDEX_NAME_CHUNKS = "md-medical-guide-chunks-v2"
HIERARCHICAL_EMBED_MODEL = "llama-text-embed-v2"
MEDICAL_GUIDE_PDF_CHUNK_VS = "Data/RAG/medical_guide_pdf_chunks_vector_store"

USE_CACHE_FILE = True

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_chunk_key(filename):
    """Extract chunk key from filename."""
    return os.path.basename(filename).replace(".md", '')


class HierarchicalRAG(RAGMedicalGuide):
    """Hierarchical RAG with document summaries and chunk retrieval."""

    def __init__(
        self,
        index_name_summary=HIERARCHICAL_INDEX_NAME_SUMMARY,
        index_name_chunks=HIERARCHICAL_INDEX_NAME_CHUNKS,
        md_dir=RAG_MD_DIR,
        tmp_md_data_pkl_path=RAG_MD_TMP_DATA_PKL_PATH,
        embed_model=HIERARCHICAL_EMBED_MODEL,
    ):
        super().__init__(
            index_name=index_name_chunks,
            md_dir=md_dir,
            tmp_md_data_pkl_path=tmp_md_data_pkl_path,
            embed_model=embed_model,
        )

        self.index_name_summary = index_name_summary
        self.index_name_chunks = index_name_chunks

        self.summary_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=4000)
        self.summary_chain = load_summarize_chain(self.summary_llm, chain_type="map_reduce")

        self.summary_index = None
        self.chunks_index = self.index
        self.cache_file_dir = os.path.join(EXP_CACHE_DIR, 'pinecone_md_hi_retrieval')

    def create_index(self):
        """Create chunks vector database."""
        if self.guide_data is None:
            self.guide_data = self.load_medical_guide_data()

        print(f"Building chunks vector database from {len(self.guide_data)} markdown files...")

        chunks_to_upsert = _split_and_format_chunks(self.guide_data)

        if not self.pc.has_index(self.index_name):
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": self.embed_model,
                    "field_map": {"text": "chunk_text"}
                }
            )

        index = self.pc.Index(self.index_name)

        batch_size = 5
        for i in tqdm(range(0, len(chunks_to_upsert), batch_size), desc="Upserting chunks"):
            i_end = min(len(chunks_to_upsert), i + batch_size)
            meta_batch = chunks_to_upsert[i:i_end]
            index.upsert_records("ns1", meta_batch)

        index.describe_index_stats()
        return index

    async def create_document_summaries(self):
        """Create summaries for each document and save to text files."""
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

    async def load_or_build_summary_index_async(self):
        """Async version of load_or_build_summary_index."""
        index_name = self.index_name_summary

        if self.pc.list_indexes() and index_name in [x["name"] for x in self.pc.list_indexes().indexes]:
            index = self.pc.Index(index_name)
            print(f"Loaded existing summary index: {index_name}")
        else:
            print(f"Summary index '{index_name}' does not exist. Creating a new one...")
            index = await self.create_summary_index()
            print(f"Created and loaded summary index: {index_name}")
        return index

    async def create_summary_index(self):
        """Create summary index."""
        if self.guide_data is None:
            self.guide_data = self.load_medical_guide_data()

        print(f"Building summary vector database from {len(self.guide_data)} markdown files...")

        summaries = await self.create_document_summaries()

        if not self.pc.has_index(self.index_name_summary):
            self.pc.create_index_for_model(
                name=self.index_name_summary,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": self.embed_model,
                    "field_map": {"text": "summary_text"}
                }
            )

        index = self.pc.Index(self.index_name_summary)

        batch_size = 5
        for i in tqdm(range(0, len(summaries), batch_size), desc="Upserting summaries"):
            i_end = min(len(summaries), i + batch_size)
            meta_batch = summaries[i:i_end]
            index.upsert_records("ns1", meta_batch)

        index.describe_index_stats()
        return index

    async def initialize_summary_index(self):
        """Initialize summary index asynchronously."""
        if self.summary_index is None:
            self.summary_index = await self.load_or_build_summary_index_async()

    async def _retrieve_relevant_documents(self, query, summary_top_k):
        """Retrieve relevant documents using summary search."""
        await self.initialize_summary_index()

        summary_results = self.summary_index.search(
            namespace="ns1",
            query={
                "top_k": summary_top_k + 2,
                "inputs": {'text': query}
            },
            rerank={
                "model": self.reranking_embed_model,
                "top_n": summary_top_k,
                "rank_fields": ["summary_text"]
            },
            fields=["summary_text", "filename"]
        )
        return summary_results

    async def hi_retrieval(
        self,
        query: str,
        summary_top_k: int,
        chunk_top_k: int,
        rerank_top_n: int,
        question_id: str = '',
    ) -> list:
        """Hierarchical retrieval: first find relevant docs, then retrieve chunks."""
        param_dict = dict(
            summary_top_k=summary_top_k,
            chunk_top_k=chunk_top_k,
            rerank_top_n=rerank_top_n,
            query=query,
            index_name_summary=self.index_name_summary,
            index_name_chunks=self.index_name_chunks,
            rerank_model=self.reranking_embed_model,
            embed_model=self.embed_model,
        )
        cache_file_path = get_catch_file_path(
            cache_dir=self.cache_file_dir,
            question_id=question_id,
            question=query,
            param_dict=param_dict,
        )

        context = load_cache_file(
            use_cache_file=USE_CACHE_FILE,
            cache_file_path=cache_file_path,
            key="context",
        )
        if context:
            print("Loading from cache, skipping hierarchical retrieval")
            return context

        # Step 1: Retrieve relevant documents using summaries
        summary_results = await self._retrieve_relevant_documents(query, summary_top_k)
        relevant_docs = [
            hit['fields']['filename']
            for hit in summary_results['result']['hits']
            if 'filename' in hit['fields']
        ]

        print(f"Found {len(relevant_docs)} relevant documents from summary search")
        for k, doc in enumerate(relevant_docs):
            print(f"{k + 1}. {doc}")

        if not relevant_docs:
            return []

        # Step 2: Retrieve chunks from relevant documents
        chunk_results = self.chunks_index.search(
            namespace="ns1",
            query={
                "top_k": chunk_top_k,
                "inputs": {'text': query}
            },
            rerank={
                "model": self.reranking_embed_model,
                "top_n": rerank_top_n,
                "rank_fields": ["chunk_text"]
            },
            fields=["chunk_text", "filename"],
        )

        # Filter chunks by relevant document IDs
        filtered_hits = [
            hit for hit in chunk_results['result']['hits']
            if hit['fields'].get('filename') in relevant_docs
        ]

        filtered_chunk_results = {
            'result': {'hits': filtered_hits[:rerank_top_n]}
        }

        context = retrieval_result_to_dict_list(filtered_chunk_results)

        if USE_CACHE_FILE:
            save_cache_file(
                cache_file_path=cache_file_path,
                data_dict_to_save={'question_id': question_id, "context": context},
                param_dict=param_dict,
            )

        return context

    async def test(self):
        """Test retrieval functionality."""
        query = "What is the long-term prognosis for primary acute angle-closure glaucoma?"
        context = await self.retrieval(query=query, rerank_top_n=3, query_top_k=10)
        print(context)


def rerank(query: str, documents: list, top_n: int) -> list:
    """Rerank documents using Pinecone's reranking API."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    rerank_documents = [
        {
            "id": str(i),
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for i, doc in enumerate(documents)
    ]

    results = pc.inference.rerank(
        model="cohere-rerank-3.5",
        query=query,
        rank_fields=["page_content"],
        documents=rerank_documents,
        top_n=top_n,
        return_documents=True,
    )

    reranked_docs = []
    for result in results.data:
        original_index = int(result.document.id)
        reranked_docs.append(documents[original_index])

    return reranked_docs


