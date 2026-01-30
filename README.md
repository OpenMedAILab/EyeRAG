# EyeRAG

EyeRAG is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for ophthalmology knowledge bases. It combines multiple RAG approaches including naive RAG, hypothetical document embeddings, hierarchical indexing, and LightRAG to provide comprehensive and accurate responses to medical queries.

## Features

- Multiple RAG methodologies: Naive RAG, Hypothetical Document Embeddings, Hierarchical Indexing, and LightRAG
- Support for various LLM providers (OpenAI, Anthropic, Google Gemini, XAI, DeepSeek)
- Integration with vector databases (ChromaDB, FAISS, Pinecone)
- Graph-based knowledge representation
- Medical knowledge base focused on ophthalmology
- Hybrid search capabilities combining chunk-based and knowledge graph retrieval

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EyeRAG.git
cd EyeRAG
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the environment template and configure your API keys:
```bash
cp .env.example .env
```

Edit the `.env` file to add your API keys for the LLM providers you plan to use.

## Configuration

The system is configured via the `config.py` file and environment variables in the `.env` file. Key configuration options include:

- LLM API keys and endpoints
- Vector database settings
- RAG methodology selection
- Medical knowledge base source (currently supports COS and AAO_PPP)

## Usage

### Basic Usage

```python
from eye_rag import # Import the modules you need

# Initialize the system with your configuration
# Run queries against the knowledge base
```

### Available RAG Methods

- `LLM_Response`: Direct LLM response without RAG
- `LLM_NaiveRAG_Response`: Traditional RAG approach
- `LLM_HypotheticalRAG_Response`: Hypothetical document embeddings
- `LLM_HierarchicalIndexRAG_Response`: Hierarchical indexing approach
- `LLM_LightRAG_Hybrid_Distillation_Response`: LightRAG hybrid approach

## Project Structure

```
EyeRAG/
├── eye_rag/          # Main EyeRAG modules
│   ├── graph/        # Graph-related components
│   ├── graph_node/   # Graph node definitions
│   ├── handle_results/ # Result processing utilities
│   ├── qa/           # Question answering components
│   ├── rag/          # RAG implementations
│   ├── ranking/      # Ranking algorithms
│   ├── tools/        # Utility tools
│   ├── eye_rag_utils.py # Utility functions
│   └── llm.py        # LLM integration
├── lightrag/         # LightRAG implementation
├── config.py         # Configuration settings
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
XAI_API_KEY=your_xai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

See [LICENSE](LICENSE) for licensing information.

## Acknowledgments

This project builds upon the LightRAG framework and incorporates various state-of-the-art techniques in RAG systems for domain-specific knowledge bases.