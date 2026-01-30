# EyeRAG

EyeRAG is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for ophthalmology knowledge bases. It combines multiple RAG approaches including naive RAG, hypothetical document embeddings, hierarchical indexing, and LightRAG to provide comprehensive and accurate responses to medical queries.

## Features

- Multiple RAG methodologies: Naive RAG, Hypothetical Document Embeddings, Hierarchical Indexing, and LightRAG
- Support for various LLM providers (OpenAI, Anthropic, Google Gemini, XAI, DeepSeek)
- Integration with vector databases (FAISS, Pinecone)
- Medical knowledge base focused on ophthalmology
- Graph-based knowledge representation
- Hybrid search capabilities combining chunk-based and knowledge graph retrieval

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OpenMedAILab/EyeRAG.git
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

## Usage

### Basic Usage

```python
from eye_rag import EyeRAG

# Initialize the system with your configuration
eye_rag = EyeRAG(config_path="config.py")

# Run queries against the knowledge base
response = eye_rag.query("What are the treatment options for glaucoma?")
print(response)
```

### Available RAG Methods

- `NaiveRAG`: Traditional RAG approach
- `HypotheticalRAG`: Hypothetical document embeddings approach
- `HiRAG`: Hierarchical indexing approach
- `LightRAG`: LightRAG hybrid approach with distillation

## Project Structure

```
EyeRAG/
├── eye_rag/                 # Main EyeRAG modules
│   ├── graph/               # Graph-based RAG implementations
│   ├── graph_node/          # Individual graph nodes
│   ├── handle_results/      # Result processing utilities
│   ├── qa/                  # Question answering components
│   ├── rag/                 # RAG implementations
│   ├── ranking/             # Ranking algorithms
│   └── tools/               # Utility tools
├── lightrag/                # LightRAG implementation
├── Data/                    # Medical knowledge base data
│   └── RAG/
│       └── COS/             # Clinical ophthalmology guides
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Configuration

The system is configured via the `config.py` file and environment variables in the `.env` file. Key configuration options include:

- LLM API keys and endpoints
- Vector database settings
- RAG methodology selection
- Medical knowledge base source

## Contributing

We welcome contributions to EyeRAG! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use EyeRAG in your research, please cite:

```
@article{eyerag2025,
  title={EyeRAG: Advanced Retrieval-Augmented Generation for Ophthalmology Knowledge Bases},
  author={EyeRAG Contributors},
  journal={Open Source Software},
  year={2025}
}
```

## Acknowledgments

This project builds upon the LightRAG framework and incorporates various state-of-the-art techniques in RAG systems for domain-specific knowledge bases.