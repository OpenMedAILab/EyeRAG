# EyeRAG

EyeRAG is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for ophthalmology knowledge bases.
It enables clinical dialogue through multiple RAG approaches including naive RAG, hypothetical document embeddings,
hierarchical indexing, and LightRAG. The system also provides evaluation and ranking capabilities to assess the quality
of different RAG methodologies for clinical applications.

## Features

- Multiple RAG methodologies: Naive RAG, Hypothetical Document Embeddings, Hierarchical Indexing, and LightRAG
- Clinical dialogue capabilities for ophthalmology applications
- Performance evaluation and ranking of different RAG methodologies
- Support for various LLM providers (OpenAI, Anthropic, Google Gemini, XAI, DeepSeek)
- Integration with vector databases (FAISS)
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

4. Download or prepare your medical guide documents in the `Data/RAG/COS/medical_guide_markdown/` directory.

## Usage

### Initial Setup

First, construct the medical guide knowledge base:

```bash
python eye_rag/tools/construct_medical_guide_db.py
```

This will create the necessary knowledge graphs and indexes from the medical guide documents.

### Clinical Dialogue and Evaluation

After constructing the knowledge base, you can run clinical dialogues using different RAG methods:

```bash
python eye_rag/tools/clinical_dialogue.py
```

This will allow you to conduct clinical conversations with the system using various RAG methodologies.

### Performance Evaluation

To evaluate and rank the performance of different RAG methods:

```bash
python eye_rag/ranking/rank.py
```

This will compare the quality of responses from different approaches and provide rankings based on evaluation metrics.

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
│   ├── llm.py               # LLM integration
│   ├── qa/                  # Question answering components
│   ├── rag/                 # RAG implementations
│   ├── ranking/             # Ranking algorithms
│   └── tools/               # Utility tools
│       ├── construct_medical_guide_db.py  # Build medical guide knowledge base
│       └── clinical_dialogue.py          # Clinical dialogue interface
├── lightrag/                # LightRAG implementation
├── Data/                    # Medical knowledge base data
│   └── RAG/
│       └── COS/             # Clinical ophthalmology guides
│           └── medical_guide_markdown/   # Medical guide documents
├── eye_rag/ranking/rank.py  # Performance evaluation and ranking
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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This project builds upon the LightRAG framework and incorporates various state-of-the-art techniques in RAG systems for domain-specific knowledge bases.
The system is designed to enable clinical dialogue and provide evaluation capabilities for comparing different RAG methodologies in ophthalmology applications.