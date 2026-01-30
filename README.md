# EyeRAG

EyeRAG is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for ophthalmology knowledge bases.
It enables clinical dialogue through multiple RAG approaches including naive RAG, hypothetical document embeddings,
hierarchical indexing, and LightRAG. The system also provides evaluation and ranking capabilities to assess the quality
of different RAG methodologies for clinical applications. This release includes the COS (Clinical Ophthalmology Society)
guidelines but excludes the AAO_PPP (American Academy of Ophthalmology Preferred Practice Patterns) data due to licensing restrictions.

## Features

- Multiple RAG methodologies: Naive RAG, Hypothetical Document Embeddings, Hierarchical Indexing, and LightRAG
- Clinical dialogue capabilities for ophthalmology applications
- Performance evaluation and ranking of different RAG methodologies
- Unified LLM access via [OpenRouter](https://openrouter.ai/) (supports OpenAI, Anthropic, Google, xAI, Meta, DeepSeek)
- Integration with vector databases (FAISS)
- Medical knowledge base focused on ophthalmology (COS guidelines included, AAO_PPP not included in this release)
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

Edit the `.env` file to add your API keys. This project uses **OpenRouter** as the unified API gateway to access multiple LLM providers with a single API key:

```bash
# Required: Get your OpenRouter API key from https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional: Direct DeepSeek API access (faster, cheaper)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

OpenRouter provides access to multiple LLMs including GPT-4o, Claude Sonnet 4, Gemini, Llama, Grok, and DeepSeek through a single API.

4. The medical guide documents are already included in `Data/RAG/COS/medical_guide_markdown/` directory. Note: This release includes only the COS (Clinical Ophthalmology Society) guidelines, not the AAO_PPP (American Academy of Ophthalmology Preferred Practice Patterns) data due to licensing restrictions.

## Usage

### Initial Setup

First, construct the medical guide knowledge base:

```bash
python eye_rag/tools/construct_medical_guide_db.py
```

This will create the necessary knowledge graphs and indexes from the medical guide documents.

### Clinical Dialogue Demo

To see a demonstration of the clinical dialogue system with example patient data:

```bash
python -m eye_rag.tools.demo_clinical_dialogue
```

This demonstrates the complete EyeRAG pipeline:
1. Clinical patient data input
2. Question processing and rewriting
3. LightRAG retrieval with context distillation
4. Evidence-based response generation

### Clinical Dialogue and Evaluation

Run clinical dialogues using different RAG methods, and evaluate and rank the performance of different RAG methods:

```bash
python eye_rag/tools/clinical_dialogue.py
```

This will allow you to conduct clinical conversations with the system using various RAG methodologies, 
and compare the quality of responses from different approaches and provide rankings based on evaluation metrics.

### Available RAG Methods

- `NaiveRAG`: Traditional RAG approach
- `HypotheticalRAG`: Hypothetical document embeddings approach
- `HiRAG`: Hierarchical indexing approach
- `LightRAG`: LightRAG hybrid approach with distillation

Note: The system supports both COS and AAO_PPP knowledge sources, but this release only includes the COS (Clinical Ophthalmology Society) guidelines due to licensing restrictions.

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
│       ├── clinical_dialogue.py           # Clinical dialogue interface
│       └── demo_clinical_dialogue.py      # Demo script with example case
├── lightrag/                # LightRAG implementation
├── Data/                    # Medical knowledge base data
│   └── RAG/
│       └── COS/             # Clinical ophthalmology guides
│           ├── medical_guide_markdown/     # Original medical guide documents
│           ├── medical_guide_markdown_summary/  # Summarized medical guide documents
│           ├── faiss_indexes/             # FAISS vector indexes
│           └── OphthaKG_COS/              # Ophthalmology knowledge graph files
├── eye_rag/ranking/rank.py  # Performance evaluation and ranking
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Configuration

The system is configured via the `config.py` file and environment variables in the `.env` file.

### API Configuration

This project uses [OpenRouter](https://openrouter.ai/) as the unified API gateway, which provides several benefits:

- **Single API key** for accessing 10+ LLM providers
- **Pay-as-you-go** pricing with no monthly commitments
- **Automatic fallbacks** and load balancing
- **Unified interface** for all models

**Supported Models via OpenRouter:**

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-3.5-Turbo |
| Anthropic | Claude Sonnet 4, Claude 3.5 Sonnet/Haiku |
| Google | Gemini 2.0 Flash, Gemini 2.5 Flash |
| Meta | Llama 3.3 8B, Llama 3.3 70B |
| xAI | Grok 3, Grok 4 |
| DeepSeek | DeepSeek Chat |

**Knowledge Sources:**

- COS (Clinical Ophthalmology Society) guidelines - Included in this release
- AAO_PPP (American Academy of Ophthalmology Preferred Practice Patterns) - Not included in this release due to licensing restrictions

### Other Configuration Options

- RAG methodology selection (`config.py`)
- Medical knowledge base source (AAO_PPP or COS) - Note: AAO_PPP data is not included in this release due to licensing restrictions
- Patient data configuration (`Data/patient_data.xlsx`) - Example patient data for clinical dialogue
- LightRAG parameters (search top-k, rerank settings)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This project builds upon the LightRAG framework and incorporates various state-of-the-art techniques in RAG systems for domain-specific knowledge bases.
The system is designed to enable clinical dialogue and provide evaluation capabilities for comparing different RAG methodologies in ophthalmology applications.
Note: This release includes only the COS (Clinical Ophthalmology Society) guidelines. The AAO_PPP (American Academy of Ophthalmology Preferred Practice Patterns) data is not included in this release due to licensing restrictions.