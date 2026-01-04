# RAG Techniques Library

A modular Python library implementing various Retrieval-Augmented Generation (RAG) techniques.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Modular Architecture**: Composable components for embedding, chunking, retrieval, and generation
- **Multiple Chunking Strategies**: Fixed-size, recursive, and semantic chunking
- **Advanced Retrieval**: Vector search, hybrid (BM25 + dense), context-enriched retrieval
- **Query Transformation**: Query rewriting, decomposition, step-back prompting, HyDE
- **Reranking**: LLM-based and cross-encoder reranking
- **Evaluation**: RAGAS-style metrics (Faithfulness, Relevancy, Precision, Recall)
- **Provider Agnostic**: Works with OpenAI, Azure, or any OpenAI-compatible API

## Installation

```bash
pip install -e .
```

For development with tests:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from rag_techniques import RAGPipeline

# Create pipeline
pipeline = RAGPipeline()

# Add documents
pipeline.add_documents([
    {"text": "Artificial Intelligence (AI) is transforming industries..."},
    {"text": "Machine Learning is a subset of AI..."},
])

# Query
response = pipeline.query("What is AI?")
print(response.answer)
```

## Configuration

Set environment variables or use a `.env` file:

```bash
export RAG_OPENAI_API_KEY="your-api-key"
export RAG_OPENAI_BASE_URL="https://api.openai.com/v1"  # or compatible API
export RAG_EMBEDDING_MODEL="text-embedding-3-small"
export RAG_LLM_MODEL="gpt-4o-mini"
```

## Components

### Chunking

```python
from rag_techniques import FixedSizeChunker, SemanticChunker

# Fixed-size chunking
chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(text)

# Semantic chunking (requires embeddings)
from rag_techniques import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
chunker = SemanticChunker(embedding_provider=embeddings)
chunks = chunker.chunk(text)
```

### Retrieval

```python
from rag_techniques import VectorRetriever, HybridRetriever

# Vector retrieval
retriever = VectorRetriever(vector_store, embedding_provider)
results = retriever.retrieve("query", top_k=5)

# Hybrid retrieval (BM25 + dense)
hybrid = HybridRetriever(
    vector_store,
    embedding_provider,
    documents=docs,
    alpha=0.5  # Balance between dense and sparse
)
results = hybrid.retrieve("query")
```

### Query Transformation

```python
from rag_techniques import QueryRewriter, QueryDecomposer, HyDE

# Query rewriting
rewriter = QueryRewriter()
better_query = rewriter.transform("original query")

# HyDE (Hypothetical Document Embeddings)
hyde = HyDE(llm_provider, embedding_provider)
embedding = hyde.get_query_embedding("query")
```

### Reranking

```python
from rag_techniques import LLMReranker, CrossEncoderReranker

# LLM-based reranking
reranker = LLMReranker()
reranked = reranker.rerank(query, results, top_k=3)

# Cross-encoder reranking (faster, local)
reranker = CrossEncoderReranker()
reranked = reranker.rerank(query, results)
```

### Evaluation

```python
from rag_techniques.evaluation import evaluate_response

results = evaluate_response(
    query="What is AI?",
    answer="AI is artificial intelligence...",
    context=["AI refers to..."],
    reference="Ground truth answer",  # optional
)

print(f"Faithfulness: {results['faithfulness'].score}")
print(f"Relevancy: {results['answer_relevancy'].score}")
```

## Architecture

```
rag_techniques/
├── core/
│   ├── embeddings.py      # Embedding providers
│   └── llm.py             # LLM providers
├── chunking/
│   ├── fixed.py           # Fixed-size chunking
│   └── semantic.py        # Semantic chunking
├── vectorstore/
│   ├── memory.py          # In-memory store
│   └── base.py            # Base classes
├── retrieval/
│   ├── vector.py          # Vector retrieval
│   ├── hybrid.py          # Hybrid (BM25 + dense)
│   └── context_enriched.py
├── reranking/
│   ├── llm.py             # LLM reranking
│   └── cross_encoder.py   # Cross-encoder
├── query/
│   ├── rewrite.py         # Query rewriting
│   ├── decompose.py       # Query decomposition
│   └── hyde.py            # HyDE
├── evaluation/
│   └── metrics.py         # RAGAS metrics
├── pipeline.py            # RAG pipeline
└── config.py              # Configuration
```

## References

- [RAG Paper](https://arxiv.org/abs/2005.11401) - Original RAG framework
- [RAGAS](https://arxiv.org/abs/2309.15217) - Evaluation metrics
- [HyDE](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings
- [ColBERT](https://arxiv.org/abs/2004.12832) - Token-level retrieval
- [Blended RAG](https://arxiv.org/abs/2404.07220) - Hybrid search

## License

MIT License
