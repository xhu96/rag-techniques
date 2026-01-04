"""
Benchmark script to compare RAG techniques.

Runs a comparative analysis of different pipelines:
- Simple RAG (Fixed Chunking)
- Hybrid RAG (BM25 + Dense)
- Different chunk sizes
Measures Faithfulness, Relevancy (via RAGAS), and Latency.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from rag_techniques import (
    RAGPipeline,
    FixedSizeChunker,
    SemanticChunker,
    VectorRetriever,
    HybridRetriever,
    OpenAIEmbeddings,
    InMemoryVectorStore,
)
from rag_techniques.evaluation import evaluate_response


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    technique: str
    query: str
    answer: str
    reference: str
    faithfulness: float
    relevancy: float
    latency_ms: float
    num_sources: int


def load_validation_data(path: str) -> List[Dict[str, Any]]:
    """Load validation data from JSON file."""
    with open(path) as f:
        return json.load(f)


def create_pipelines(documents: List[Dict[str, Any]], embedding_provider) -> Dict[str, RAGPipeline]:
    """Create different pipeline configurations for comparison."""
    pipelines = {}
    
    # 1. Simple RAG with fixed chunking
    store1 = InMemoryVectorStore()
    pipeline1 = RAGPipeline(
        embedding_provider=embedding_provider,
        chunker=FixedSizeChunker(chunk_size=500, chunk_overlap=100),
        vector_store=store1,
    )
    pipeline1.add_documents(documents)
    pipelines["simple_rag_500"] = pipeline1
    
    # 2. Simple RAG with larger chunks
    store2 = InMemoryVectorStore()
    pipeline2 = RAGPipeline(
        embedding_provider=embedding_provider,
        chunker=FixedSizeChunker(chunk_size=1000, chunk_overlap=200),
        vector_store=store2,
    )
    pipeline2.add_documents(documents)
    pipelines["simple_rag_1000"] = pipeline2
    
    # 3. Hybrid retrieval (requires documents for BM25)
    store3 = InMemoryVectorStore()
    chunker3 = FixedSizeChunker(chunk_size=500, chunk_overlap=100)
    pipeline3 = RAGPipeline(
        embedding_provider=embedding_provider,
        chunker=chunker3,
        vector_store=store3,
    )
    pipeline3.add_documents(documents)
    
    # Create hybrid retriever
    chunk_docs = [{"text": c.text, "id": f"chunk_{c.index}"} for c in pipeline3._chunks]
    hybrid_retriever = HybridRetriever(
        vector_store=store3,
        embedding_provider=embedding_provider,
        documents=chunk_docs,
        alpha=0.5,
    )
    pipeline3.retriever = hybrid_retriever
    pipelines["hybrid_rag"] = pipeline3
    
    return pipelines


def run_benchmark(
    pipelines: Dict[str, RAGPipeline],
    validation_data: List[Dict[str, Any]],
    max_queries: int = 10,
) -> List[BenchmarkResult]:
    """Run benchmark on all pipelines."""
    results = []
    
    for query_data in validation_data[:max_queries]:
        query = query_data["question"]
        reference = query_data.get("ideal_answer", "")
        
        for technique_name, pipeline in pipelines.items():
            print(f"  Running {technique_name} on: {query[:50]}...")
            
            # Measure latency
            start_time = time.time()
            response = pipeline.query(query, top_k=3)
            latency_ms = (time.time() - start_time) * 1000
            
            # Evaluate response
            try:
                context = [s.text for s in response.sources]
                eval_results = evaluate_response(
                    query=query,
                    answer=response.answer,
                    context=context,
                    reference=reference if reference else None,
                )
                faithfulness = eval_results.get("faithfulness", {}).score if "faithfulness" in eval_results else 0.0
                relevancy = eval_results.get("answer_relevancy", {}).score if "answer_relevancy" in eval_results else 0.0
            except Exception as e:
                print(f"    Evaluation error: {e}")
                faithfulness = 0.0
                relevancy = 0.0
            
            results.append(BenchmarkResult(
                technique=technique_name,
                query=query,
                answer=response.answer[:200],
                reference=reference[:200] if reference else "",
                faithfulness=faithfulness,
                relevancy=relevancy,
                latency_ms=latency_ms,
                num_sources=len(response.sources),
            ))
    
    return results


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Group by technique
    techniques = {}
    for r in results:
        if r.technique not in techniques:
            techniques[r.technique] = []
        techniques[r.technique].append(r)
    
    print(f"\n{'Technique':<20} {'Faithfulness':>12} {'Relevancy':>10} {'Latency (ms)':>12} {'Queries':>8}")
    print("-" * 70)
    
    for technique, tech_results in techniques.items():
        avg_faith = sum(r.faithfulness for r in tech_results) / len(tech_results)
        avg_rel = sum(r.relevancy for r in tech_results) / len(tech_results)
        avg_latency = sum(r.latency_ms for r in tech_results) / len(tech_results)
        
        print(f"{technique:<20} {avg_faith:>12.2f} {avg_rel:>10.2f} {avg_latency:>12.1f} {len(tech_results):>8}")
    
    print("-" * 70)
    print("\nBest technique by metric:")
    
    best_faith = max(techniques.items(), key=lambda x: sum(r.faithfulness for r in x[1]) / len(x[1]))
    best_rel = max(techniques.items(), key=lambda x: sum(r.relevancy for r in x[1]) / len(x[1]))
    best_latency = min(techniques.items(), key=lambda x: sum(r.latency_ms for r in x[1]) / len(x[1]))
    
    print(f"  - Faithfulness: {best_faith[0]}")
    print(f"  - Relevancy: {best_rel[0]}")
    print(f"  - Latency: {best_latency[0]}")


def main():
    """Run the benchmark."""
    print("=" * 80)
    print("RAG Techniques Benchmark")
    print("=" * 80)
    
    # Sample documents (replace with your own)
    documents = [
        {
            "text": """Artificial Intelligence (AI) is a broad field of computer science 
            focused on creating intelligent machines that can perform tasks typically 
            requiring human intelligence. This includes learning, reasoning, problem-solving, 
            perception, and language understanding. AI systems can be classified into 
            narrow AI (designed for specific tasks) and general AI (human-like cognitive 
            abilities across various domains).""",
            "source": "ai_intro.txt"
        },
        {
            "text": """Machine Learning (ML) is a subset of AI that enables systems to learn 
            and improve from experience without being explicitly programmed. ML algorithms 
            use statistical techniques to identify patterns in data. Common types include 
            supervised learning, unsupervised learning, and reinforcement learning.""",
            "source": "ml_basics.txt"
        },
        {
            "text": """Deep Learning uses neural networks with many layers to learn 
            hierarchical representations of data. It has achieved remarkable success in 
            image recognition, natural language processing, and speech recognition. 
            Key architectures include CNNs for images and Transformers for text.""",
            "source": "deep_learning.txt"
        },
        {
            "text": """Retrieval-Augmented Generation (RAG) combines information retrieval 
            with language model generation. Instead of relying solely on training data, 
            RAG systems retrieve relevant documents from a knowledge base and use them 
            to ground their responses. This reduces hallucinations and allows for 
            dynamic knowledge updates without retraining.""",
            "source": "rag_explained.txt"
        },
    ]
    
    # Validation queries
    validation_data = [
        {
            "question": "What is the difference between AI and Machine Learning?",
            "ideal_answer": "AI is a broad field focused on creating intelligent machines, while Machine Learning is a subset of AI that enables systems to learn from experience without explicit programming."
        },
        {
            "question": "How does RAG reduce hallucinations?",
            "ideal_answer": "RAG reduces hallucinations by retrieving relevant documents from a knowledge base and using them to ground the language model's responses, rather than relying solely on training data."
        },
        {
            "question": "What are the types of machine learning?",
            "ideal_answer": "The common types of machine learning are supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error)."
        },
    ]
    
    print("\n📚 Creating pipeline configurations...")
    
    # Note: This requires API key for embeddings
    try:
        embedding_provider = OpenAIEmbeddings()
        pipelines = create_pipelines(documents, embedding_provider)
        
        print(f"\n🔬 Running benchmark with {len(validation_data)} queries...")
        results = run_benchmark(pipelines, validation_data)
        
        print_summary(results)
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("\nTo run this benchmark, set your API key:")
        print("  export RAG_OPENAI_API_KEY='your-key'")
        print("\nOr run with mock embeddings for testing.")


if __name__ == "__main__":
    main()
