"""
Quick start example for RAG Techniques library.

Demonstrates basic usage of the RAGPipeline with sample documents.
"""

from rag_techniques import RAGPipeline, configure

# Configure settings (or use environment variables)
# configure(openai_api_key="your-key", openai_base_url="https://api.openai.com/v1")

# Sample documents about AI
SAMPLE_DOCUMENTS = [
    {
        "text": """Artificial Intelligence (AI) is a broad field of computer science focused on 
        creating intelligent machines that can perform tasks typically requiring human intelligence. 
        This includes learning, reasoning, problem-solving, perception, and language understanding. 
        AI systems can be classified into narrow AI, which is designed for specific tasks, and 
        general AI, which would have human-like cognitive abilities across various domains.""",
        "source": "ai_intro.txt"
    },
    {
        "text": """Machine Learning (ML) is a subset of AI that enables systems to learn and 
        improve from experience without being explicitly programmed. ML algorithms use statistical 
        techniques to identify patterns in data. Common types include supervised learning 
        (learning from labeled examples), unsupervised learning (finding hidden patterns), 
        and reinforcement learning (learning through trial and error with rewards).""",
        "source": "ml_basics.txt"
    },
    {
        "text": """Deep Learning is a specialized form of machine learning using neural networks 
        with many layers (hence 'deep'). These networks can automatically learn hierarchical 
        representations of data. Deep learning has achieved remarkable success in image recognition, 
        natural language processing, and speech recognition. Key architectures include CNNs for 
        images and Transformers for text.""",
        "source": "deep_learning.txt"
    },
    {
        "text": """Large Language Models (LLMs) are AI systems trained on vast amounts of text data 
        to understand and generate human-like text. Models like GPT, Claude, and LLaMA use the 
        Transformer architecture and can perform tasks like translation, summarization, question 
        answering, and code generation. They exhibit emergent capabilities as they scale.""",
        "source": "llm_overview.txt"
    },
    {
        "text": """Retrieval-Augmented Generation (RAG) combines information retrieval with 
        language model generation. Instead of relying solely on training data, RAG systems 
        retrieve relevant documents from a knowledge base and use them to ground their responses. 
        This approach reduces hallucinations and allows for dynamic knowledge updates without 
        retraining the model.""",
        "source": "rag_explained.txt"
    },
]


def main():
    """Run the quick start example."""
    print("=" * 60)
    print("RAG Techniques - Quick Start Example")
    print("=" * 60)
    
    # Create pipeline with default settings
    pipeline = RAGPipeline()
    
    # Add documents
    print("\n📚 Adding documents...")
    num_chunks = pipeline.add_documents(SAMPLE_DOCUMENTS)
    print(f"   Created {num_chunks} chunks from {len(SAMPLE_DOCUMENTS)} documents")
    
    # Sample queries
    queries = [
        "What is the difference between AI and Machine Learning?",
        "How does RAG help reduce hallucinations?",
        "What are the types of machine learning?",
    ]
    
    print("\n" + "=" * 60)
    print("Running Queries")
    print("=" * 60)
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 40)
        
        response = pipeline.query(query, top_k=3)
        
        print(f"💡 Answer: {response.answer}")
        print(f"\n📖 Sources ({len(response.sources)} chunks retrieved):")
        for i, source in enumerate(response.sources[:2], 1):
            preview = source.text[:100] + "..." if len(source.text) > 100 else source.text
            print(f"   {i}. [{source.score:.3f}] {preview}")
        
        print()
    
    print("=" * 60)
    print("✅ Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
