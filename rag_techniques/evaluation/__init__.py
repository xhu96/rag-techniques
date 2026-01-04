"""
Evaluation framework for RAG systems.

Implements RAGAS-inspired metrics:
- Faithfulness: Hallucination detection
- AnswerRelevancy: Query-Response alignment
- ContextPrecision: Signal-to-noise ratio in retrieval
- ContextRecall: Coverage of ground truth info
"""

from rag_techniques.evaluation.metrics import (
    Metric,
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    evaluate_response,
)

__all__ = [
    "Metric",
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    "evaluate_response",
]
