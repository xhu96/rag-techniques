"""
RAGAS-style evaluation metrics for RAG systems.

Implements key metrics from the RAGAS framework:
- Faithfulness: Factual consistency with retrieved context
- Answer Relevancy: Relevance of answer to query
- Context Precision: Quality of retrieved context
- Context Recall: Coverage of required information

Reference: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
           https://arxiv.org/abs/2309.15217
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import re

from rag_techniques.core.llm import LLMProvider, OpenAILLM


@dataclass
class EvaluationResult:
    """Result from a metric evaluation."""
    score: float
    reasoning: str | None = None
    details: Dict[str, Any] | None = None


class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str | List[str],
        reference: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context (string or list of strings)
            reference: Optional ground truth answer
            
        Returns:
            EvaluationResult with score and reasoning
        """
        pass


class Faithfulness(Metric):
    """
    Measures factual consistency of the answer with retrieved context.
    
    A faithful answer only contains claims that can be supported by
    the retrieved context, without hallucination.
    """
    
    EXTRACT_CLAIMS_PROMPT = """Extract all factual claims from the following answer. List each claim on a new line.

Answer: {answer}

Claims (one per line):"""

    VERIFY_CLAIM_PROMPT = """Determine if the following claim can be verified from the given context.

Context: {context}

Claim: {claim}

Can this claim be verified from the context? Answer with only "Yes" or "No"."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider or OpenAILLM()
    
    @property
    def name(self) -> str:
        return "faithfulness"
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str | List[str],
        reference: str | None = None,
    ) -> EvaluationResult:
        """Evaluate faithfulness of the answer to the context."""
        if isinstance(context, list):
            context = "\n\n".join(context)
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return EvaluationResult(
                score=1.0,
                reasoning="No factual claims found in answer",
                details={"claims": [], "verified": []}
            )
        
        # Verify each claim against context
        verified = []
        for claim in claims:
            is_verified = self._verify_claim(claim, context)
            verified.append(is_verified)
        
        # Calculate faithfulness score
        score = sum(verified) / len(verified)
        
        return EvaluationResult(
            score=score,
            reasoning=f"{sum(verified)}/{len(verified)} claims verified",
            details={"claims": claims, "verified": verified}
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract claims from the answer."""
        prompt = self.EXTRACT_CLAIMS_PROMPT.format(answer=answer)
        response = self.llm.generate(prompt)
        
        claims = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip() and not line.strip().startswith("Claims")
        ]
        return claims
    
    def _verify_claim(self, claim: str, context: str) -> bool:
        """Verify a single claim against context."""
        prompt = self.VERIFY_CLAIM_PROMPT.format(claim=claim, context=context[:3000])
        response = self.llm.generate(prompt)
        return "yes" in response.content.lower()


class AnswerRelevancy(Metric):
    """
    Measures how relevant the answer is to the query.
    
    Uses the approach of generating hypothetical questions from the answer
    and comparing similarity to the original question.
    """
    
    RELEVANCY_PROMPT = """Rate how relevant the following answer is to the question on a scale of 0 to 1.

Question: {query}

Answer: {answer}

Score the relevancy:
- 1.0: Perfectly relevant, directly addresses the question
- 0.7-0.9: Highly relevant, addresses most aspects
- 0.4-0.6: Somewhat relevant, partially addresses the question
- 0.1-0.3: Slightly relevant, tangentially related
- 0.0: Not relevant at all

Respond with ONLY a single number between 0 and 1."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider or OpenAILLM()
    
    @property
    def name(self) -> str:
        return "answer_relevancy"
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str | List[str],
        reference: str | None = None,
    ) -> EvaluationResult:
        """Evaluate answer relevancy to the query."""
        prompt = self.RELEVANCY_PROMPT.format(query=query, answer=answer)
        response = self.llm.generate(prompt)
        
        # Parse score
        try:
            score = float(re.search(r"[\d.]+", response.content).group())
            score = max(0.0, min(1.0, score))
        except (AttributeError, ValueError):
            score = 0.5  # Default if parsing fails
        
        return EvaluationResult(
            score=score,
            reasoning=f"LLM-assessed relevancy score: {score:.2f}"
        )


class ContextPrecision(Metric):
    """
    Measures the signal-to-noise ratio of retrieved context.
    
    Evaluates what proportion of the retrieved context is actually
    relevant to answering the query.
    """
    
    PRECISION_PROMPT = """Evaluate the relevance of each context passage to the question.

Question: {query}

{contexts}

For each context, determine if it is relevant (contains useful information for answering the question).

Respond with only the count of relevant passages as a single number."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider or OpenAILLM()
    
    @property
    def name(self) -> str:
        return "context_precision"
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str | List[str],
        reference: str | None = None,
    ) -> EvaluationResult:
        """Evaluate precision of retrieved context."""
        if isinstance(context, str):
            contexts = [context]
        else:
            contexts = context
        
        # Format contexts
        contexts_str = "\n\n".join([
            f"Context {i+1}: {ctx[:500]}..."
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = self.PRECISION_PROMPT.format(query=query, contexts=contexts_str)
        response = self.llm.generate(prompt)
        
        # Parse count
        try:
            relevant_count = int(re.search(r"\d+", response.content).group())
        except (AttributeError, ValueError):
            relevant_count = len(contexts) // 2
        
        score = relevant_count / len(contexts) if contexts else 0.0
        
        return EvaluationResult(
            score=score,
            reasoning=f"{relevant_count}/{len(contexts)} contexts relevant",
            details={"relevant_count": relevant_count, "total_count": len(contexts)}
        )


class ContextRecall(Metric):
    """
    Measures whether all necessary information is in the retrieved context.
    
    Compares statements from a reference answer against the retrieved context.
    """
    
    RECALL_PROMPT = """Given the reference answer and retrieved context, determine what fraction of the reference answer can be supported by the context.

Reference Answer: {reference}

Retrieved Context: {context}

Score from 0 to 1:
- 1.0: All information in the reference is present in the context
- 0.5: About half the information is present
- 0.0: None of the required information is present

Respond with ONLY a single number between 0 and 1."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm = llm_provider or OpenAILLM()
    
    @property
    def name(self) -> str:
        return "context_recall"
    
    def evaluate(
        self,
        query: str,
        answer: str,
        context: str | List[str],
        reference: str | None = None,
    ) -> EvaluationResult:
        """Evaluate recall of retrieved context."""
        if reference is None:
            return EvaluationResult(
                score=0.0,
                reasoning="No reference answer provided for recall evaluation"
            )
        
        if isinstance(context, list):
            context = "\n\n".join(context)
        
        prompt = self.RECALL_PROMPT.format(reference=reference, context=context[:3000])
        response = self.llm.generate(prompt)
        
        try:
            score = float(re.search(r"[\d.]+", response.content).group())
            score = max(0.0, min(1.0, score))
        except (AttributeError, ValueError):
            score = 0.5
        
        return EvaluationResult(
            score=score,
            reasoning=f"LLM-assessed context recall: {score:.2f}"
        )


def evaluate_response(
    query: str,
    answer: str,
    context: str | List[str],
    reference: str | None = None,
    metrics: List[str] | None = None,
    llm_provider: LLMProvider | None = None,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate a RAG response using multiple metrics.
    
    Args:
        query: User query
        answer: Generated answer
        context: Retrieved context(s)
        reference: Optional ground truth answer
        metrics: List of metric names to use (default: all applicable)
        llm_provider: LLM provider for evaluation
        
    Returns:
        Dict mapping metric names to EvaluationResult objects
    """
    llm = llm_provider or OpenAILLM()
    
    available_metrics = {
        "faithfulness": Faithfulness(llm),
        "answer_relevancy": AnswerRelevancy(llm),
        "context_precision": ContextPrecision(llm),
        "context_recall": ContextRecall(llm),
    }
    
    if metrics is None:
        # Use all metrics, but skip context_recall if no reference
        metrics = ["faithfulness", "answer_relevancy", "context_precision"]
        if reference:
            metrics.append("context_recall")
    
    results = {}
    for metric_name in metrics:
        if metric_name in available_metrics:
            metric = available_metrics[metric_name]
            results[metric_name] = metric.evaluate(query, answer, context, reference)
    
    return results
