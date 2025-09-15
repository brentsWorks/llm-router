"""
Hybrid classification system for LLM Router.

This module implements a hybrid classifier that combines RAG (Retrieval-Augmented Generation)
classification with rule-based classification using confidence thresholds for intelligent fallback.
"""

import logging
from typing import Optional
from enum import Enum

from .models import PromptClassification
from .rag_classification import RAGClassifier, RAGClassificationError
from .classification import KeywordClassifier
from .llm_fallback import LLMFallbackClassifier, LLMFallbackError

logger = logging.getLogger(__name__)


class ClassificationMethod(Enum):
    """Classification methods used by hybrid classifier."""
    RAG = "rag"
    RULE = "rule"
    LLM_FALLBACK = "llm_fallback"
    FALLBACK = "fallback"
    ERROR = "error"


class HybridClassificationError(Exception):
    """Exception raised for hybrid classification errors."""
    pass


class HybridClassifier:
    """
    Hybrid classifier combining RAG and rule-based classification.
    
    This classifier uses confidence thresholds to intelligently choose between:
    1. RAG classification (high confidence semantic matches)
    2. Rule-based classification (fallback for low confidence or missing examples)
    3. Best-effort fallback (when both have low confidence)
    """

    def __init__(
        self,
        rag_classifier: RAGClassifier,
        llm_fallback: Optional[LLMFallbackClassifier] = None,
        rag_threshold: float = 0.7
    ):
        """Initialize hybrid classifier.
        
        Args:
            rag_classifier: RAG classifier instance
            llm_fallback: Optional LLM fallback classifier for edge cases
            rag_threshold: Minimum confidence for using RAG classification
            
        Raises:
            HybridClassificationError: If configuration is invalid
        """
        # Validate thresholds
        if not isinstance(rag_threshold, (int, float)) or not (0.0 <= rag_threshold <= 1.0):
            raise HybridClassificationError(
                f"RAG threshold must be between 0.0 and 1.0, got: {rag_threshold}"
            )
        
        self.rag_classifier = rag_classifier
        self.llm_fallback = llm_fallback
        self.rag_threshold = rag_threshold
        
        # Track last classification method used
        self._last_method = None
        
        logger.info(
            f"Hybrid classifier initialized with RAG threshold: {rag_threshold}"
        )

    def classify(self, prompt: str) -> PromptClassification:
        """Classify a prompt using hybrid approach.
        
        Decision logic:
        1. Try RAG classification first
        2. If RAG confidence >= rag_threshold, use RAG result
        3. Otherwise, try LLM fallback if available
        4. If LLM fallback fails or unavailable, raise error
        
        Args:
            prompt: The input prompt to classify
            
        Returns:
            PromptClassification with category, confidence, and reasoning
            
        Raises:
            HybridClassificationError: If classification fails
        """
        # Validate input
        if prompt is None:
            raise HybridClassificationError("Prompt cannot be None")
        
        if not prompt or not prompt.strip():
            raise HybridClassificationError("Prompt cannot be empty")
        
        prompt = prompt.strip()
        
        # Try RAG classification first
        rag_result = None
        rag_error = None
        
        try:
            logger.debug("Attempting RAG classification")
            rag_result = self.rag_classifier.classify(prompt)
            logger.debug(f"RAG classification: {rag_result.category} (confidence: {rag_result.confidence})")
            
            # If RAG confidence is high enough, use it
            if rag_result.confidence >= self.rag_threshold:
                self._last_method = ClassificationMethod.RAG
                return self._create_hybrid_result(
                    rag_result,
                    method="RAG",
                    reason=f"High confidence RAG classification (≥{self.rag_threshold})"
                )
            else:
                logger.debug(f"RAG confidence too low ({rag_result.confidence} < {self.rag_threshold}), trying LLM fallback")
                
        except Exception as e:
            logger.warning(f"RAG classification failed: {e}")
            rag_result = None
            rag_error = e
        
        # Try LLM fallback if RAG had low confidence or failed
        if self.llm_fallback is not None:
            try:
                logger.debug("Attempting LLM fallback classification")
                llm_result = self.llm_fallback.classify(prompt)
                logger.debug(f"LLM fallback classification: {llm_result.category} (confidence: {llm_result.confidence})")
                
                self._last_method = ClassificationMethod.LLM_FALLBACK
                return self._create_hybrid_result(
                    llm_result,
                    method="LLM fallback",
                    reason=f"LLM fallback after RAG {'failed' if rag_result is None else f'low confidence ({rag_result.confidence})'}"
                )
                
            except Exception as e:
                logger.warning(f"LLM fallback classification failed: {e}")
                # Fall through to error case
        
        # All methods failed
        self._last_method = ClassificationMethod.ERROR
        error_msg = f"All classifiers failed. RAG error: {rag_error if 'rag_error' in locals() else 'N/A'}"
        if self.llm_fallback is not None:
            error_msg += f", LLM fallback unavailable"
        logger.error(error_msg)
        raise HybridClassificationError(error_msg)

    def _create_hybrid_result(
        self,
        base_result: PromptClassification,
        method: str,
        reason: str
    ) -> PromptClassification:
        """Create a hybrid classification result with enhanced reasoning.
        
        Args:
            base_result: Base classification result
            method: Classification method used
            reason: Reason for using this method
            
        Returns:
            Enhanced PromptClassification
        """
        # Enhance reasoning with hybrid decision info
        enhanced_reasoning = f"[Hybrid-{method}] {reason}. {base_result.reasoning}"
        
        return PromptClassification(
            category=base_result.category,
            confidence=base_result.confidence,
            embedding=base_result.embedding,
            reasoning=enhanced_reasoning
        )

    def get_last_classification_method(self) -> Optional[str]:
        """Get the classification method used in the last classify() call.
        
        Returns:
            String representation of the last method used, or None if no classification yet
        """
        if self._last_method is None:
            return None
        return self._last_method.value

    def set_rag_threshold(self, threshold: float) -> None:
        """Set the RAG confidence threshold.
        
        Args:
            threshold: New RAG threshold (0.0-1.0)
            
        Raises:
            HybridClassificationError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise HybridClassificationError(
                f"RAG threshold must be between 0.0 and 1.0, got: {threshold}"
            )
        
        old_threshold = self.rag_threshold
        self.rag_threshold = threshold
        logger.info(f"Updated RAG threshold: {old_threshold} → {threshold}")

    def set_rule_threshold(self, threshold: float) -> None:
        """Set the rule-based confidence threshold.
        
        Args:
            threshold: New rule threshold (0.0-1.0)
            
        Raises:
            HybridClassificationError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise HybridClassificationError(
                f"Rule threshold must be between 0.0 and 1.0, got: {threshold}"
            )
        
        old_threshold = self.rule_threshold
        self.rule_threshold = threshold
        logger.info(f"Updated rule threshold: {old_threshold} → {threshold}")

    def get_thresholds(self) -> tuple[float, float]:
        """Get current confidence thresholds.
        
        Returns:
            Tuple of (rag_threshold, rule_threshold)
        """
        return (self.rag_threshold, self.rule_threshold)

    def get_stats(self) -> dict:
        """Get classifier statistics and configuration.
        
        Returns:
            Dictionary with classifier stats
        """
        return {
            "rag_threshold": self.rag_threshold,
            "rule_threshold": self.rule_threshold,
            "last_method": self.get_last_classification_method(),
            "rag_classifier_threshold": self.rag_classifier.get_confidence_threshold(),
        }


# Factory function for easy instantiation
def create_hybrid_classifier(
    rag_classifier: RAGClassifier,
    rule_classifier: Optional[KeywordClassifier] = None,
    **kwargs
) -> HybridClassifier:
    """Create a hybrid classifier instance.
    
    Args:
        rag_classifier: RAG classifier instance
        rule_classifier: Rule-based classifier (creates new if None)
        **kwargs: Additional configuration options
        
    Returns:
        Configured HybridClassifier instance
    """
    if rule_classifier is None:
        rule_classifier = KeywordClassifier()
    
    return HybridClassifier(
        rag_classifier=rag_classifier,
        rule_classifier=rule_classifier,
        **kwargs
    )
