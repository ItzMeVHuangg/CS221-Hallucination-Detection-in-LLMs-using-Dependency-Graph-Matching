"""
NLI (Natural Language Inference) Scorer — uses a Cross-Encoder model
to determine if a summary claim is entailed by the source document.

This replaces pure string matching (rapidfuzz) with semantic understanding.

The model classifies (premise, hypothesis) pairs into:
  - entailment     (the document supports the claim)
  - neutral        (the document neither supports nor contradicts)
  - contradiction  (the document contradicts the claim)

We use the entailment probability as a soft faithfulness score.

Default model: cross-encoder/nli-deberta-v3-small
  - Fast enough for CPU inference
  - Strong NLI accuracy (~90% on MNLI)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Guard import
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class NLIScorer:
    """
    Cross-Encoder NLI scorer for semantic faithfulness evaluation.

    Given a (document, claim) pair, returns the probability that
    the document entails the claim.

    Args:
        model_name: HuggingFace model ID for NLI.
        device:     Torch device string.
        enable:     Whether NLI scoring is active.
        max_length: Maximum token length for the Cross-Encoder input.
    """

    # Label mapping for cross-encoder/nli-deberta-v3-small
    # Index 0 = contradiction, 1 = neutral, 2 = entailment
    _ENTAILMENT_IDX = 2
    _CONTRADICTION_IDX = 0

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        enable: bool = True,
        max_length: int = 512,
    ):
        self.enable = enable and _TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.max_length = max_length

        if self.enable:
            logger.info(f"Loading NLI model: {model_name} …")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device(device)
            self.model.to(self.device)
            self.model.eval()

            # Detect label order from model config
            if hasattr(self.model.config, "label2id"):
                label2id = self.model.config.label2id
                # Handle different label naming conventions
                for key, idx in label2id.items():
                    if "entail" in key.lower():
                        self._ENTAILMENT_IDX = idx
                    elif "contradict" in key.lower():
                        self._CONTRADICTION_IDX = idx

            logger.info(f"NLI model loaded. Entailment idx={self._ENTAILMENT_IDX}")
        else:
            self.tokenizer = None
            self.model = None
            if enable and not _TRANSFORMERS_AVAILABLE:
                logger.warning("NLI scoring requested but transformers unavailable.")

    @torch.no_grad()
    def score_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Compute the entailment probability P(premise ⊨ hypothesis).

        Args:
            premise:    The source document text (or relevant sentences).
            hypothesis: The claim to verify (e.g., an SVO triple as text).

        Returns:
            Float in [0, 1] — probability that premise entails hypothesis.
            Returns 0.5 (neutral) if NLI is disabled.
        """
        if not self.enable:
            return 0.5

        try:
            inputs = self.tokenizer(
                premise,
                hypothesis,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            entailment_prob = probs[self._ENTAILMENT_IDX].item()
            return entailment_prob
        except Exception as e:
            logger.warning(f"NLI scoring failed: {e}")
            return 0.5

    @torch.no_grad()
    def score_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> List[float]:
        """
        Batch entailment scoring.

        Args:
            premises:    List of source texts.
            hypotheses:  List of claims to verify.

        Returns:
            List of entailment probabilities.
        """
        if not self.enable:
            return [0.5] * len(premises)

        try:
            inputs = self.tokenizer(
                premises,
                hypotheses,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entailment_probs = probs[:, self._ENTAILMENT_IDX].cpu().tolist()
            return entailment_probs
        except Exception as e:
            logger.warning(f"Batch NLI scoring failed: {e}")
            return [0.5] * len(premises)

    def score_svo_triples(
        self,
        document: str,
        svo_triples: List[Tuple[str, str, str]],
    ) -> List[float]:
        """
        Score a list of SVO triples against a document using NLI.

        Each SVO triple is converted to a natural-language hypothesis,
        then scored for entailment against the document.

        Args:
            document:    Source document text.
            svo_triples: List of (subject, verb, object) tuples.

        Returns:
            List of entailment scores, one per triple.
        """
        if not self.enable or not svo_triples:
            return [0.5] * len(svo_triples)

        # Convert SVO triples to natural language hypotheses
        hypotheses = []
        for subj, verb, obj in svo_triples:
            hypothesis = f"{subj} {verb} {obj}"
            hypotheses.append(hypothesis)

        # Truncate document to fit within model limits
        # Use first ~400 tokens worth of text (rough estimate)
        doc_truncated = document[:2000]

        premises = [doc_truncated] * len(hypotheses)
        return self.score_batch(premises, hypotheses)
