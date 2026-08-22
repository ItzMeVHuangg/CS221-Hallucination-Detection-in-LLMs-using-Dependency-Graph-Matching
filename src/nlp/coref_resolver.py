"""
Coreference Resolution — resolves pronouns back to their antecedents
before dependency parsing.

This dramatically improves SVO recall because pronouns like "he", "it",
"they" are replaced with the actual noun phrase they refer to.

Uses the `fastcoref` library (LingMess model) for high-quality,
fast coreference resolution.

Example:
    Input:  "Apple released a new phone. It was very popular."
    Output: "Apple released a new phone. Apple's new phone was very popular."
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Guard import — fastcoref is optional
try:
    from fastcoref import FCoref
    _FASTCOREF_AVAILABLE = True
except ImportError:
    _FASTCOREF_AVAILABLE = False
    logger.warning(
        "fastcoref not installed. Coreference resolution will be disabled. "
        "Install with: pip install fastcoref"
    )


class CoreferenceResolver:
    """
    Resolves coreferences in text using fastcoref (LingMess model).

    When fastcoref is not available, acts as a no-op passthrough.

    Args:
        device: Torch device string ('cpu', 'cuda', 'cuda:0', etc.)
        enable: Whether to actually resolve coreferences.
    """

    def __init__(
        self,
        device: str = "cpu",
        enable: bool = True,
    ):
        self.enable = enable and _FASTCOREF_AVAILABLE

        if self.enable:
            logger.info("Loading fastcoref (LingMess) model …")
            self._model = FCoref(device=device)
            logger.info("Coreference model loaded.")
        else:
            self._model = None
            if enable and not _FASTCOREF_AVAILABLE:
                logger.warning("Coreference resolution requested but fastcoref unavailable.")

    def resolve(self, text: str) -> str:
        """
        Replace pronouns with their antecedents in the given text.

        Args:
            text: Raw input text.

        Returns:
            Text with pronouns replaced by their referents.
            If coref is disabled, returns the original text unchanged.
        """
        if not self.enable or not self._model:
            return text

        if not text or not text.strip():
            return text

        try:
            preds = self._model.predict(texts=[text])
            # fastcoref returns a list of Prediction objects
            resolved = preds[0].get_resolved_text()
            return resolved if resolved else text
        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}. Returning original text.")
            return text

    def resolve_batch(self, texts: List[str]) -> List[str]:
        """
        Batch coreference resolution.

        Args:
            texts: List of raw text strings.

        Returns:
            List of texts with pronouns resolved.
        """
        if not self.enable or not self._model:
            return texts

        if not texts:
            return texts

        try:
            preds = self._model.predict(texts=texts)
            results = []
            for pred, original in zip(preds, texts):
                resolved = pred.get_resolved_text()
                results.append(resolved if resolved else original)
            return results
        except Exception as e:
            logger.warning(f"Batch coreference resolution failed: {e}")
            return texts
