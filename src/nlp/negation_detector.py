"""
Negation Detector — identifies negated predicates in dependency trees.

Problem:
  "X did NOT do Y" and "X did Y" produce the same SVO triple (X, do, Y),
  causing false matches. This module detects negation so the scorer
  can penalize sign-flipped claims.

Strategy:
  1. For each verb token, check if any child has dep label 'neg'.
  2. If the verb is negated in the summary but NOT in the document
     (or vice versa), it's a contradiction — heavily penalize.

This is a rule-based approach operating on spaCy dependency trees.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NegatedSVO:
    """An SVO triple with negation information."""
    subject: str
    verb: str
    obj: str
    is_negated: bool  # True if the verb is negated

    def __hash__(self):
        return hash((self.subject, self.verb, self.obj, self.is_negated))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.verb == other.verb and
                self.obj == other.obj and
                self.is_negated == other.is_negated)


class NegationDetector:
    """
    Detects negation in dependency-parsed text.

    Works with spaCy Doc objects to check for 'neg' dependency labels
    attached to verbs, and produces negation-aware SVO triples.
    """

    # Negation dependency labels
    _NEG_DEPS = {"neg"}

    # Negation tokens (for additional heuristic matching)
    _NEG_TOKENS = {
        "not", "n't", "never", "no", "neither", "nor", "nobody",
        "nothing", "nowhere", "hardly", "scarcely", "barely",
    }

    # Subject and object dependency labels (mirrors DependencyParser)
    _SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
    _OBJECT_DEPS = {"dobj", "obj", "pobj", "iobj", "attr", "oprd"}

    def detect_negated_verbs(self, spacy_doc) -> Set[int]:
        """
        Find all verb token indices that are negated.

        Args:
            spacy_doc: A spaCy Doc object.

        Returns:
            Set of token indices for negated verbs.
        """
        negated_verb_indices = set()

        for token in spacy_doc:
            if token.dep_ in self._NEG_DEPS:
                # The head of 'neg' is typically the negated verb
                head = token.head
                if head.pos_ in ("VERB", "AUX"):
                    negated_verb_indices.add(head.i)

            # Also check for negation adverbs modifying verbs
            if (token.lemma_.lower() in self._NEG_TOKENS and
                    token.dep_ in ("advmod", "neg") and
                    token.head.pos_ in ("VERB", "AUX")):
                negated_verb_indices.add(token.head.i)

        return negated_verb_indices

    def extract_negation_aware_svos(self, spacy_doc) -> List[NegatedSVO]:
        """
        Extract SVO triples with negation flags from a spaCy Doc.

        Args:
            spacy_doc: A spaCy Doc object.

        Returns:
            List of NegatedSVO objects.
        """
        negated_verbs = self.detect_negated_verbs(spacy_doc)
        triples = []

        for token in spacy_doc:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            verb_lemma = token.lemma_.lower()
            is_neg = token.i in negated_verbs

            # Collect subjects
            subjects = []
            for child in token.children:
                if child.dep_ in self._SUBJECT_DEPS:
                    subjects.append(child.lemma_.lower())

            # Collect objects
            objects = []
            for child in token.children:
                if child.dep_ in self._OBJECT_DEPS:
                    objects.append(child.lemma_.lower())
                if child.dep_ == "prep":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            objects.append(gc.lemma_.lower())

            for subj in subjects:
                for obj in objects:
                    if subj and obj and subj != verb_lemma and obj != verb_lemma:
                        triples.append(NegatedSVO(
                            subject=subj,
                            verb=verb_lemma,
                            obj=obj,
                            is_negated=is_neg,
                        ))

        # Deduplicate
        seen = set()
        unique = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    @staticmethod
    def compute_negation_penalty(
        doc_neg_svos: List[NegatedSVO],
        sum_neg_svos: List[NegatedSVO],
    ) -> float:
        """
        Compute a penalty score for negation mismatches.

        If the summary says "X did Y" but the document says "X did NOT do Y"
        (or vice versa), this is a serious factual error.

        Args:
            doc_neg_svos: Negation-aware SVOs from the document.
            sum_neg_svos: Negation-aware SVOs from the summary.

        Returns:
            Penalty in [0, 1] where 0 = no negation issues,
            1 = all summary SVOs have negation mismatches.
        """
        if not sum_neg_svos:
            return 0.0

        # Build lookup: (subj, verb, obj) -> is_negated for document
        doc_neg_map: Dict[Tuple[str, str, str], bool] = {}
        for svo in doc_neg_svos:
            key = (svo.subject, svo.verb, svo.obj)
            doc_neg_map[key] = svo.is_negated

        mismatches = 0
        checked = 0

        for svo in sum_neg_svos:
            key = (svo.subject, svo.verb, svo.obj)
            if key in doc_neg_map:
                checked += 1
                if svo.is_negated != doc_neg_map[key]:
                    # Negation flip detected!
                    mismatches += 1
                    logger.debug(
                        f"Negation mismatch: '{svo.subject} {svo.verb} {svo.obj}' "
                        f"doc_neg={doc_neg_map[key]}, sum_neg={svo.is_negated}"
                    )

        if checked == 0:
            return 0.0

        return mismatches / checked
