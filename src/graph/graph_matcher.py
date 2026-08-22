import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from rapidfuzz import fuzz

from .dependency_parser import SVOTriple, ParsedDoc
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# ─── Result container ────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    svo_score:            float    # [0, 1]  SVO triple recall
    entity_score:         float    # [0, 1]  entity recall
    lexical_score:        float    # [0, 1]  node Jaccard
    nli_score:            float    # [0, 1]  NLI entailment     [NEW]
    negation_penalty:     float    # [0, 1]  negation mismatch  [NEW]
    composite_score:      float    # [0, 1]  weighted combination (faithfulness)
    hallucination_score:  float    # [0, 1]  = 1 − composite  (higher ⇒ hallucinated)
    is_hallucinated:      bool     # thresholded decision

    # Diagnostics
    n_svo_summary:        int = 0
    n_svo_matched:        int = 0
    n_entities_summary:   int = 0
    n_entities_matched:   int = 0
    matched_triples:      List[Tuple] = None
    unmatched_triples:    List[Tuple] = None

    def to_dict(self) -> Dict:
        return {
            "svo_score":           round(self.svo_score, 4),
            "entity_score":        round(self.entity_score, 4),
            "lexical_score":       round(self.lexical_score, 4),
            "nli_score":           round(self.nli_score, 4),
            "negation_penalty":    round(self.negation_penalty, 4),
            "composite_score":     round(self.composite_score, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "is_hallucinated":     self.is_hallucinated,
            "n_svo_summary":       self.n_svo_summary,
            "n_svo_matched":       self.n_svo_matched,
            "n_entities_summary":  self.n_entities_summary,
            "n_entities_matched":  self.n_entities_matched,
        }


# ─── Graph Matcher ───────────────────────────────────────────────────────────

class GraphMatcher:
    """
    Compare source-document graph vs summary graph to score hallucination.

    v2 upgrade: adds NLI scoring and negation penalty.

    Args:
        svo_weight:         weight of SVO match signal
        entity_weight:      weight of entity match signal
        lexical_weight:     weight of lexical (node) overlap
        nli_weight:         weight of NLI entailment signal       [NEW]
        negation_penalty_w: weight of negation penalty             [NEW]
        threshold:          composite faithfulness score below which we flag hallucination
        use_fuzzy_match:    allow soft string matching
        fuzzy_threshold:    minimum fuzzy ratio [0-100] to count as match
        nli_scorer:         NLIScorer instance (or None to disable)
        negation_detector:  NegationDetector instance (or None to disable)
    """

    def __init__(
        self,
        svo_weight:         float = 0.30,
        entity_weight:      float = 0.25,
        lexical_weight:     float = 0.15,
        nli_weight:         float = 0.30,
        negation_penalty_w: float = 0.20,
        threshold:          float = 0.50,
        use_fuzzy_match:    bool  = True,
        fuzzy_threshold:    int   = 80,
        nli_scorer:         Optional[object] = None,
        negation_detector:  Optional[object] = None,
    ):
        self.svo_weight         = svo_weight
        self.entity_weight      = entity_weight
        self.lexical_weight     = lexical_weight
        self.nli_weight         = nli_weight
        self.negation_penalty_w = negation_penalty_w
        self.threshold          = threshold
        self.use_fuzzy          = use_fuzzy_match
        self.fuzzy_threshold    = fuzzy_threshold
        self.nli_scorer         = nli_scorer
        self.negation_detector  = negation_detector

        # Normalize signal weights to sum to 1
        total_w = svo_weight + entity_weight + lexical_weight + nli_weight
        if total_w > 0:
            self.svo_weight     = svo_weight / total_w
            self.entity_weight  = entity_weight / total_w
            self.lexical_weight = lexical_weight / total_w
            self.nli_weight     = nli_weight / total_w

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def match(
        self,
        doc_parsed:  ParsedDoc,
        sum_parsed:  ParsedDoc,
        doc_graph:   nx.DiGraph,
        sum_graph:   nx.DiGraph,
        doc_spacy_doc=None,
        sum_spacy_doc=None,
    ) -> MatchResult:
        """
        Compare document vs summary and return a MatchResult.

        Args:
            doc_parsed:    ParsedDoc for the source document.
            sum_parsed:    ParsedDoc for the generated summary.
            doc_graph:     NetworkX DiGraph for the document.
            sum_graph:     NetworkX DiGraph for the summary.
            doc_spacy_doc: Raw spaCy Doc for negation detection (optional).
            sum_spacy_doc: Raw spaCy Doc for negation detection (optional).
        """
        # Signal 1: SVO matching
        svo_score, n_svo, n_matched, matched_t, unmatched_t = \
            self._svo_match(doc_parsed, sum_parsed)

        # Signal 2: Entity matching
        entity_score, n_ents, n_ents_matched = \
            self._entity_match(doc_parsed, sum_parsed)

        # Signal 3: Lexical overlap
        lexical_score = \
            self._lexical_overlap(doc_graph, sum_graph)

        # Signal 4: NLI entailment scoring [NEW]
        nli_score = self._nli_score(doc_parsed, sum_parsed)

        # Negation penalty [NEW]
        neg_penalty = self._negation_penalty(doc_spacy_doc, sum_spacy_doc)

        # Composite faithfulness score
        composite = (
            self.svo_weight     * svo_score +
            self.entity_weight  * entity_score +
            self.lexical_weight * lexical_score +
            self.nli_weight     * nli_score
        )

        # Apply negation penalty
        composite = composite * (1.0 - self.negation_penalty_w * neg_penalty)
        composite = max(0.0, min(1.0, composite))

        hallucination_score = 1.0 - composite

        return MatchResult(
            svo_score=svo_score,
            entity_score=entity_score,
            lexical_score=lexical_score,
            nli_score=nli_score,
            negation_penalty=neg_penalty,
            composite_score=composite,
            hallucination_score=hallucination_score,
            is_hallucinated=(composite < self.threshold),
            n_svo_summary=n_svo,
            n_svo_matched=n_matched,
            n_entities_summary=n_ents,
            n_entities_matched=n_ents_matched,
            matched_triples=matched_t,
            unmatched_triples=unmatched_t,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 1 — SVO triple matching
    # ─────────────────────────────────────────────────────────────────────────

    def _svo_match(
        self,
        doc_parsed: ParsedDoc,
        sum_parsed: ParsedDoc,
    ) -> Tuple[float, int, int, List, List]:
        """
        For each SVO triple in the summary, check if it is supported
        by the document's SVO triples.

        Returns: (score, n_summary_svos, n_matched, matched_list, unmatched_list)
        """
        sum_svos = sum_parsed.svo_triples
        doc_svos = doc_parsed.svo_triples

        if not sum_svos:
            return 0.5, 0, 0, [], []

        doc_set = set(doc_svos)

        matched   = []
        unmatched = []

        for s_triple in sum_svos:
            if self._triple_in_set(s_triple, doc_set):
                matched.append((s_triple.subject, s_triple.verb, s_triple.obj))
            else:
                unmatched.append((s_triple.subject, s_triple.verb, s_triple.obj))

        score = len(matched) / len(sum_svos)
        return score, len(sum_svos), len(matched), matched, unmatched

    def _triple_in_set(self, query: SVOTriple, doc_set: Set[SVOTriple]) -> bool:
        """Check if query SVO triple is supported (exact or fuzzy)."""
        if query in doc_set:
            return True

        if not self.use_fuzzy:
            return False

        for doc_t in doc_set:
            if (self._fuzzy_match(query.subject, doc_t.subject) and
                    self._fuzzy_match(query.verb, doc_t.verb) and
                    self._fuzzy_match(query.obj, doc_t.obj)):
                return True

            # Partial match: at least two of three components agree
            matches = [
                self._fuzzy_match(query.subject, doc_t.subject),
                self._fuzzy_match(query.verb, doc_t.verb),
                self._fuzzy_match(query.obj, doc_t.obj),
            ]
            if sum(matches) >= 2:
                return True

        return False

    def _fuzzy_match(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        if a in b or b in a:
            return True
        ratio = fuzz.ratio(a, b)
        return ratio >= self.fuzzy_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 2 — Named entity matching
    # ─────────────────────────────────────────────────────────────────────────

    def _entity_match(
        self,
        doc_parsed: ParsedDoc,
        sum_parsed: ParsedDoc,
    ) -> Tuple[float, int, int]:
        """
        Fraction of summary named entities that appear in the document.

        Returns: (score, n_summary_entities, n_matched)
        """
        sum_ents = set(e[0].lower() for e in sum_parsed.entities)
        doc_ents = set(e[0].lower() for e in doc_parsed.entities)
        doc_text_lower = doc_parsed.text.lower()

        if not sum_ents:
            return 0.5, 0, 0

        matched = 0
        for ent in sum_ents:
            if ent in doc_ents:
                matched += 1
            elif ent in doc_text_lower:
                matched += 0.5
            elif self.use_fuzzy:
                for doc_ent in doc_ents:
                    if self._fuzzy_match(ent, doc_ent):
                        matched += 0.8
                        break

        score = min(matched / len(sum_ents), 1.0)
        return score, len(sum_ents), int(matched)

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 3 — Lexical overlap (node-level Jaccard)
    # ─────────────────────────────────────────────────────────────────────────

    def _lexical_overlap(
        self,
        doc_graph: nx.DiGraph,
        sum_graph: nx.DiGraph,
    ) -> float:
        """
        Node-level Jaccard similarity between graph vocabularies.
        Only WORD-type nodes are compared.
        """
        doc_words = {n for n, d in doc_graph.nodes(data=True)
                     if d.get("type", "WORD") == "WORD"}
        sum_words = {n for n, d in sum_graph.nodes(data=True)
                     if d.get("type", "WORD") == "WORD"}

        if not sum_words:
            return 0.5

        intersection = doc_words & sum_words

        if not (doc_words | sum_words):
            return 0.5

        recall    = len(intersection) / len(sum_words)
        precision = len(intersection) / len(doc_words) if doc_words else 0.0

        if precision + recall == 0:
            return 0.0

        # Slightly bias toward recall (faithfulness = summary ⊂ document)
        blended = 0.3 * precision + 0.7 * recall
        return blended

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 4 — NLI entailment scoring [NEW]
    # ─────────────────────────────────────────────────────────────────────────

    def _nli_score(
        self,
        doc_parsed: ParsedDoc,
        sum_parsed: ParsedDoc,
    ) -> float:
        """
        Use NLI Cross-Encoder to score semantic entailment of summary SVOs.

        Returns average entailment probability across all summary SVOs.
        Falls back to 0.5 if NLI scorer is not available.
        """
        if not self.nli_scorer:
            return 0.5

        svo_triples = [
            (t.subject, t.verb, t.obj) for t in sum_parsed.svo_triples
        ]

        if not svo_triples:
            return 0.5

        scores = self.nli_scorer.score_svo_triples(
            document=doc_parsed.text,
            svo_triples=svo_triples,
        )

        return sum(scores) / len(scores) if scores else 0.5

    # ─────────────────────────────────────────────────────────────────────────
    # Negation penalty [NEW]
    # ─────────────────────────────────────────────────────────────────────────

    def _negation_penalty(self, doc_spacy_doc, sum_spacy_doc) -> float:
        """
        Compute negation mismatch penalty.

        Returns 0 if negation detector is not available or no spaCy docs provided.
        """
        if not self.negation_detector or not doc_spacy_doc or not sum_spacy_doc:
            return 0.0

        doc_neg_svos = self.negation_detector.extract_negation_aware_svos(doc_spacy_doc)
        sum_neg_svos = self.negation_detector.extract_negation_aware_svos(sum_spacy_doc)

        return self.negation_detector.compute_negation_penalty(doc_neg_svos, sum_neg_svos)
