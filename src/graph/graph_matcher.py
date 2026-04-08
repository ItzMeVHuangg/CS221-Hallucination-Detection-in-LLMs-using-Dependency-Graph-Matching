"""
Graph Matcher — computes hallucination score by comparing:
  G_doc  : dependency graph of source document
  G_sum  : dependency graph of generated summary

Three complementary matching signals are combined:

  1. SVO Match Score
       For each SVO triple in G_sum, check if an equivalent or
       similar triple exists in G_doc.
       Score = |matched_SVOs| / |total_SVOs_in_summary|

  2. Entity Match Score
       Fraction of named entities in the summary found in the document.

  3. Lexical Overlap Score
       Lemma-level Jaccard similarity between document and summary nodes.

  Final Score = weighted sum of the three signals
  Hallucination Score = 1 − Final Score    (higher ⇒ more hallucinated)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from rapidfuzz import fuzz

from .dependency_parser import SVOTriple, ParsedDoc
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# ─── Result container ────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    svo_score:        float    # [0, 1]  SVO triple recall
    entity_score:     float    # [0, 1]  entity recall
    lexical_score:    float    # [0, 1]  node Jaccard
    composite_score:  float    # [0, 1]  weighted combination (faithfulness)
    hallucination_score: float # [0, 1]  = 1 − composite  (higher ⇒ hallucinated)
    is_hallucinated:  bool     # thresholded decision

    # Diagnostics
    n_svo_summary:    int = 0
    n_svo_matched:    int = 0
    n_entities_summary: int = 0
    n_entities_matched: int = 0
    matched_triples:  List[Tuple] = None
    unmatched_triples: List[Tuple] = None

    def to_dict(self) -> Dict:
        return {
            "svo_score":           round(self.svo_score, 4),
            "entity_score":        round(self.entity_score, 4),
            "lexical_score":       round(self.lexical_score, 4),
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

    Args:
        svo_weight:      weight of SVO match signal
        entity_weight:   weight of entity match signal
        lexical_weight:  weight of lexical (node) overlap
        threshold:       composite faithfulness score below which we flag hallucination
        use_lemmatization: already handled upstream (ParsedDoc uses lemmas)
        use_fuzzy_match: allow soft string matching (e.g. 'president' ↔ 'presidency')
        fuzzy_threshold: minimum fuzzy ratio [0-100] to count as match
    """

    def __init__(
        self,
        svo_weight:      float = 0.40,
        entity_weight:   float = 0.35,
        lexical_weight:  float = 0.25,
        threshold:       float = 0.50,
        use_fuzzy_match: bool  = True,
        fuzzy_threshold: int   = 80,
    ):
        assert abs(svo_weight + entity_weight + lexical_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.svo_weight      = svo_weight
        self.entity_weight   = entity_weight
        self.lexical_weight  = lexical_weight
        self.threshold       = threshold
        self.use_fuzzy       = use_fuzzy_match
        self.fuzzy_threshold = fuzzy_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def match(
        self,
        doc_parsed:  ParsedDoc,
        sum_parsed:  ParsedDoc,
        doc_graph:   nx.DiGraph,
        sum_graph:   nx.DiGraph,
    ) -> MatchResult:
        """
        Compare document vs summary and return a MatchResult.
        """
        svo_score, n_svo, n_matched, matched_t, unmatched_t = \
            self._svo_match(doc_parsed, sum_parsed)

        entity_score, n_ents, n_ents_matched = \
            self._entity_match(doc_parsed, sum_parsed)

        lexical_score = \
            self._lexical_overlap(doc_graph, sum_graph)

        composite = (
            self.svo_weight    * svo_score +
            self.entity_weight * entity_score +
            self.lexical_weight * lexical_score
        )
        hallucination_score = 1.0 - composite

        return MatchResult(
            svo_score=svo_score,
            entity_score=entity_score,
            lexical_score=lexical_score,
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
            # No triples in summary ⇒ no SVO hallucination signal;
            # give a neutral 0.5 so it doesn't dominate the composite.
            return 0.5, 0, 0, [], []

        # Build a set of doc triples for O(1) lookup
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
        # Exact match
        if query in doc_set:
            return True

        if not self.use_fuzzy:
            return False

        # Fuzzy match — iterate doc_set (acceptable: doc is typically short)
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
        # Exact
        if a == b:
            return True
        # One is substring of other
        if a in b or b in a:
            return True
        # Fuzzy ratio
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
        # Also check raw document text for entity mentions
        doc_text_lower = doc_parsed.text.lower()

        if not sum_ents:
            return 0.5, 0, 0   # neutral

        matched = 0
        for ent in sum_ents:
            if ent in doc_ents:
                matched += 1
            elif ent in doc_text_lower:
                # entity text appears literally in document
                matched += 0.5
            elif self.use_fuzzy:
                # fuzzy search against doc entities
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
            return 0.5   # neutral

        intersection = doc_words & sum_words
        union        = doc_words | sum_words

        if not union:
            return 0.5

        # Weighted: recall-biased (how much of summary is in doc)
        recall    = len(intersection) / len(sum_words)
        precision = len(intersection) / len(doc_words) if doc_words else 0.0
        # F1-like blend, recall-weighted
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        # Slightly bias toward recall (faithfulness = summary ⊂ document)
        blended = 0.3 * precision + 0.7 * recall
        return blended
