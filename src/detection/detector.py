"""
HallucinationDetector — orchestrates the full detection pipeline:

  document → [dependency parse] → G_doc
  summary  → [dependency parse] → G_sum
  (G_doc, G_sum) → [graph matching] → MatchResult → hallucination label

Usage:
    detector = HallucinationDetector.from_config(cfg)
    results  = detector.detect_batch(samples)
"""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from src.graph.dependency_parser import DependencyParser, ParsedDoc
from src.graph.graph_builder import GraphBuilder
from src.graph.graph_matcher import GraphMatcher, MatchResult

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    End-to-end hallucination detector.

    Args:
        parser:  DependencyParser instance
        builder: GraphBuilder instance
        matcher: GraphMatcher instance
    """

    def __init__(
        self,
        parser:  DependencyParser,
        builder: GraphBuilder,
        matcher: GraphMatcher,
    ):
        self.parser  = parser
        self.builder = builder
        self.matcher = matcher

    # ─── Factory from config ─────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Dict) -> "HallucinationDetector":
        graph_cfg    = cfg.get("graph", {})
        matching_cfg = cfg.get("matching", {})

        parser = DependencyParser(
            model_name=graph_cfg.get("spacy_model", "en_core_web_sm"),
        )

        builder = GraphBuilder(
            include_ner  = graph_cfg.get("include_ner",  True),
            include_svo  = graph_cfg.get("include_svo",  True),
            include_deps = True,
        )

        # Weights must sum to 1 — read and normalise
        sw = matching_cfg.get("svo_weight",     0.40)
        ew = matching_cfg.get("entity_weight",  0.35)
        lw = matching_cfg.get("lexical_weight", 0.25)
        total = sw + ew + lw
        sw, ew, lw = sw / total, ew / total, lw / total

        matcher = GraphMatcher(
            svo_weight      = sw,
            entity_weight   = ew,
            lexical_weight  = lw,
            threshold       = matching_cfg.get("threshold",       0.50),
            use_fuzzy_match = matching_cfg.get("use_fuzzy_match", True),
            fuzzy_threshold = int(matching_cfg.get("fuzzy_threshold", 80)),
        )

        return cls(parser=parser, builder=builder, matcher=matcher)

    # ─── Core detection ──────────────────────────────────────────────────────

    def detect_one(self, sample: Dict) -> Dict:
        """
        Detect hallucination for a single sample.

        Args:
            sample: dict with 'document' and 'summary_gen' keys.

        Returns:
            sample enriched with 'detection' sub-dict and 'predicted_label'.
        """
        document    = sample["document"]
        summary_gen = sample.get("summary_gen", "")

        if not summary_gen.strip():
            logger.warning(f"Empty summary_gen for id={sample.get('id')}; "
                           "skipping detection.")
            sample["detection"] = {}
            sample["predicted_label"] = -1
            return sample

        # Parse
        doc_parsed = self.parser.parse(document)
        sum_parsed = self.parser.parse(summary_gen)

        # Build graphs
        doc_graph = self.builder.build(doc_parsed)
        sum_graph = self.builder.build(sum_parsed)

        # Match
        result: MatchResult = self.matcher.match(
            doc_parsed=doc_parsed,
            sum_parsed=sum_parsed,
            doc_graph=doc_graph,
            sum_graph=sum_graph,
        )

        # Attach to sample
        sample["detection"]       = result.to_dict()
        sample["predicted_label"] = int(result.is_hallucinated)

        # Optionally store graphs for visualisation
        sample["_doc_graph"] = doc_graph
        sample["_sum_graph"] = sum_graph

        return sample

    def detect_batch(
        self,
        samples: List[Dict],
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run detection on a list of samples.

        Returns:
            Same list, enriched with detection results.
        """
        from tqdm import tqdm
        iterator = tqdm(samples, desc="Detecting hallucinations") \
            if verbose else samples

        for sample in iterator:
            self.detect_one(sample)

        logger.info(f"Detection complete for {len(samples)} samples.")
        return samples

    # ─── Convenience getters ─────────────────────────────────────────────────

    @staticmethod
    def get_predictions(samples: List[Dict]) -> List[int]:
        return [s.get("predicted_label", -1) for s in samples]

    @staticmethod
    def get_scores(samples: List[Dict]) -> List[float]:
        return [s.get("detection", {}).get("hallucination_score", 0.5)
                for s in samples]

    @staticmethod
    def get_labels(samples: List[Dict]) -> List[int]:
        return [s.get("label", -1) for s in samples]
