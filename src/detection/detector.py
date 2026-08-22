
import logging
from typing import Any, Dict, List, Optional

import torch
import networkx as nx

from src.graph.dependency_parser import DependencyParser, ParsedDoc
from src.graph.graph_builder import GraphBuilder
from src.graph.graph_matcher import GraphMatcher, MatchResult

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' device to the best available device string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


class HallucinationDetector:

    def __init__(
        self,
        parser:            DependencyParser,
        builder:           GraphBuilder,
        matcher:           GraphMatcher,
        negation_detector: Optional[object] = None,
    ):
        self.parser            = parser
        self.builder           = builder
        self.matcher           = matcher
        self.negation_detector = negation_detector

    # ─── Factory from config ─────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Dict) -> "HallucinationDetector":
        graph_cfg    = cfg.get("graph", {})
        matching_cfg = cfg.get("matching", {})
        nlp_cfg      = cfg.get("nlp", {})
        device       = _resolve_device(cfg.get("model", {}).get("device", "cpu"))

        # ── Coreference Resolution [NEW] ────────────────────────────────
        coref_resolver = None
        if nlp_cfg.get("enable_coref", False):
            try:
                from src.nlp.coref_resolver import CoreferenceResolver
                coref_resolver = CoreferenceResolver(
                    device=device,
                    enable=True,
                )
                logger.info("✓ Coreference resolution enabled")
            except Exception as e:
                logger.warning(f"Could not load coreference resolver: {e}")

        # ── NLI Scorer [NEW] ────────────────────────────────────────────
        nli_scorer = None
        if nlp_cfg.get("enable_nli", False):
            try:
                from src.nlp.nli_scorer import NLIScorer
                nli_scorer = NLIScorer(
                    model_name=nlp_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-small"),
                    device=device,
                    enable=True,
                    max_length=nlp_cfg.get("nli_max_length", 512),
                )
                logger.info("✓ NLI scoring enabled")
            except Exception as e:
                logger.warning(f"Could not load NLI scorer: {e}")

        # ── Negation Detector [NEW] ─────────────────────────────────────
        negation_detector = None
        if nlp_cfg.get("enable_negation", True):
            try:
                from src.nlp.negation_detector import NegationDetector
                negation_detector = NegationDetector()
                logger.info("✓ Negation detection enabled")
            except Exception as e:
                logger.warning(f"Could not load negation detector: {e}")

        # ── Parser ──────────────────────────────────────────────────────
        parser = DependencyParser(
            model_name=graph_cfg.get("spacy_model", "en_core_web_sm"),
            coref_resolver=coref_resolver,
        )

        builder = GraphBuilder(
            include_ner  = graph_cfg.get("include_ner",  True),
            include_svo  = graph_cfg.get("include_svo",  True),
            include_deps = True,
        )

        # ── Matcher ─────────────────────────────────────────────────────
        matcher = GraphMatcher(
            svo_weight         = matching_cfg.get("svo_weight",         0.30),
            entity_weight      = matching_cfg.get("entity_weight",      0.25),
            lexical_weight     = matching_cfg.get("lexical_weight",     0.15),
            nli_weight         = matching_cfg.get("nli_weight",         0.30),
            negation_penalty_w = matching_cfg.get("negation_penalty_w", 0.20),
            threshold          = matching_cfg.get("threshold",          0.50),
            use_fuzzy_match    = matching_cfg.get("use_fuzzy_match",    True),
            fuzzy_threshold    = int(matching_cfg.get("fuzzy_threshold", 80)),
            nli_scorer         = nli_scorer,
            negation_detector  = negation_detector,
        )

        return cls(
            parser=parser,
            builder=builder,
            matcher=matcher,
            negation_detector=negation_detector,
        )

    # ─── Core detection ──────────────────────────────────────────────────────

    def detect_one(self, sample: Dict) -> Dict:
        document    = sample["document"]
        summary_gen = sample.get("summary_gen", "")

        if not summary_gen.strip():
            logger.warning(f"Empty summary_gen for id={sample.get('id')}; "
                           "skipping detection.")
            sample["detection"] = {}
            sample["predicted_label"] = -1
            return sample

        # Parse (with coref if enabled)
        doc_parsed = self.parser.parse(document)
        sum_parsed = self.parser.parse(summary_gen)

        # Build graphs
        doc_graph = self.builder.build(doc_parsed)
        sum_graph = self.builder.build(sum_parsed)

        # Match (with NLI + negation if enabled)
        result: MatchResult = self.matcher.match(
            doc_parsed=doc_parsed,
            sum_parsed=sum_parsed,
            doc_graph=doc_graph,
            sum_graph=sum_graph,
            doc_spacy_doc=doc_parsed.spacy_doc,
            sum_spacy_doc=sum_parsed.spacy_doc,
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
