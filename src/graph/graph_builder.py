"""
Graph Builder — converts ParsedDoc into a directed NetworkX graph.

Node types:
  - WORD   : content word (lemma)
  - ENTITY : named entity (text + label)

Edge types:
  - DEP    : dependency relation  (label = UD dep label)
  - SVO    : subject-verb-object  (label = 'SVO')
  - CO_ENT : two entities in same sentence

Nodes carry attributes: {type, pos, entity_type}
Edges carry attributes: {type, label, weight}
"""

import logging
from typing import Dict, List, Optional

import networkx as nx

from .dependency_parser import ParsedDoc, DepTriple, SVOTriple

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build a dependency graph from a ParsedDoc.

    Args:
        include_ner:   add named entity nodes & co-entity edges
        include_svo:   add explicit SVO triple edges
        include_deps:  add raw dependency edges
    """

    def __init__(
        self,
        include_ner:  bool = True,
        include_svo:  bool = True,
        include_deps: bool = True,
    ):
        self.include_ner  = include_ner
        self.include_svo  = include_svo
        self.include_deps = include_deps

    # ─────────────────────────────────────────────────────────────────────────

    def build(self, parsed: ParsedDoc) -> nx.DiGraph:
        """
        Build and return a directed dependency graph for `parsed`.
        """
        G = nx.DiGraph()
        G.graph["text"] = parsed.text[:200]    # store snippet for debugging

        # 1. Dependency edges
        if self.include_deps:
            for triple in parsed.dep_triples:
                self._add_dep_triple(G, triple)

        # 2. SVO edges
        if self.include_svo:
            for svo in parsed.svo_triples:
                self._add_svo_triple(G, svo)

        # 3. Named entity nodes
        if self.include_ner:
            for ent_text, ent_label in parsed.entities:
                node_id = f"ENT:{ent_text.lower()}"
                G.add_node(node_id, type="ENTITY",
                           text=ent_text, entity_type=ent_label)

            # Co-entity edges (entities that appear in the same sentence)
            ents = [f"ENT:{e[0].lower()}" for e in parsed.entities]
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    if ents[i] != ents[j]:
                        G.add_edge(ents[i], ents[j],
                                   type="CO_ENT", label="co_entity", weight=0.5)
                        G.add_edge(ents[j], ents[i],
                                   type="CO_ENT", label="co_entity", weight=0.5)

        return G

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _add_dep_triple(self, G: nx.DiGraph, t: DepTriple):
        """Add head → child with dependency label."""
        if not t.head or not t.child:
            return
        # Nodes
        if t.head not in G:
            G.add_node(t.head, type="WORD", pos=t.head_pos)
        if t.child not in G:
            G.add_node(t.child, type="WORD", pos=t.child_pos)
        # Edge (allow multiple dep labels between same pair via key)
        G.add_edge(t.head, t.child,
                   type="DEP", label=t.relation, weight=1.0)

    def _add_svo_triple(self, G: nx.DiGraph, t: SVOTriple):
        """Add subject → verb → object with SVO label."""
        if not t.subject or not t.verb or not t.obj:
            return
        for node in (t.subject, t.verb, t.obj):
            if node not in G:
                G.add_node(node, type="WORD", pos="")
        # subj → verb
        G.add_edge(t.subject, t.verb,
                   type="SVO", label="SVO_subj", weight=1.5)
        # verb → obj
        G.add_edge(t.verb, t.obj,
                   type="SVO", label="SVO_obj", weight=1.5)

    # ─────────────────────────────────────────────────────────────────────────

    def build_batch(self, parsed_docs: List[ParsedDoc]) -> List[nx.DiGraph]:
        return [self.build(p) for p in parsed_docs]

    # ─── Graph statistics helper ─────────────────────────────────────────────

    @staticmethod
    def graph_stats(G: nx.DiGraph) -> Dict:
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": (sum(d for _, d in G.degree()) / G.number_of_nodes()
                           if G.number_of_nodes() > 0 else 0.0),
        }
