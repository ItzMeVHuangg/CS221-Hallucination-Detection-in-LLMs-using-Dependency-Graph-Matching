"""
Dependency Parser — wraps spaCy to extract:
  1. Dependency triples   (head_lemma, dep_label, child_lemma)
  2. SVO triples          (subject_lemma, verb_lemma, object_lemma)
  3. Named entities       [(text, label), ...]

All parsing is done locally via spaCy.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class DepTriple:
    """A single dependency triple: head --[relation]--> child."""
    head:     str       # head token lemma
    relation: str       # Universal Dependency label (nsubj, dobj, …)
    child:    str       # dependent token lemma
    head_pos: str = ""  # POS of head
    child_pos: str = "" # POS of child

    def __hash__(self):
        return hash((self.head, self.relation, self.child))

    def __eq__(self, other):
        return (self.head == other.head and
                self.relation == other.relation and
                self.child == other.child)


@dataclass
class SVOTriple:
    """Subject–Verb–Object triple."""
    subject: str
    verb:    str
    obj:     str

    def __hash__(self):
        return hash((self.subject, self.verb, self.obj))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.verb == other.verb and
                self.obj == other.obj)


@dataclass
class ParsedDoc:
    """Everything extracted from a single text."""
    text:      str
    tokens:    List[str]              = field(default_factory=list)
    lemmas:    List[str]              = field(default_factory=list)
    entities:  List[Tuple[str, str]]  = field(default_factory=list)  # (text, label)
    dep_triples: List[DepTriple]      = field(default_factory=list)
    svo_triples: List[SVOTriple]      = field(default_factory=list)


# ─── Parser ──────────────────────────────────────────────────────────────────

class DependencyParser:
    """
    Thin wrapper around spaCy for structured extraction.

    Args:
        model_name: spaCy model ('en_core_web_sm', 'en_core_web_md', 'en_core_web_trf')
        disable:    pipeline components to disable (speeds up processing)
    """

    # Dependency labels that indicate the object of a verb
    _OBJECT_DEPS = {"dobj", "obj", "pobj", "iobj", "attr", "oprd"}
    # Dependency labels that indicate the subject of a verb
    _SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
    # POS tags to skip for content filtering
    _SKIP_POS = {"PUNCT", "SPACE", "DET", "PART", "SCONJ", "CCONJ", "NUM", "SYM", "X"}

    def __init__(self, model_name: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {model_name}")
        try:
            self.nlp: Language = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model '{model_name}' not found. Downloading …")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        # Increase max length for long documents
        self.nlp.max_length = 2_000_000
        logger.info("spaCy model loaded.")

    # ─────────────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> ParsedDoc:
        """Full parse of a single text string."""
        doc = self.nlp(text)
        return ParsedDoc(
            text=text,
            tokens=[t.text for t in doc if not t.is_space],
            lemmas=[t.lemma_.lower() for t in doc if not t.is_space],
            entities=[(ent.text, ent.label_) for ent in doc.ents],
            dep_triples=self._extract_dep_triples(doc),
            svo_triples=self._extract_svo_triples(doc),
        )

    def parse_batch(self, texts: List[str], batch_size: int = 64) -> List[ParsedDoc]:
        """Batch parse for efficiency."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            results.append(ParsedDoc(
                text=doc.text,
                tokens=[t.text for t in doc if not t.is_space],
                lemmas=[t.lemma_.lower() for t in doc if not t.is_space],
                entities=[(ent.text, ent.label_) for ent in doc.ents],
                dep_triples=self._extract_dep_triples(doc),
                svo_triples=self._extract_svo_triples(doc),
            ))
        return results

    # ─── Private extraction methods ──────────────────────────────────────────

    def _is_content_token(self, token) -> bool:
        return (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.pos_ not in self._SKIP_POS
            and len(token.lemma_.strip()) > 1
        )

    def _extract_dep_triples(self, doc) -> List[DepTriple]:
        """Extract all dependency triples from a spaCy doc."""
        triples = []
        for token in doc:
            if token.dep_ in ("ROOT", "punct", "cc", "det"):
                continue
            head = token.head
            if head == token:
                continue
            triples.append(DepTriple(
                head=head.lemma_.lower(),
                relation=token.dep_,
                child=token.lemma_.lower(),
                head_pos=head.pos_,
                child_pos=token.pos_,
            ))
        return triples

    def _extract_svo_triples(self, doc) -> List[SVOTriple]:
        """
        Extract Subject–Verb–Object triples.

        Strategy:
          For each verb token, find its subject(s) and object(s)
          in the immediate syntactic neighbourhood.
        """
        triples = []

        for token in doc:
            # Look for verbs (VERB or AUX with object dependency)
            if token.pos_ not in ("VERB", "AUX"):
                continue

            verb_lemma = token.lemma_.lower()

            # Collect subjects
            subjects = []
            for child in token.children:
                if child.dep_ in self._SUBJECT_DEPS:
                    subjects.append(self._get_span_lemma(child))
                # Check conjuncts
                for conj in child.conjuncts:
                    if conj.dep_ in self._SUBJECT_DEPS:
                        subjects.append(self._get_span_lemma(conj))

            # Collect objects
            objects = []
            for child in token.children:
                if child.dep_ in self._OBJECT_DEPS:
                    objects.append(self._get_span_lemma(child))
                # Prepositional objects
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            objects.append(self._get_span_lemma(grandchild))

            # Also check xcomp / advcl for chained verbs
            for child in token.children:
                if child.dep_ in ("xcomp", "advcl") and child.pos_ == "VERB":
                    for gc in child.children:
                        if gc.dep_ in self._OBJECT_DEPS:
                            objects.append(self._get_span_lemma(gc))

            # Build cross-product of subj × obj
            for subj in subjects:
                for obj in objects:
                    if subj and obj and subj != verb_lemma and obj != verb_lemma:
                        triples.append(SVOTriple(
                            subject=subj,
                            verb=verb_lemma,
                            obj=obj,
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
    def _get_span_lemma(token) -> str:
        """Return the lemma of a token (or compound noun head)."""
        return token.lemma_.lower().strip()
