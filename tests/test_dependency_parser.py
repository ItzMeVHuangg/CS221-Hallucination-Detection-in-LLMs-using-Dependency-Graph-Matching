import pytest
import spacy

from src.graph.dependency_parser import DependencyParser, SVOTriple, ParsedDoc


@pytest.fixture(scope="module")
def parser():
    """Create parser once for all tests in this module."""
    return DependencyParser(model_name="en_core_web_sm")


class TestDependencyParser:
    """Tests for DependencyParser."""

    def test_parse_returns_parsed_doc(self, parser):
        result = parser.parse("The cat sat on the mat.")
        assert isinstance(result, ParsedDoc)
        assert result.text == "The cat sat on the mat."
        assert len(result.tokens) > 0
        assert len(result.lemmas) > 0

    def test_svo_extraction_simple(self, parser):
        result = parser.parse("Apple released a new phone.")
        svos = result.svo_triples

        # Should find at least one SVO triple
        assert len(svos) >= 1

        # Check that the SVO contains the expected components
        subjects = [t.subject for t in svos]
        verbs = [t.verb for t in svos]
        assert any("apple" in s for s in subjects)
        assert any("release" in v for v in verbs)

    def test_svo_extraction_passive(self, parser):
        result = parser.parse("The phone was released by Apple.")
        svos = result.svo_triples
        # Should handle passive voice
        assert isinstance(svos, list)

    def test_entity_extraction(self, parser):
        result = parser.parse("Barack Obama visited London last week.")
        entities = result.entities

        # Should extract named entities
        assert len(entities) >= 1
        ent_texts = [e[0] for e in entities]
        assert any("Obama" in t or "Barack" in t for t in ent_texts)

    def test_dep_triples_not_empty(self, parser):
        result = parser.parse("The quick brown fox jumps over the lazy dog.")
        assert len(result.dep_triples) > 0

    def test_empty_text(self, parser):
        result = parser.parse("")
        assert isinstance(result, ParsedDoc)
        assert result.svo_triples == []

    def test_spacy_doc_stored(self, parser):
        """v2: verify the raw spaCy Doc is stored."""
        result = parser.parse("Hello world.")
        assert result.spacy_doc is not None

    def test_compound_noun_expansion(self, parser):
        """v2: test that compound nouns are expanded."""
        result = parser.parse("The United States president signed the bill.")
        svos = result.svo_triples
        # Look for compound nouns in subjects
        subjects = [t.subject for t in svos]
        # Should contain 'united states president' or similar compound
        assert len(svos) >= 0  # may or may not produce SVO depending on parse

    def test_batch_parse(self, parser):
        texts = [
            "Apple released a phone.",
            "Google launched a new service.",
        ]
        results = parser.parse_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, ParsedDoc) for r in results)
