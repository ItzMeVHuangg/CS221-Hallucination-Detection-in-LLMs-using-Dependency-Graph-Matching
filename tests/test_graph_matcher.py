import pytest

from src.graph.dependency_parser import DependencyParser, ParsedDoc, SVOTriple
from src.graph.graph_builder import GraphBuilder
from src.graph.graph_matcher import GraphMatcher, MatchResult


@pytest.fixture(scope="module")
def parser():
    return DependencyParser(model_name="en_core_web_sm")


@pytest.fixture(scope="module")
def builder():
    return GraphBuilder(include_ner=True, include_svo=True)


@pytest.fixture
def matcher():
    """Matcher without NLI (pure graph matching)."""
    return GraphMatcher(
        svo_weight=0.40,
        entity_weight=0.35,
        lexical_weight=0.15,
        nli_weight=0.10,
        negation_penalty_w=0.0,
        threshold=0.50,
    )


class TestGraphMatcher:
    """Tests for GraphMatcher."""

    def test_faithful_summary_high_score(self, parser, builder, matcher):
        """A summary that closely matches the document should score high."""
        doc = "Apple released a new iPhone model in September."
        summary = "Apple released a new iPhone."

        doc_parsed = parser.parse(doc)
        sum_parsed = parser.parse(summary)
        doc_graph = builder.build(doc_parsed)
        sum_graph = builder.build(sum_parsed)

        result = matcher.match(doc_parsed, sum_parsed, doc_graph, sum_graph)

        assert isinstance(result, MatchResult)
        assert result.composite_score >= 0.0
        assert result.hallucination_score >= 0.0
        assert result.hallucination_score <= 1.0

    def test_hallucinated_summary_low_score(self, parser, builder, matcher):
        """A summary with fabricated entities should score lower."""
        doc = "Apple released a new iPhone model in September."
        summary = "Microsoft launched Windows in January in Tokyo."

        doc_parsed = parser.parse(doc)
        sum_parsed = parser.parse(summary)
        doc_graph = builder.build(doc_parsed)
        sum_graph = builder.build(sum_parsed)

        result = matcher.match(doc_parsed, sum_parsed, doc_graph, sum_graph)

        # Hallucination score should be relatively high
        assert result.hallucination_score > 0.0

    def test_empty_summary(self, parser, builder, matcher):
        doc = "Some document text."
        summary = ""

        doc_parsed = parser.parse(doc)
        sum_parsed = parser.parse(summary)
        doc_graph = builder.build(doc_parsed)
        sum_graph = builder.build(sum_parsed)

        result = matcher.match(doc_parsed, sum_parsed, doc_graph, sum_graph)
        assert isinstance(result, MatchResult)

    def test_result_to_dict(self, parser, builder, matcher):
        doc_parsed = parser.parse("Test document.")
        sum_parsed = parser.parse("Test summary.")
        doc_graph = builder.build(doc_parsed)
        sum_graph = builder.build(sum_parsed)

        result = matcher.match(doc_parsed, sum_parsed, doc_graph, sum_graph)
        d = result.to_dict()

        assert "svo_score" in d
        assert "entity_score" in d
        assert "lexical_score" in d
        assert "nli_score" in d
        assert "negation_penalty" in d
        assert "hallucination_score" in d
        assert "is_hallucinated" in d

    def test_scores_bounded(self, parser, builder, matcher):
        """All scores should be in [0, 1]."""
        doc_parsed = parser.parse("The president visited France last Monday.")
        sum_parsed = parser.parse("The president traveled to France.")
        doc_graph = builder.build(doc_parsed)
        sum_graph = builder.build(sum_parsed)

        result = matcher.match(doc_parsed, sum_parsed, doc_graph, sum_graph)

        for score in [result.svo_score, result.entity_score,
                      result.lexical_score, result.nli_score,
                      result.composite_score, result.hallucination_score]:
            assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
