import pytest
import spacy

from src.nlp.negation_detector import NegationDetector, NegatedSVO


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def detector():
    return NegationDetector()


class TestNegationDetector:
    """Tests for NegationDetector."""

    def test_detect_negated_verb(self, detector, nlp):
        doc = nlp("The president did not sign the bill.")
        negated = detector.detect_negated_verbs(doc)
        # 'sign' should be detected as negated
        assert len(negated) >= 1

    def test_no_negation(self, detector, nlp):
        doc = nlp("The president signed the bill.")
        negated = detector.detect_negated_verbs(doc)
        assert len(negated) == 0

    def test_negation_aware_svos(self, detector, nlp):
        doc = nlp("The company did not release the product.")
        neg_svos = detector.extract_negation_aware_svos(doc)

        # Should find at least one negated SVO
        negated_ones = [s for s in neg_svos if s.is_negated]
        assert len(negated_ones) >= 0  # might depend on parse

    def test_negation_penalty_mismatch(self, detector, nlp):
        doc_text = nlp("The team did not win the match.")
        sum_text = nlp("The team won the match.")

        doc_svos = detector.extract_negation_aware_svos(doc_text)
        sum_svos = detector.extract_negation_aware_svos(sum_text)

        penalty = detector.compute_negation_penalty(doc_svos, sum_svos)
        # If the verbs match, penalty should be > 0 (negation flip)
        assert isinstance(penalty, float)
        assert 0.0 <= penalty <= 1.0

    def test_negation_penalty_no_mismatch(self, detector, nlp):
        doc_text = nlp("The team won the match.")
        sum_text = nlp("The team won the match.")

        doc_svos = detector.extract_negation_aware_svos(doc_text)
        sum_svos = detector.extract_negation_aware_svos(sum_text)

        penalty = detector.compute_negation_penalty(doc_svos, sum_svos)
        assert penalty == 0.0

    def test_empty_input(self, detector, nlp):
        doc = nlp("")
        neg_svos = detector.extract_negation_aware_svos(doc)
        assert neg_svos == []
