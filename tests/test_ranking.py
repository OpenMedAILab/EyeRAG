"""Tests for the ranking module."""

import pytest

from eye_rag.ranking.rank import (
    aggregate_choices,
    convert_points_to_score,
    SCORE_DIMENSIONS,
    SCORING_CRITERIA,
)


class TestAggregateChoices:
    """Tests for the aggregate_choices function."""

    def test_both_agree_a(self):
        """Both passes agree on A."""
        assert aggregate_choices('A', 'A') == 'A'

    def test_both_agree_b(self):
        """Both passes agree on B."""
        assert aggregate_choices('B', 'B') == 'B'

    def test_both_agree_e(self):
        """Both passes agree on E (equivalent)."""
        assert aggregate_choices('E', 'E') == 'E'

    def test_conflicting_ab_returns_e(self):
        """Conflicting A vs B indicates positional bias, returns E."""
        assert aggregate_choices('A', 'B') == 'E'
        assert aggregate_choices('B', 'A') == 'E'

    def test_one_e_one_a_returns_a(self):
        """When one is E and other is A, returns A (non-conservative)."""
        assert aggregate_choices('A', 'E') == 'A'
        assert aggregate_choices('E', 'A') == 'A'

    def test_one_e_one_b_returns_b(self):
        """When one is E and other is B, returns B (non-conservative)."""
        assert aggregate_choices('B', 'E') == 'B'
        assert aggregate_choices('E', 'B') == 'B'

    def test_error_propagates(self):
        """Error in either pass propagates."""
        assert aggregate_choices('Error', 'A') == 'Error'
        assert aggregate_choices('A', 'Error') == 'Error'
        assert aggregate_choices('Error', 'Error') == 'Error'


class TestConvertPointsToScore:
    """Tests for the convert_points_to_score function."""

    def test_returns_raw_points(self):
        """Function returns raw points directly."""
        assert convert_points_to_score(10, 5) == (10, 5)
        assert convert_points_to_score(0, 0) == (0, 0)
        assert convert_points_to_score(15, 15) == (15, 15)


class TestScoreDimensions:
    """Tests for scoring dimensions configuration."""

    def test_dimensions_count(self):
        """Check correct number of dimensions."""
        assert len(SCORE_DIMENSIONS) == 5

    def test_all_dimensions_have_criteria(self):
        """All dimensions have corresponding criteria."""
        for dim in SCORE_DIMENSIONS:
            assert dim in SCORING_CRITERIA
            assert isinstance(SCORING_CRITERIA[dim], str)
            assert len(SCORING_CRITERIA[dim]) > 0

    def test_expected_dimensions(self):
        """Check expected dimension names."""
        expected = [
            "Clinical Accuracy and Safety",
            "Patient-Centered Response",
            "Professional Communication and Clarity",
            "Completeness and Practical Applicability",
            "Patient-Readiness",
        ]
        assert SCORE_DIMENSIONS == expected
