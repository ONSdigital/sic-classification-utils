"""Tests for SAYT retrieval and ranking behavior."""

from industrial_classification_utils.sayt.sayt import SAYTSuggester


def test_prefix_full_string_match_ranks_expected_terms(small_corpus):
    """Return relevant prefix matches and exclude unrelated terms."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car")

    assert "Car Wash" in results
    assert "Car Waxing" in results
    assert "Dog grooming" not in results


def test_duplicate_terms_increase_rank_via_counts(small_corpus):
    """Prefer duplicated underlying search terms when scores tie."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car w")
    assert results[0] == "Car Waxing"


def test_duplicate_display_variants_are_returned_shorter_first(small_corpus):
    """Rank duplicate display variants using the configured tie-breakers."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car wa")
    ind1 = results.index("Car Wash")
    ind2 = results.index("CAR WASH (duplicate)")
    assert ind1 < ind2


def test_fuzzy_prefix_can_recover_from_simple_typo(small_corpus):
    """Recover expected results when the query has a small typo."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("carpentey")
    assert "Carpentry services" in results


def test_ngram_recovers_from_typo_when_prefix_does_not_match(small_corpus):
    """Use n-gram retrieval when typoed input misses prefix matching."""
    s = SAYTSuggester(
        small_corpus,
        min_chars=3,
        ngram_enable=True,
        ngram_n=3,
        ngram_max_df=1.0,
        max_suggestions=5,
        semantic_enable=False,
    )
    results = s.suggest("groming")
    assert results[0] == "Dog grooming"
