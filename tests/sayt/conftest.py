"""Shared pytest fixtures for SAYT tests."""

import pytest


@pytest.fixture
def small_corpus():
    """Return a small mixed corpus with duplicates for ranking tests."""
    return [
        ("Car wash", "Car Wash"),
        ("Car wash", "CAR WASH (duplicate)"),
        ("Car waxing", "Car Waxing"),
        ("Waxing car", "Car Waxing"),
        ("Carpentry services", "Carpentry services"),
        ("Dog grooming", "Dog grooming"),
    ]
