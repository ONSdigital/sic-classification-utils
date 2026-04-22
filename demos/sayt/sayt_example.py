"""Run small example for industry/organisation descriptions SAYT."""

# pylint: disable=protected-access, R0801

# %%
from survey_assist_utils.logging import get_logger

from industrial_classification_utils.sayt import SAYTSuggester

logger = get_logger(__name__)
# %%
############# toy example to verify SAYT works #############
small_corpus = [
    ("Car wash", "Car Wash"),
    ("Car wash", "CAR WASH (duplicate)"),
    ("Car waxing", "Car Waxing"),
    ("Waxing car", "Car Waxing"),
    ("Carpentry services", "Carpentry services"),
    ("Dog grooming", "Dog grooming"),
    ("Cat grooming", "Cat grooming"),
    ("USed car sales", "Used car sales"),
    ("Car rental", "Car rental"),
    ("Car repair", "Car repair"),
    ("Car servicing", "Car servicing"),
]
suggester = SAYTSuggester(small_corpus)

# %%
for q in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    print("searching for:", q)
    print("prefix", "->", suggester._get_prefix_suggestions(q))
    print("ngram", "->", suggester._get_ngram_suggestions(q))
    print("semantic", "->", suggester._get_semantic_suggestions(q))
    print("combined", "->", suggester.suggest(q))
    print()

# %%
