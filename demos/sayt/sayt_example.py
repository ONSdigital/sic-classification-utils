"""Run small example for industry/organisation descriptions SAYT."""

# pylint: disable=protected-access, R0801

# %%
from survey_assist_utils.logging import get_logger

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
)
from industrial_classification_utils.sayt.sayt_core import _normalise

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
# set max_df high to avoid filtering out n-grams in this tiny corpus
suggester = SAYTSuggester(
    small_corpus,
    retrievers=[
        PrefixRetrieverSpec(),
        NgramRetrieverSpec(max_df=0.8),
        SemanticRetrieverSpec(),
    ],
)


# %%
for q in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    # We wouldn't normally call the retrievers directly like this, but it's useful to
    # verify they are wired up as expected and to see their individual contributions
    # before we look at the combined suggestions.
    q_norm = _normalise(q)
    print("searching for:", q)
    for configured in suggester._retrievers:
        print(
            configured.name,
            "->",
            configured.retriever.suggest_with_scores(q_norm, 5),
        )
    print("combined", "->", suggester.suggest_with_scores(q, 5))
    print("combined_nice", "->", suggester.suggest(q, 5))
    print()

# %%
