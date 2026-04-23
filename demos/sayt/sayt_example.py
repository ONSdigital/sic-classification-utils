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
# set max_df high to avoid filtering out n-grams in this tiny corpus
suggester = SAYTSuggester(small_corpus, ngram_max_df=0.8)


# %%
for q in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    print("searching for:", q)
    print("prefix", "->", suggester._prefix_retriever.suggest_with_scores(q, 5))
    print("ngram", "->", suggester._ngram_retriever.suggest_with_scores(q, 5))
    print("semantic", "->", suggester._semantic_retriever.suggest_with_scores(q, 5))
    print("combined", "->", suggester.suggest_with_scores(q, 5))
    print("combined_nice", "->", suggester.suggest(q, 5))
    print()

# %%
