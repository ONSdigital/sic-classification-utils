"""Example usage of SAYT for survey industry/organisation descriptions."""

# pylint: disable=protected-access,R0801

# %%
from industrial_classification_utils.sayt.sayt import SAYTSuggester

# %%
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
for q in ["car", "cars", "waxi", "grom", "wash", "duplicate"]:
    print("searching for:", q)
    print("prefix", "->", suggester._get_prefix_suggestions(q))
    print("ngram", "->", suggester._get_ngram_suggestions(q))
    print("semantic", "->", suggester._get_semantic_suggestions(q))
    print("combined", "->", suggester.suggest(q))
    print()


# %%
suggester._clean_corpus(small_corpus)
# %%
