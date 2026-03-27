"""Run small tests for industry/organisation descriptions SAYT."""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,C0103,R0801

# %%
import dotenv
import pandas as pd

from industrial_classification_utils.sayt.sayt import SAYTSuggester

# %%
data_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET") or "data/"

# %%
############# toy example to verify SAYT does something #############
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

for q in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    print("searching for:", q)
    print("prefix", "->", suggester._get_prefix_suggestions(q))
    print("ngram", "->", suggester._get_ngram_suggestions(q))
    print("semantic", "->", suggester._get_semantic_suggestions(q))
    print("combined", "->", suggester.suggest(q))
    print()

# %%
################# now try with lookup from SAYT team #############
sayt_df = pd.read_csv(
    f"{data_bucket}evaluation-pipeline/SAYT/Lookup_IT3_Final.csv", dtype=str
)
sayt_df["code"] = sayt_df["SIC07"].apply(lambda x: x if len(x) == 5 else f"0{x}")
sayt_df["display_text"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = list(zip(sayt_df["SIC_lookup"], sayt_df["display_text"]))
sayt_suggester = SAYTSuggester(sayt_corpus, min_chars=3)

# %%
for q in ["car", "cars", "waxi", "auto", "hea", "heal", "health"]:
    print("searching for:", q)
    print("prefix", "->", sayt_suggester._get_prefix_suggestions(q))
    print("ngram", "->", sayt_suggester._get_ngram_suggestions(q))
    print("semantic", "->", sayt_suggester._get_semantic_suggestions(q))
    print("combined", "->", sayt_suggester.suggest(q, 10))
    print()

# %%
############### now try with our long lookup and rephrased titles #############
rephrased_df = pd.read_csv(
    f"{data_bucket}sic_knowledgebase/sic_rephrased_descriptions_2025_02_03.csv",
    dtype=str,
)
sic_index_df = pd.read_excel(
    f"{data_bucket}sic_knowledgebase/extended_SIC_index.xlsx", skiprows=2, dtype=str
)

rephrased_df["code"] = rephrased_df["input_code"].apply(
    lambda x: x if len(x) == 5 else f"0{x}"
)
sic_index_df["code"] = sic_index_df["UK SIC 2007"].apply(
    lambda x: x if len(x) == 5 else f"0{x}"
)
rephrased_df["display_text"] = (
    rephrased_df["reviewed_description"] + ": " + rephrased_df["code"]
)
merged_df = sic_index_df.merge(
    rephrased_df[["code", "display_text"]], on="code", how="outer", indicator=True
)

activities_corpus = list(zip(merged_df["Activity"], merged_df["display_text"]))
activities_suggester = SAYTSuggester(activities_corpus, min_chars=3)
# note the startup embedding here takes much longer

# %%
for q in ["car", "cars", "waxi", "auto", "hea", "heal", "health"]:
    print("searching for:", q)
    print("prefix", "->", activities_suggester._get_prefix_suggestions(q))
    print("ngram", "->", activities_suggester._get_ngram_suggestions(q))
    print("semantic", "->", activities_suggester._get_semantic_suggestions(q))
    print("combined", "->", activities_suggester.suggest_with_scores(q))
    print("combined", "->", activities_suggester.suggest(q, 10))
    print()
# %%
