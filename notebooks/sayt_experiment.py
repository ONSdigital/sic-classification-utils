"""Experimenting with different approaches to improve SAYT results."""

# ruff: noqa: PLR2004
# pylint: disable=C0301,C0103,C0115,C0116,W0621,W0105,R0903
# %%
import os

import pandas as pd
from classifai.indexers import VectorStore
from classifai.indexers.dataclasses import VectorStoreSearchInput
from classifai.vectorisers import GcpVectoriser, HuggingFaceVectoriser, VectoriserBase
from dotenv import get_key, load_dotenv
from sklearn.feature_extraction.text import CountVectorizer

load_dotenv("./.env")
project_id = get_key(".env", "PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file. Please set it.")

data_bucket = get_key(".env", "PREPROD_DATA_BUCKET") or "./"

# out_dir = f"{data_bucket}SAYT_semantic_search_test_results/"
out_dir = "./SAYT_semantic_search_results/"

if not out_dir.startswith("gs://"):
    os.makedirs(out_dir, exist_ok=True)

# %%
lookup_df = pd.read_csv(
    f"{data_bucket}Lookup_IT3_Final.csv",
    usecols=["text", "id"],
    dtype={
        "text": pd.StringDtype,
        "id": pd.StringDtype,
    },
)
lookup_df["id"] = lookup_df["id"].apply(lambda x: x if len(x) == 5 else f"0{x}")

# %%
matching_df = pd.read_csv(
    f"{data_bucket}SAYT_matching.csv",
    usecols=[
        "Correct SIC code",
        "Full entry looking for",
        "Search term used first 5 characters",
    ],
    dtype={
        "Correct SIC code": pd.StringDtype,
        "Full entry looking for": pd.StringDtype,
        "Search term used first 5 characters": pd.StringDtype,
    },
    nrows=100,  # Excel formatting causes 10s of thousands of blank input rows after the real 100
)
matching_df["Correct SIC code"] = matching_df["Correct SIC code"].apply(
    lambda x: x if len(x) == 5 else f"0{x}"
)
matching_df["query_id"] = [f"{idx}" for idx in range(len(matching_df))]

# %%
hfv = HuggingFaceVectoriser("sentence-transformers/all-MiniLM-L6-v2")
# hfv = HuggingFaceVectoriser( # Note - this version requires current `main` branch of classifAI, not yet available in existing releases
#   'nomic-ai/nomic-embed-text-v1.5',
#   model_kwargs={'trust_remote_code': True},
#   model_kwargs={'trust_remote_code': True}
# )
gcpv = GcpVectoriser(
    project_id=project_id,  # Note - this version requires access to a GCP project where VertexAI has been enabled
    task_type="SEMANTIC_SIMILARITY",
    location="europe-west2",
    model_name="text-embedding-004",
    vertexai=True,
)


# %%
class MockSAYTVectoriser(VectoriserBase):
    # Note:
    # This isn't a perfect replication of the SAYT
    #   vectorisation process, but it is pretty close.
    # The 'hybrid' scoring approach is also dropped here,
    #   which may have a small impact
    #   (different approach based on number of search results).
    def __init__(self, corpus):
        self.model = CountVectorizer(
            analyzer="char_wb",
            lowercase=True,
            stop_words="english",
            max_df=0.2,
            ngram_range=(3, 3),
        )
        self.model.fit(corpus)
        super().__init__()

    def transform(self, texts: list[str] | str):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.transform(texts).toarray()


# %%
hf_vs = VectorStore(
    file_name=f"{data_bucket}Lookup_IT3_Final.csv",
    data_type="csv",
    vectoriser=hfv,
    batch_size=8,
    output_dir="hf_vs",
    overwrite=True,
)

# Note: you'll need an active GCP project to use this one - if that's not available,
#       remove all GCP related things in this file, and just look at the
#       huggingface vectoriser results instead (it's almost as good).
gcp_vs = VectorStore(
    file_name=f"{data_bucket}Lookup_IT3_Final.csv",
    data_type="csv",
    vectoriser=gcpv,
    batch_size=128,
    output_dir="gcp_vs",
    overwrite=True,
)

# %%

sayt_v = MockSAYTVectoriser(lookup_df["text"])
sayt_vs = VectorStore(
    file_name=f"{data_bucket}Lookup_IT3_Final.csv",
    data_type="csv",
    vectoriser=sayt_v,
    batch_size=8,
    output_dir="sayt_vs",
    overwrite=True,
)
# %%
matching_search_data_full = VectorStoreSearchInput(
    {
        "id": matching_df.index.to_list(),
        "query": matching_df["Full entry looking for"].to_list(),
    }
)

matching_search_data_partial = VectorStoreSearchInput(
    {
        "id": matching_df.index.to_list(),
        "query": matching_df["Search term used first 5 characters"].to_list(),
    }
)


def collate_matches(vector_store, search_input, out_col_name):
    search_results = vector_store.search(search_input, n_results=10)
    search_results["doc_id"] = search_results["doc_id"].apply(
        lambda x: x if len(x) == 5 else f"0{x}"
    )
    search_results_gpby = search_results.groupby("query_id", as_index=False)
    return search_results_gpby.agg(list)[["query_id", "doc_id", "doc_text"]].rename(
        columns={
            "doc_id": out_col_name + "_codes",
            "doc_text": out_col_name + "_descriptions",
        }
    )


hf_matches_full = collate_matches(
    hf_vs, matching_search_data_full, "hf_ordered_matches_FULL"
)
gcp_matches_full = collate_matches(
    gcp_vs, matching_search_data_full, "gcp_ordered_matches_FULL"
)
sayt_matches_full = collate_matches(
    sayt_vs, matching_search_data_full, "sayt_ordered_matches_FULL"
)
hf_matches_partial = collate_matches(
    hf_vs, matching_search_data_partial, "hf_ordered_matches_PARTIAL"
)
gcp_matches_partial = collate_matches(
    gcp_vs, matching_search_data_partial, "gcp_ordered_matches_PARTIAL"
)
sayt_matches_partial = collate_matches(
    sayt_vs, matching_search_data_partial, "sayt_ordered_matches_PARTIAL"
)

# %%
matching_df = pd.merge(
    matching_df,
    hf_matches_full,
    left_on="query_id",
    right_on="query_id",
    suffixes=["", ""],
)
matching_df = pd.merge(
    matching_df,
    gcp_matches_full,
    left_on="query_id",
    right_on="query_id",
    suffixes=["", ""],
)
matching_df = pd.merge(
    matching_df,
    sayt_matches_full,
    left_on="query_id",
    right_on="query_id",
    suffixes=["", ""],
)
matching_df = pd.merge(
    matching_df,
    hf_matches_partial,
    left_on="query_id",
    right_on="query_id",
    suffixes=["", ""],
)
matching_df = pd.merge(
    matching_df,
    gcp_matches_partial,
    left_on="query_id",
    right_on="query_id",
    suffixes=["", ""],
)
matching_df = pd.merge(
    matching_df,
    sayt_matches_partial,
    left_on="query_id",
    right_on="query_id",
    suffixes=["", ""],
)

useful_cols = [
    "Correct SIC code",
    "Full entry looking for",
    "Search term used first 5 characters",
    "hf_ordered_matches_FULL_codes",
    "hf_ordered_matches_FULL_descriptions",
    "hf_ordered_matches_PARTIAL_codes",
    "hf_ordered_matches_PARTIAL_descriptions",
    "gcp_ordered_matches_FULL_codes",
    "gcp_ordered_matches_FULL_descriptions",
    "gcp_ordered_matches_PARTIAL_codes",
    "gcp_ordered_matches_PARTIAL_descriptions",
    "sayt_ordered_matches_FULL_codes",
    "sayt_ordered_matches_FULL_descriptions",
    "sayt_ordered_matches_PARTIAL_codes",
    "sayt_ordered_matches_PARTIAL_descriptions",
]

# %%
print("Saving output file in .csv and .parquet formats")
matching_df[useful_cols].to_csv(
    f"{out_dir}SAYT_semantic_search_results.csv", index=False
)
try:
    matching_df[useful_cols].to_parquet(
        f"{out_dir}SAYT_semantic_search_results.parquet", index=False
    )
except ImportError:
    print("failed to save parquet file of results, skipping...")


# %%
def get_accuracy(row, col, top_k):
    ans = row["Correct SIC code"]
    top_k = row[col][:top_k]
    return ans in top_k


# %%
for solution in ["hf", "gcp", "sayt"]:
    print(
        f"\n\nUsing {solution} vectoriser approach:\n--------------------------------"
    )
    for top_k in [1, 3, 5, 10]:
        print(f"---------------------------\nAccuracy for top-{top_k} matches:")
        for col in [
            f"{solution}_ordered_matches_FULL_codes",
            f"{solution}_ordered_matches_PARTIAL_codes",
        ]:
            print(
                f"""{col} accuracy: {
                    (matching_df.apply(lambda x, col=col, top_k=top_k: get_accuracy(x, col, top_k), axis=1)).sum():.3f
                    }%"""
            )
# %%
"""
'Hybrid' search idea:
---------------------
- create both (replication of) current BLAISE n-gram vectoriser, and a 'semantic' vectoriser <-- both normalised
- search against both for two distinct shortlists
- calculate weighting for n-gram vs semantic scores based on number of characters entered
- create merged shortlist based on initial ones & weighting (TBD - how to handle same codes, different examples)
- considerations: speed/performance? Would need to be quite quick, may impact whether to use GCP vectoriser

Other ideas:
------------
- TF-IDF with or without PCA to replace current n-gram solution?
"""
