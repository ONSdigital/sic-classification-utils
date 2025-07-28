# %%
# flake8: noqa: E402, B018
# pylint: disable=invalid-name, protected-access, pointless-statement, wrong-import-position, line-too-long, missing-module-docstring

# %%
from pprint import pprint

from industrial_classification_utils.embed.embedding import get_config

# from sic_soc_llm.data_models.sicDB import sicDB


# %%
config = get_config()

# %%
config

# %%
# sic_index_df = load_sic_index(config["lookups"]["sic_index"])
# sic_df = load_sic_structure(config["lookups"]["sic_structure"])
# sic = load_hierarchy(sic_df, sic_index_df)

# %%
# def download_from_gcs(project_id, bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     storage_client = storage.Client(project=project_id)
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(
#         f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to {destination_file_name}"
#     )
# # Example usage
# project_id = "classifai-sandbox"
# bucket_name = "classifai-app-data"
# source_blob_name = "2025-07-16_sic_knowledge_base.csv"
# destination_file_name = "2025-07-16_sic_knowledge_base.csv"
# download_from_gcs(project_id,bucket_name, source_blob_name, destination_file_name)

# %%
# source_blob_name = "soc2020volume2thecodingindexexcel16102024.xlsx"
# destination_file_name = "soc2020volume2thecodingindexexcel16102024.xlsx"
# download_from_gcs(project_id,bucket_name, source_blob_name, destination_file_name)

# %%
from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.llm.llm import ClassificationLLM

# from sic_soc_llm.soc_embedding import EmbeddingHandler

# %%
embed = EmbeddingHandler()

# %%
# embed.embed_index()

# %%
embed._index_size

# %%
# soc_index = load_sic_index(get_config()["lookups"]["sic_index"])
# embed.create_documents(sic_index)

# %%
# if uni_chat.soc is None:
#     soc_index_df = load_soc_index(config["lookups"]["soc_index"])
#     soc_df_input = load_soc_structure(get_config()["lookups"]["soc_structure"]) #552
#     soc_df = socDB.create_soc_dataframe(soc_df_input)
#     uni_chat.soc = load_hierarchy(soc_df, soc_index_df)

# %%
uni_chat = ClassificationLLM(model_name="gemini-2.0-flash", verbose=True)

# %%
# uni_chat.embed.embed_index()

# %%
# uni_chat.soc["2237"]

# %%
embed.search_index("Brewery")

# %%
# industry_descr="CHILDREN S NURSERY"
# job_title="NURSERY PRACTITIONER"
# job_description="LOOK AFTER PRE SCHOOL CHILDREN"

# industry_descr="kitchen design and sales"
# job_title="project coordinator"
# job_description="admin for kitchen design aftersales customer service"

# %%
industry_descr = "adult social care"
job_title = "community assessment officer"
job_description = "social services"

# %%
short_list = embed.search_index_multi(
    query=[industry_descr, job_title, job_description]
)

# %%
short_list

# %%
sa_soc_rag = uni_chat.sa_rag_sic_code(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    candidates_limit=10,
    short_list=short_list,
)

# %%
candidate_list = uni_chat._prompt_candidate_list(sa_soc_rag[1], candidates_limit=10)

# %%
candidate_list

# %%
sic_response_unambiguous = uni_chat.unambiguous_sic_code(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    sic_candidates=candidate_list,
)

# %%
pprint(sic_response_unambiguous[0].model_dump(), indent=2, width=80)

# %%
# sic_response_reranker = uni_chat.reranker_sic(
#     industry_descr=industry_descr,
#     job_title=job_title,
#     job_description=job_description,
#     candidates_limit=5,
#     expand_search_terms=True)

# %%
# pprint(sic_response_reranker[0].model_dump(), indent=2, width=80)

# %%
sic_followup = uni_chat.formulate_open_question(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    llm_output=sa_soc_rag[0].sic_candidates,
)

# %%
pprint(sic_followup[0].model_dump(), indent=2, width=80)

# %%
sic_followup = uni_chat.formulate_open_question(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    llm_output=sic_response_unambiguous[0].alt_candidates,
)

# %%
pprint(sic_followup[0].model_dump(), indent=2, width=80)

# %%
filtered_list = [elem.class_code for elem in sic_response_unambiguous[0].alt_candidates]

# %%
filtered_list

# %%
filtered_candidates = uni_chat._prompt_candidate_list_filtered(
    sa_soc_rag[1], candidates_limit=10, filtered_list=filtered_list, activities_limit=5
)

# %%
filtered_candidates

# %%
sic_closed_followup = uni_chat.formulate_closed_question(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    llm_output=filtered_candidates,
)

# %%
pprint(sic_closed_followup[0].model_dump(), indent=2, width=80)

# %%
pprint(filtered_candidates, indent=2, width=80)
