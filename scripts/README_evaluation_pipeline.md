# Scripts for Running Evaluation Pipeline

This module allows user to run full classification pipeline using Survey Assist (**SA**), which goal is Standard Industry Code (**SIC**) assignment based on responses provided.

The scripts are designed to perform different steps allowing SIC classification. The classification steps include initial classification, using only respondents answers. If the first attempt is ambiguous, then a follow-up question and synthetic follow-up answer is generated. Based on the follow-up answer, the industry description is extended and second classification attempt is performed. The final output includes SIC code assigned to each respondent, as well as the information about whether the classification was unambiguous or not.

The pipeline input file (csv or parquet) is expected to contain columns: sic2007_employee, soc2020_job_title, soc2020_job_description, sic2007_self_employed. The output file will include additional columns generated during the pipeline, as specified in the table below. Parameters of the pipeline, such as embedding model name, LLM model name, batch size, etc. can be configured using the metadata JSON file, details about the fields are provided in the table further below.

## Pipeline stages

The pipeline is intended to be used in specific order:
Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6 (use the script for stage 1 with `-s` flag) -> Stage 7 (use the script for stage 2 with `-s` flag).

|Stage|Pipeline process|Required columns|Columns added|
|--|--|--|--|
|1|Create `merged_industry_description`. Perform Semantic Search| `sic2007_employee`, `soc2020_job_title`, `soc2020_job_description`, `sic2007_self_employed`|`merged_industry_description`, `semantic_search_results`|
|2|Initial classification and ambiguity assesment| `soc2020_job_title`, `soc2020_job_description`, `merged_industry_desc`, `semantic_search_results`|`unambiguously_codable`, `initial_code`, `alt_sic_candidates`|
|3|Generate follow up question when `unambiguously_codable` is `False`| `unambiguously_codable`, `merged_industry_desc`, `soc2020_job_title`, `soc2020_job_description`, `alt_sic_candidates`|`followup_question`|
|4|Generate follow up answer| `unambiguously_codable`, `merged_industry_desc`, `soc2020_job_title`, `soc2020_job_description`, `followup_question`|`followup_answer`|
|5|Modify `merged_industry_desc`| `merged_industry_desc`, `followup_question`, `followup_answer`| `extended_industry_desc`|
|6|Second semantic search, using modified industry label| `soc2020_job_title`, `soc2020_job_description`, `extended_industry_desc`|`second_semantic_search_results`|
|7|Final classification and ambiguity assesment| `soc2020_job_title`, `soc2020_job_description`, `extended_industry_desc`, `second_semantic_search_results`|`unambiguously_codable_final`, `final_code`, `alt_sic_candidates_final`|


## Usage
### Prerequsites
- Python 3.12
- Poetry (This project uses Poetry for dependency management)
- Authentication to gcloud with gcloud `gcloud auth application-default login`

### Running the pipeline
To run the whole pipeline (two prompts approach), use `run_full_pipeline.sh`, available in `sic_classification_utils/scripts`:

```bash
./run_full_pipeline.sh [-p <1|2>] -i </path/to/tlfs_data.{csv|parquet}> -o </path/to/output/folder> [-m </path/to/tlfs_data_metadata.json>] [-b 20]
```

Where:
- `-p 2` (optional): Flag indicating whether to run one-prompt (1) or two-prompt (2) version of the pipeline. Default is 2.
- `-i </path/to/tlfs_data.{csv|parquet}>`: Relative path to the input file in .csv or .parquet format. Requires columns as specified above.
- `-o </path/to/output/folder>`: The path to the specified output folder.
- `-m </path/to/tlfs_data_metadata.json>` (optional): The path to the persisted metadata JSON file. If not provided, default values for metadata fields will be used.
- `-b 20` (optional): The size of processing batch. Default is 20. For stages 2 and 3 the maximum batch size is 10.



### Running individual scripts (Stages 1 to 7):

2. Run:
```bash
poetry run python path/to/stage/to/run.py -i <path/to/input/file.{csv|parquet}> -o <path/to/output/folder> [-m <path/to/metadata.json>] [-n <output_shortname>] [-b <batch_size>] [-s] [-r]
```
Where:
- `path/to/stage/to/run.py`: relative path to the script.
- `-i <path/to/input/file.{csv|parquet}>`: Relative path to the input file in .csv or .parquet format. Requires columns as specified above.
- `-m <path/to/metadata.json>` (optional): The path to the persisted metadata JSON file from the previous stage.
- `-o <path/to/output/folder>`: The path to the specified output folder.
- `-n <output_shortname>` (optional): Optional output file name. Default: `STG[#]`, where # is the stage number.
- `-b <batch_size>` (optional): The size of processing batch. For stages using LLM (2,3,4 and 7) the maximum batch size is 10.
- `-s` (optional): Flag indicating second run of classification steps. If flag is present, returns results for final classification using modified output column names to avoid conflicts with initial classification outputs.
- `-r` (optional): Flag indicating whether to resume from the last completed batch. If flag is present, the script will attempt to load persisted output and resume from the last completed batch. If not present, the script will start from the beginning, even if there is persisted output available.


## Metadata (pipeline parameters)
The metadata JSON file is used to store configuration values and the state of the pipeline. The input metadata is optional, and if not provided, default values will be used. Checkpointing is implemented by storing the state of the pipeline in the intermediate metadata file after each batch, such as the last completed batch ID for each stage. This allows the pipeline to be resumed from the last completed batch in case of interruptions or failures.
The following fields are stored in the metadata file:

| Field | Default Value | Description |
|--|--|--|
| original_dataset_name | <path/to/input/file.{csv/parquet}> | Path to the original dataset |
| embedding_model_name | all-MiniLM-L6-v2 | Embedding model used for semantic search |
| db_dir | src/industrial_classification_utils/data/vector_store | Directory for vector store database |
| k_matches | 20 | Number of semantic search matches |
| sic_index_file | extended_SIC_index.xlsx | SIC index file |
| sic_structure_file | publisheduksicsummaryofstructureworksheet.xlsx | SIC structure file |
| model_name | gemini-2.5-flash | LLM model used for classification |
| code_digits | 5 | Number of digits in SIC code |
| candidates_limit | 10 | Maximum number of SIC candidates |
| batch_size | 100 | Processing batch size |
| batch_size_async | 10 | Batch size for asynchronous processing |
