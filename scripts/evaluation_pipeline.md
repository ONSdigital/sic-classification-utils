# Scripts for Running Evaluation Pipeline

This module allows user to run evaluation pipeline using Survey Assist (**SA**), which goal is Standard Industry Code (**SIC**) assignment based on responses provided.

The scripts are designed to perform different steps allowing SIC classification. The classification steps include initial classification, using only respondents answers, as well as follow up answer, if first attempt is ambiguous.

The input data is expected to be in a csv format for Stage 1 of the pipeline, and parquet for all other stages.

The .csv file is epected to contain columns: sic2007_employee, soc2020_job_title, soc2020_job_description, sic2007_self_employed.

Following files are expected to contain the same columns, along with columns created in steps before.

The pipeline uses LLM (gemini-2.5-flash).

## Pipeline stages

The pipeline is intended to be used in specific order:
Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6 (use the script for stage 1 with `-s` flag) -> Stage 7 (use the script for stage 2 with `-s` flag).

|Stage|Pipeline process|Required|Columns added|
|--|--|--|--|
|1|Create `merged_industry_description`. Perform Semantic Search|File in .csv format. `sic2007_employee`, `soc2020_job_title`, `soc2020_job_description`, `sic2007_self_employed`|`merged_industry_description`, `semantic_search_results`|
|2|Initial classification and ambiguity assesment|File in .parquet format. `soc2020_job_title`, `soc2020_job_description`, `merged_industry_description`, `semantic_search_results`|`unambiguously_codable`, `initial_code`, `alt_sic_candidates`|
|3|Generate follow up question when `unambiguously_codable` is `False`|File in .parquet format. `unambiguously_codable`, `merged_industry_description`, `soc2020_job_title`, `soc2020_job_description`, `alt_sic_candidates`|`followup_question`|
|4|Generate follow up answer|File in .parquet format. `unambiguously_codable`, `merged_industry_description`, `soc2020_job_title`, `soc2020_job_description`, `followup_question`|`followup_answer`|
|5|Modify `merged_industry_description`|File in .parquet format. `merged_industry_desc`, `followup_question`, `followup_answer`|None (eddit `merged_industry_description`)|
|6|Second semantic search, using modified industry label|File in .csv format. `soc2020_job_title`, `soc2020_job_description`, `merged_industry_description`|`second_semantich_search`|
|7|Final classification and ambiguity assesment|File in .parquet format. `soc2020_job_title`, `soc2020_job_description`, `merged_industry_description`, `second_semantic_search_results`|`unambiguously_codable_final`, `final_code`, `alt_sic_candidates_final`|


## Usage
### Prerequsites
- Python 3.12
- Poetry (This project uses Poetry for dependency management)
- Start up the vector store form `sic-classification-vector-store` repo. Run:
<br>
```bash
make run-vector-store
```
- Authenticate/reauthenticate gcloud:
```bash
gcloud auth application-default login
```
- Create `metadata.json` file (template available in `scripts/stage_1_add_semantic_search.py` within the module-level docstring)

### Running the pipeline
To run the whole pipeline (two prompts approach), use `run_full_pipeline.sh`, available in `sic_classification_utils/scripts`:

```bash
./run_full_pipeline.sh 2 output/file/path /path/to/tlfs_data.csv /path/to/tlfs_data_metadata.json 20
```
### Running script stage 1 (Stages 1 and 6):
Running Stage 1 differs from running other stages.
1. Start vector store
2. Run:
```bash
poetry run python path/to/stage_1_add_semantic_search.py [-n output/file/name] -b <batch size> path/to/input.csv path/to/initial_metadata.json path/to/output_folder [-s]
```
Where:
- `path/to/stage_1_add_semantic_search.py`: relative path to the script.
- `-n output/file/name` (optional): Optional output file name. Default: `STG1`.
- `-b <batch size>`: The size of processing batch.
- `path/to/input.csv`: Relative path to the input file in .csv format. Requires columns as specified above.
- `path/to/initial_metadata.json`: The path to the persisted metadata JSON file.
- `path/to/output_folder`: The path to the specified output folder.
- `-s`(optional): Flag indicating first or second semantic search. If flag is present, returns `secons_semantic_search_results`.

### Running scripts stages 2-5 (Stages 2-5 and 7):
1. (re-)authenicate with gcloud `gcloud auth application-default login`
2. Run:
```bash
poetry run python path/to/stage/to/run.py [-n <output/file/name>] -b <batch size> path/to/input/file.parquet path/to/metadata.json path/to/output/folder [-s]
```
Where:
- `path/to/stage/to/run.py`: relative path to the script.
- `-n output/file/name` (optional): Optional output file name. Default: `STG[#]`, where # is the stage number.
- `-b <batch size>`: The size of processing batch. For stages 2 and 3 the maximum batch size is 10.
- `path/to/input.parquet`: Relative path to the input file in .parquet format. Requires columns as specified above.
- `path/to/initial_metadata.json`: The path to the persisted metadata JSON file from the previous stage.
- `path/to/output_folder`: The path to the specified output folder.
- `-s` (optional) **only in stage 2**: Flag indicating initial or final classification. If flag is present, returns results for final classification.
