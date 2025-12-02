#!/usr/bin/env bash

echo "USAGE: ./run_full_pipeline.sh <1 for one-prompt, 2 for 2-prompt> <output_folder> <input_csv> <input_metadata_json> <batch_size>"
echo ""
echo "Keep in mind - you need a local vector store running for stage 1, and gcloud authentication for later stages."
echo ""

set -e

if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters" >&2
    exit 2
fi

pipeline_choice=$1
output_folder=$2
input_csv=$3
input_metadata_json=$4
batch_size=$5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RUNNING: STAGE 1"
"$SCRIPT_DIR"/stage_1_add_semantic_search.py -n "STG1" -b "$batch_size" "$input_csv" "$input_metadata_json" "$output_folder"

if [ "$pipeline_choice" -eq "1" ]; then
    echo "RUNNING: STAGE 2 (one-prompt pipeline)";
    "$SCRIPT_DIR"/stage_2_one_prompt_assign_sic_code.py -n "STG2" -b "$batch_size" "$output_folder""/STG1.parquet" "$output_folder""/STG1_metadata.json" "$output_folder"

    echo "RUNNING: STAGE 4 (one-prompt pipeline)";
    "$SCRIPT_DIR"/stage_4_add_synthetic_responses.py -n "STG4" -b "$batch_size" "$output_folder""/STG2.parquet" "$output_folder""/STG2_metadata.json" "$output_folder"

else
    echo "RUNNING: STAGE 2 (initial classification)";
    "$SCRIPT_DIR"/stage_2_add_unambiguously_codable_status.py -n "STG2" -b "$batch_size" "$output_folder""/STG1.parquet" "$output_folder""/STG1_metadata.json" "$output_folder"

    echo "RUNNING: STAGE 3";
    "$SCRIPT_DIR"/stage_3_add_open_questions.py -n "STG3" -b "$batch_size" "$output_folder""/STG2.parquet" "$output_folder""/STG2_metadata.json" "$output_folder"

    echo "RUNNING: STAGE 4";
    "$SCRIPT_DIR"/stage_4_add_synthetic_responses.py -n "STG4" -b "$batch_size" "$output_folder""/STG3.parquet" "$output_folder""/STG3_metadata.json" "$output_folder"
fi

echo "RUNNING: STAGE 5"
"$SCRIPT_DIR"/stage_5_rephrase_industry_description.py -n "STG5" -b "$batch_size" "$output_folder""/STG4.parquet" "$output_folder""/STG4_metadata.json" "$output_folder"


echo "RUNNING: STAGE 1 (second search)"
"$SCRIPT_DIR"/stage_1_add_semantic_search.py -n "STG1_second_search" -b "$batch_size" "$output_folder""/STG5.csv" -s "$input_metadata_json" "$output_folder"

echo "RUNNING: STAGE 2 (final classification)"
"$SCRIPT_DIR"/stage_2_add_unambiguously_codable_status.py -n "STG2_final" -b 10 -s "$output_folder""/STG1_second_search.parquet" "$output_folder""/STG1_second_search_metadata.json" "$output_folder"

# echo "RUNNING: STAGE 5";
# "$SCRIPT_DIR"/stage_5_assign_final_sic_code.py -n "STG5" -b "$batch_size" "$output_folder""/STG4.parquet" "$output_folder""/STG4_metadata.json" "$output_folder"

echo "Pipeline Completed Successfully!"
