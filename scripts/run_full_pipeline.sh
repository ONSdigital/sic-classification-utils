#!/usr/bin/env bash

echo "USAGE: ./run_full_pipeline.sh -p <1 for one-prompt, 2 for 2-prompt> -i <input_csv> -o <output_folder> -m <input_metadata_json> -b <batch_size>"
echo ""
echo "Keep in mind - gcloud authentication needed for later stages."
echo ""

set -e

# Parse arguments using  -i (input_file), -m (input_metadata_json), -o (output_folder), -b (batch_size), -p (pipeline_choice)
while getopts "p:o:i:m:b:" opt; do
    case $opt in
        p) pipeline_choice="$OPTARG" ;;
        o) output_folder="$OPTARG" ;;
        i) input_file="$OPTARG" ;;
        m) input_metadata_json="$OPTARG" ;;
        b) batch_size="$OPTARG" ;;
        *)
            echo "USAGE: ./run_full_pipeline.sh -p <1 or 2> -o <output_folder> -i <input_file> -m <input_metadata_json> -b <batch_size>"
            exit 2
            ;;
    esac
done


# Set default for pipeline_choice if not provided
pipeline_choice="${pipeline_choice:-2}"

# Set default for batch_size if not provided
batch_size="${batch_size:-100}"

# input_metadata_json is optional, set to empty string if not provided
input_metadata_json="${input_metadata_json:-}"

# Check required arguments
if [ -z "$output_folder" ] || [ -z "$input_file" ]; then
    echo "Missing required arguments."
    echo "USAGE: ./run_full_pipeline.sh -p <1 or 2> -o <output_folder> -i <input_file> -m <input_metadata_json> -b <batch_size>"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RUNNING: STAGE 1"
"$SCRIPT_DIR"/stage_1_add_semantic_search.py -r -n "STG1" -b "$batch_size" -i "$input_file" -m "$input_metadata_json" -o "$output_folder"

if [ "$pipeline_choice" -eq "1" ]; then
    echo "RUNNING: STAGE 2 (one-prompt pipeline)";
    "$SCRIPT_DIR"/stage_2_one_prompt_assign_sic_code.py -n "STG2" -i "$output_folder""/STG1.parquet" -m "$output_folder""/STG1_metadata.json" -o "$output_folder"

    echo "RUNNING: STAGE 4 (one-prompt pipeline)";
    "$SCRIPT_DIR"/stage_4_add_synthetic_responses.py -n "STG4" -i "$output_folder""/STG2.parquet" -m "$output_folder""/STG2_metadata.json" -o "$output_folder"

else
    echo "RUNNING: STAGE 2 (initial classification)";
    "$SCRIPT_DIR"/stage_2_add_unambiguously_codable_status.py -r -n "STG2" -i "$output_folder""/STG1.parquet" -m "$output_folder""/STG1_metadata.json" -o "$output_folder"

    echo "RUNNING: STAGE 3";
    "$SCRIPT_DIR"/stage_3_add_open_questions.py -r -n "STG3" -i "$output_folder""/STG2.parquet" -m "$output_folder""/STG2_metadata.json" -o "$output_folder"

    echo "RUNNING: STAGE 4";
    "$SCRIPT_DIR"/stage_4_add_synthetic_responses.py -n "STG4" -i "$output_folder""/STG3.parquet" -m "$output_folder""/STG3_metadata.json" -o "$output_folder"
fi

echo "RUNNING: STAGE 5"
"$SCRIPT_DIR"/stage_5_modify_industry_description.py -n "STG5" -i "$output_folder""/STG4.parquet" -m "$output_folder""/STG4_metadata.json" -o "$output_folder"

echo "RUNNING: STAGE 6"
"$SCRIPT_DIR"/stage_1_add_semantic_search.py -s -n "STG6" -i "$output_folder""/STG5.csv" -m "$output_folder""/STG5_metadata.json" -o "$output_folder"

echo "RUNNING: STAGE 7"
"$SCRIPT_DIR"/stage_2_add_unambiguously_codable_status.py -s -n "STG7" -i "$output_folder""/STG6.parquet" -m "$output_folder""/STG6_metadata.json" -o "$output_folder"

echo "Pipeline Completed Successfully!"
