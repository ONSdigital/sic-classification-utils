#!/usr/bin/env bash

echo "USAGE: ./run_final_pipeline.sh -o <output_folder> -i <input_parquet> -m <input_metadata_json> -b <batch_size>"
echo ""
echo "Keep in mind - you need a local vector store running for stage 1, and gcloud authentication for later stages."
echo ""

set -e

# Parse arguments using -o (output_folder), -i (input_parquet), -m (input_metadata_json), -b (batch_size)
while getopts "o:i:m:b:" opt; do
    case $opt in
        o) output_folder="$OPTARG" ;;
        i) input_parquet="$OPTARG" ;;
        m) input_metadata_json="$OPTARG" ;;
        b) batch_size="$OPTARG" ;;
        *)
            echo "USAGE: ./run_final_pipeline.sh -o <output_folder> -i <input_parquet> -m <input_metadata_json> -b <batch_size>"
            exit 2
            ;;
    esac
done

# Set default for batch_size if not provided
batch_size="${batch_size:-100}"

# input_metadata_json is optional; use empty string if not provided
input_metadata_json="${input_metadata_json:-}"

# Check required arguments
if [ -z "$output_folder" ] || [ -z "$input_parquet" ]; then
    echo "Missing required arguments." <&2
    echo "USAGE: ./run_final_pipeline.sh -o <output_folder> -i <input_parquet> -m <input_metadata_json> -b <batch_size>"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RUNNING: STAGE 5"
"$SCRIPT_DIR"/stage_5_modify_industry_description.py -n "STG5" -b "$batch_size" -i "$input_parquet" -m "$input_metadata_json" -o "$output_folder"

echo "RUNNING: STAGE 6"
"$SCRIPT_DIR"/stage_1_add_semantic_search.py -s -n "STG6" -b "$batch_size" -i "$output_folder""/STG5.csv" -m "$output_folder""/STG5_metadata.json" -o "$output_folder"

echo "RUNNING: STAGE 7"
"$SCRIPT_DIR"/stage_2_add_unambiguously_codable_status.py -s -n "STG7" -b 10 -i "$output_folder""/STG6.parquet" -m "$output_folder""/STG6_metadata.json" -o "$output_folder"

echo "Pipeline Completed Successfully!"
