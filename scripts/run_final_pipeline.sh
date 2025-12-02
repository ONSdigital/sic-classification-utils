#!/usr/bin/env bash

echo "USAGE: ./run_full_pipeline.sh <output_folder> <input_parquet> <input_metadata_json> <batch_size>"
echo ""
echo "Keep in mind - you need a local vector store running for stage 1, and gcloud authentication for later stages."
echo ""

set -e

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters" <&2
    exit 2
fi

output_folder=$1
input_parquet=$2
input_metadata_json=$3
batch_size=$4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RUNNING: STAGE 5"
"$SCRIPT_DIR"/stage_5_rephrase_industry_description.py -n "STG5" -b "$batch_size" "$input_parquet" "$input_metadata_json" "$output_folder"

echo "RUNNING: STAGE 1"
"$SCRIPT_DIR"/stage_1_add_semantic_search.py -n "STG1_second_search" -b "$batch_size" "$output_folder""/STG5.csv" -s "$input_metadata_json" "$output_folder"

echo "RUNNING: STAGE 2"
"$SCRIPT_DIR"/stage_2_add_unambiguously_codable_status.py -n "STG2_final" -b 10 -s "$output_folder""/STG1_second_search.parquet" "$output_folder""/STG1_second_search_metadata.json" "$output_folder"

echo "Pipeline Completed Successfully!"
