#!/usr/bin/env bash

echo "USAGE: ./run_full_pipeline.sh <1 for one-prompt, 2 for 2-prompt> <output_folder> <input_csv> <input_metadata_json> <batch_size>"
echo ""
echo "Keep in mind - you need a local vector store running for stage 1, and gcloud authentication for later stages."
echo ""

if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters" >&2
    exit 2
fi

pipeline_choice=$1
output_folder=$2
input_csv=$3
input_metadata_json=$4
batch_size=$5

echo "RUNNING: STAGE 1"
./stage_1_add_semantic_search.py -n "STG1" -b "$batch_size" "$input_csv" "$input_metadata_json" "$output_folder"
rm -r "$output_folder""/intermediate_outputs"

if [ "$pipeline_choice" -eq "1" ]; then
    echo "RUNNING: STAGE 2 (one-prompt pipeline)";
    ./stage_2_get_rag_sic_code.py -n "STG2_oneprompt" -b "$batch_size" "$output_folder""/STG1.parquet" "$output_folder""/STG1_metadata.json" "$output_folder"
    rm -r "$output_folder""/intermediate_outputs"
else
    echo "RUNNING: STAGE 2 (two-prompt pipeline)";
    ./stage_2_get_rag_sic_code.py -n "STG2" -b "$batch_size" "$output_folder""/STG1.parquet" "$output_folder""/STG1_metadata.json" "$output_folder"
    rm -r "$output_folder""/intermediate_outputs"

    echo "RUNNING: STAGE 3";
    ./stage_2_get_rag_sic_code.py -n "STG3" -b "$batch_size" "$output_folder""/STG2.parquet" "$output_folder""/STG2_metadata.json" "$output_folder"
    rm -r "$output_folder""/intermediate_outputs"

    echo "RUNNING: STAGE 4";
    ./stage_2_get_rag_sic_code.py -n "STG4" -b "$batch_size" "$output_folder""/STG3.parquet" "$output_folder""/STG3_metadata.json" "$output_folder"
    rm -r "$output_folder""/intermediate_outputs"

    echo "RUNNING: STAGE 5";
    ./stage_2_get_rag_sic_code.py -n "STG5" -b "$batch_size" "$output_folder""/STG4.parquet" "$output_folder""/STG4_metadata.json" "$output_folder"
    rm -r "$output_folder""/intermediate_outputs"
fi

echo "Pipeline Completed Successfully!"