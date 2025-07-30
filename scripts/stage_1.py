"""
TODO
"""
import pandas as pd
import requests
from argparse import ArgumentParser as AP
import os
import json

#####################################################
# Constants:
METADATA = {}

VECTOR_STORE_URL_BASE = 'http://0.0.0.0:8088'
STATUS_ENDPOINT = '/v1/sic-vector-store/status'
SEARCH_ENDPOINT = '/v1/sic-vector-store/search-index'

INDUSTRY_DESCR_COL = 'sic2007_employee'
JOB_TITLE_COL = 'soc2020_job_title'
JOB_DESCRIPTION_COL = 'soc2020_job_description'
#####################################################

def parse_args():
    """
    TODO
    """
    parser = AP()
    parser.add_argument('input_file', 
                        type=str, 
                        help="relative path to the input CSV dataset")
    parser.add_argument('output_folder', 
                        type=str, 
                        help="relative path to the output folder location (will be created if it doesn't exist)")
    parser.add_argument('--output_shortname', 
                        '-n', 
                        type=str, 
                        default='STG1', 
                        help='output filename prefix for easy identification (optional, default: STG1)')
    return parser.parse_args()

def check_vector_store_ready():
    """
    TODO
    """
    response = requests.get(f'{VECTOR_STORE_URL_BASE}{STATUS_ENDPOINT}')
    try:
        response.raise_for_status()
    except:
        print('Could not interact with locally-running vector store')
        raise
    if response.json()['status'] != "ready":
        raise EnvironmentError('The vector store is still loading, re-try in a few minutes')
    return True

def get_semantic_search_results(row):
    """
    TODO
    """
    payload = {
        "industry_descr": row[INDUSTRY_DESCR_COL],
        "job_title": row[JOB_TITLE_COL],
        "job_description": row[JOB_DESCRIPTION_COL]
    }

    # Prevent undefined behaviour from VectorStore by
    # sanitising non-string inputs (e.g. None)
    for k in payload.keys():
        if not isinstance(payload[k], str):
            payload[k] = ""

    response = requests.post(f'{VECTOR_STORE_URL_BASE}{SEARCH_ENDPOINT}', json=payload)
    response.raise_for_status()
    response_json = response.json()
    try:
        results = response_json['results']

    except (KeyError, AttributeError) as e:
        print('results key missing from JSON response from vector store', response_json)
        raise e
    
    reduced_results = [{'code': r['code'], 'distance': r['distance']} for r in results]
    return reduced_results

def persist_results(df: pd.DataFrame, output_folder: str):
    """
    TODO
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Saving results to CSV...")
    df.to_csv(f'{output_folder}/{args.output_shortname}.csv')
    print("Saving results to pickle...")
    df.to_pickle(f'{output_folder}/{args.output_shortname}.gz')
    print("Saving setup metadata to JSON...")
    with open(f'{output_folder}/{args.output_shortname}_metadata.json', 'w') as f:
        json.dump(METADATA, f)
    print('Done!')

if __name__=='__main__':
    args = parse_args()

    check_vector_store_ready()
    print('Vector store is ready')

    df = pd.read_csv(args.input_file)
    print('Input loaded')

    print('running semantic search...')
    df['semantic_search_results'] = df.apply(get_semantic_search_results, axis=1)
    print('semantic search complete')

    print('persisting results...')
    persist_results(df, args.output_folder)



