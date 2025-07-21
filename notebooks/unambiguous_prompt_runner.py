#!/usr/bin/env python3
"""Simplified industrial classification script with batching, logging, and checkpointing."""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.llm.llm import ClassificationLLM


@dataclass
class Config:
    """Configuration for processing parameters."""

    input_file: str
    output_file: str
    checkpoint_file: str
    batch_size: int = 50
    sleep_between_requests: float = 1.0
    sleep_between_batches: float = 5.0
    code_digits: int = 5
    candidates_limit: int = 7
    resume_from_checkpoint: bool = True


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("classification_processing.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_checkpoint(config: Config, logger) -> list[dict]:
    """Load existing results from checkpoint file."""
    if not config.resume_from_checkpoint or not Path(config.checkpoint_file).exists():
        return []

    try:
        with open(config.checkpoint_file) as f:
            results = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(results)} existing results from checkpoint")
        return results
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return []


def save_result_to_checkpoint(result: dict, checkpoint_file: str, logger):
    """Save a single result to checkpoint file (append mode)."""
    try:
        with open(checkpoint_file, "a") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def process_single_row(row: pd.Series, uni_chat, config: Config, logger) -> dict:
    """Process a single row with error handling and rate limiting."""
    try:
        # Add delay for rate limiting
        time.sleep(config.sleep_between_requests)

        # Perform the classification
        search_results = uni_chat.embed.search_index_multi(
            [
                row["sic2007_employee"],
                row["soc2020_job_title"],
                row["soc2020_job_description"],
            ]
        )

        short_list = uni_chat._prompt_candidate_list(
            search_results,
            code_digits=config.code_digits,
            candidates_limit=config.candidates_limit,
        )

        sa_response = uni_chat.unambiguous_sic_code(
            industry_descr=row["sic2007_employee"],
            job_title=row["soc2020_job_title"],
            job_description=row["soc2020_job_description"],
            shortlist=short_list,
        )

        result = {
            "unique_id": row["unique_id"],
            "codable": sa_response[0].codable,
            "code": sa_response[0].class_code,
            "alt_candidates": [i.class_code for i in sa_response[0].alt_candidates],
            "processed_at": pd.Timestamp.now().isoformat(),
        }

        return result

    except Exception as e:
        logger.error(f"Error processing row {row['unique_id']}: {e}")
        return {
            "unique_id": row["unique_id"],
            "codable": None,
            "code": None,
            "alt_candidates": [],
            "error": str(e),
            "processed_at": pd.Timestamp.now().isoformat(),
        }


def process_batch(
    batch_df: pd.DataFrame, uni_chat, config: Config, logger
) -> list[dict]:
    """Process a batch of rows."""
    results = []

    for _idx, row in tqdm(
        batch_df.iterrows(), total=len(batch_df), desc="Processing batch"
    ):
        result = process_single_row(row, uni_chat, config, logger)
        results.append(result)

        # Save immediately to checkpoint
        save_result_to_checkpoint(result, config.checkpoint_file, logger)

    return results


def main_processing(config: Config):
    """Main processing function."""
    logger = setup_logging()

    # Initialize models
    logger.info("Initializing models...")
    embed = EmbeddingHandler()
    uni_chat = ClassificationLLM(
        "gemini-2.0-flash", embedding_handler=embed, verbose=False
    )

    # Load data
    logger.info(f"Loading data from {config.input_file}")
    test_set = pd.read_csv(config.input_file)
    test_subset = test_set[
        [
            "unique_id",
            "soc2020_job_title",
            "soc2020_job_description",
            "sic2007_employee",
        ]
    ]

    # Load existing results
    existing_results = load_checkpoint(config, logger)
    processed_ids = {r["unique_id"] for r in existing_results}

    # Filter out already processed rows
    unprocessed_df = test_subset[~test_subset["unique_id"].isin(processed_ids)]
    logger.info(
        f"Processing {len(unprocessed_df)} rows (out of {len(test_subset)} total)"
    )

    if len(unprocessed_df) == 0:
        logger.info("All rows already processed!")
        results_df = pd.DataFrame(existing_results)
        results_df.to_csv(config.output_file, index=False)
        return results_df

    # Process in batches
    all_results = existing_results.copy()

    for i in tqdm(
        range(0, len(unprocessed_df), config.batch_size), desc="Processing batches"
    ):
        batch_df = unprocessed_df.iloc[i : i + config.batch_size].copy()
        batch_num = i // config.batch_size + 1
        logger.info(f"Processing batch {batch_num} ({len(batch_df)} rows)")

        batch_results = process_batch(batch_df, uni_chat, config, logger)
        all_results.extend(batch_results)

        # Sleep between batches (except for the last batch)
        if i + config.batch_size < len(unprocessed_df):
            logger.info(f"Sleeping {config.sleep_between_batches}s between batches")
            time.sleep(config.sleep_between_batches)

    # Convert to DataFrame and save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.output_file, index=False)
    logger.info(f"Results saved to {config.output_file}")

    # Print summary statistics
    total = len(results_df)
    successful = (results_df["codable"]).sum()
    failed = (not results_df["codable"]).sum()
    errors = results_df["codable"].isna().sum()

    logger.info(
        f"Processing complete! Total: {total}, Successful: {successful}, Failed: {failed}, Errors: {errors}"
    )

    return results_df


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(description="Industrial Classification Processing")
    parser.add_argument("--input-file", required=True, help="Input CSV file path")
    parser.add_argument("--output-file", required=True, help="Output CSV file path")
    parser.add_argument(
        "--checkpoint-file",
        default="classification_checkpoint.jsonl",
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing"
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=1.0,
        help="Sleep time between individual requests (seconds)",
    )
    parser.add_argument(
        "--sleep-between-batches",
        type=float,
        default=5.0,
        help="Sleep time between batches (seconds)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Do not resume from checkpoint"
    )

    args = parser.parse_args()

    config = Config(
        input_file=args.input_file,
        output_file=args.output_file,
        checkpoint_file=args.checkpoint_file,
        batch_size=args.batch_size,
        sleep_between_requests=args.sleep_between_requests,
        sleep_between_batches=args.sleep_between_batches,
        resume_from_checkpoint=not args.no_resume,
    )

    results_df = main_processing(config)

    print("\nProcessing complete!")
    print(f"Total results: {len(results_df)}")
    print(f"Successful classifications: {(results_df['codable']).sum()}")
    print(f"Results saved to: {config.output_file}")


# Alternative: Use directly as a module
def run_classification(input_file: str, output_file: str, **kwargs):
    """Convenience function to run classification programmatically."""
    config = Config(
        input_file=input_file,
        output_file=output_file,
        checkpoint_file=kwargs.get(
            "checkpoint_file", "classification_checkpoint.jsonl"
        ),
        batch_size=kwargs.get("batch_size", 50),
        sleep_between_requests=kwargs.get("sleep_between_requests", 1.0),
        sleep_between_batches=kwargs.get("sleep_between_batches", 5.0),
        resume_from_checkpoint=kwargs.get("resume_from_checkpoint", True),
    )

    return main_processing(config)


if __name__ == "__main__":
    main()
