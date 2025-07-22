#!/usr/bin/env python3
"""Simplified industrial classification script with batching, logging, and checkpointing."""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.llm.llm import ClassificationLLM


@dataclass
class Config:
    """Configuration for processing parameters.""" #noqa R0902

    input_file: str
    output_file: str
    checkpoint_file: str
    batch_size: int = 50
    sleep_between_requests: float = 1.0
    sleep_between_batches: float = 5.0
    code_digits: int = 5
    candidates_limit: int = 7


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("classification_processing.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_checkpoint(config: Config, logger: logging.Logger) -> list[dict[str, Any]]:
    """Load existing results from checkpoint file."""
    if not Path(config.checkpoint_file).exists():
        return []

    try:
        with open(config.checkpoint_file, encoding="utf-8") as f:
            results = [json.loads(line) for line in f]
        logger.info("Loaded %d existing results from checkpoint", len(results))
        return results
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Error loading checkpoint: %s", e)
        return []


def save_result_to_checkpoint(
    result: dict[str, Any], checkpoint_file: str, logger: logging.Logger
) -> None:
    """Save a single result to checkpoint file (append mode)."""
    try:
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    except OSError as e:
        logger.error("Error saving checkpoint: %s", e)


def process_single_row(
    row: pd.Series, uni_chat: ClassificationLLM, config: Config, logger: logging.Logger
) -> dict[str, Any]:
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

        # Access protected method with pylint disable comment
        # pylint: disable=protected-access
        short_list = uni_chat._prompt_candidate_list(
            search_results,
            code_digits=config.code_digits,
            candidates_limit=config.candidates_limit,
        )
        # pylint: enable=protected-access

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

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error processing row %s: %s", row["unique_id"], e)
        return {
            "unique_id": row["unique_id"],
            "codable": None,
            "code": None,
            "alt_candidates": [],
            "error": str(e),
            "processed_at": pd.Timestamp.now().isoformat(),
        }


def process_batch(
    batch_df: pd.DataFrame,
    uni_chat: ClassificationLLM,
    config: Config,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
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


def main_processing( #noqa R0914
    config: Config, resume_from_checkpoint: bool = True
) -> pd.DataFrame:
    """Main processing function."""
    logger = setup_logging()

    # Initialize models
    logger.info("Initializing models...")
    embed = EmbeddingHandler()

    # Create ClassificationLLM - adjust parameters as needed for your implementation
    # uni_chat = ClassificationLLM("gemini-2.0-flash", embeddin_handler=embed, verbose=False)
    uni_chat = ClassificationLLM("gemini-2.0-flash", embed, verbose=False)

    # Load data
    logger.info("Loading data from %s", config.input_file)
    test_set = pd.read_csv(config.input_file)
    test_subset = test_set[
        [
            "unique_id",
            "soc2020_job_title",
            "soc2020_job_description",
            "sic2007_employee",
        ]
    ]

    # Load existing results if resuming
    existing_results = []
    processed_ids = set()

    if resume_from_checkpoint:
        existing_results = load_checkpoint(config, logger)
        processed_ids = {r["unique_id"] for r in existing_results}

    # Filter out already processed rows
    unprocessed_df = test_subset[~test_subset["unique_id"].isin(processed_ids)]
    logger.info(
        "Processing %d rows (out of %d total)", len(unprocessed_df), len(test_subset)
    )

    if len(unprocessed_df) == 0:
        logger.info("All rows already processed!")
        results_df = pd.DataFrame(existing_results)
        results_df.to_csv(config.output_file, index=False)
        return results_df

    # Process in batches
    all_results = existing_results.copy()

    batch_count = (len(unprocessed_df) + config.batch_size - 1) // config.batch_size

    for i in tqdm(
        range(0, len(unprocessed_df), config.batch_size), desc="Processing batches"
    ):
        batch_df = unprocessed_df.iloc[i : i + config.batch_size].copy()
        batch_num = i // config.batch_size + 1
        logger.info(
            "Processing batch %d/%d (%d rows)", batch_num, batch_count, len(batch_df)
        )

        batch_results = process_batch(batch_df, uni_chat, config, logger)
        all_results.extend(batch_results)

        # Sleep between batches (except for the last batch)
        if i + config.batch_size < len(unprocessed_df):
            logger.info(
                "Sleeping %s seconds between batches", config.sleep_between_batches
            )
            time.sleep(config.sleep_between_batches)

    # Convert to DataFrame and save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.output_file, index=False)
    logger.info("Results saved to %s", config.output_file)

    # Print summary statistics
    total = len(results_df)
    successful = results_df["codable"].sum() if "codable" in results_df.columns else 0
    failed = (~results_df["codable"]).sum() if "codable" in results_df.columns else 0
    errors = (
        results_df["codable"].isna().sum() if "codable" in results_df.columns else 0
    )

    logger.info(
        "Processing complete! Total: %d, Successful: %d, Failed: %d, Errors: %d",
        total,
        successful,
        failed,
        errors,
    )

    return results_df


def main() -> None:
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
    )

    results_df = main_processing(config, resume_from_checkpoint=not args.no_resume)

    print("\nProcessing complete!")
    print(f"Total results: {len(results_df)}")
    if "codable" in results_df.columns:
        print(f"Successful classifications: {results_df['codable'].sum()}")
    print(f"Results saved to: {config.output_file}")


def run_classification(input_file: str, output_file: str, **kwargs) -> pd.DataFrame:
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
    )

    resume = kwargs.get("resume_from_checkpoint", True)
    return main_processing(config, resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
