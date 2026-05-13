"""This module provides utilities for embedding and searching SIC index data.

It includes functionality to build vector store from published SIC index files (xls).
It loads the SIC index and structure files, constructs the SIC hierarchy, extracts leaf node text,
and builds an embedding vector store using the EmbeddingHandler class.
"""

import logging
import tempfile

from industrial_classification.hierarchy.sic_hierarchy import load_hierarchy

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.utils.constants import get_default_config
from industrial_classification_utils.utils.sic_data_access import (
    load_sic_index,
    load_sic_structure,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_default_config()


def load_embedding_handler_from_sic_index_files(
    *,
    db_dir: str = config["embedding"].db_dir,
    sic_index_file: tuple[str, str] = config["lookups"]["sic_index"],
    sic_structure_file: tuple[str, str] = config["lookups"]["sic_structure"],
    **kwargs,
) -> EmbeddingHandler:
    """Utility function to load an EmbeddingHandler instance with default configuration.

    Args:
        db_dir: Directory where the vector store is located or will be created.
        sic_index_file: Tuple of (file_path, file_type) for the SIC index file (xls).
        sic_structure_file: Tuple of (file_path, file_type) for the SIC structure file (xls).
        **kwargs: Additional keyword arguments to pass to the EmbeddingHandler constructor
            (e.g., embedding_model_name, k_matches).

    Returns:
        An instance of EmbeddingHandler initialized with the (published) SIC index data.
    """
    logger.info("Loading SIC index file: %s", sic_index_file)
    sic_index_df = load_sic_index(sic_index_file)

    logger.info("Loading SIC structure file: %s", sic_structure_file)
    sic_df = load_sic_structure(sic_structure_file)

    sic = load_hierarchy(sic_df, sic_index_df)

    df = sic.all_leaf_text()
    df["label"] = df["code"].apply(
        lambda x: (x.replace(".", "").replace("/", "") + "0")[:5]
    )

    # write to temporary csv for vector store build
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp_csv:
        df.to_csv(temp_csv.name, index=False)
        logger.info("Temporary CSV for vector store created at: %s", temp_csv.name)
        return EmbeddingHandler(
            db_dir=db_dir, index_source_file=temp_csv.name, **kwargs
        )
