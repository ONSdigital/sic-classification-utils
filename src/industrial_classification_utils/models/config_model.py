"""This module defines configuration models for the industrial classification utilities.

The models are implemented using Python's `TypedDict` and are used to represent
configuration settings for various components of the system, such as language
models and lookup tables.

Classes:
    LLMConfig: Configuration for language and embedding models.
    LookupsConfig: Configuration for SIC-related lookup tables.
"""

from typing import TypedDict

from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model and vector store.

    Attributes:
        embedding_model_name (str): Name of the embedding model.
        db_dir (str): Directory for the database.
        index_source_file (str | None): Optional path to the source file for the index.
        k_matches (int): Number of matches to return in similarity search.
    """

    embedding_model_name: str
    db_dir: str
    index_source_file: str | None = None
    k_matches: int


class EmbeddingStatus(EmbeddingConfig):
    """Extended configuration model that includes the status and size of the embedding index.

    Attributes:
        embedding_model_name (str): Name of the embedding model.
        db_dir (str): Directory for the database.
        k_matches (int): Number of matches to return in similarity search.
        status (str): Status of the embedding index (e.g., "initialized", "updated
        index_size (int | None): Optional field to track the size of the index.
    """

    status: str
    index_size: int


class LLMConfig(TypedDict):
    """Configuration for language and embedding models and location of
    the vector store.

    Attributes:
        llm_model_name (str): Name of the language model.
        model_location (str): Location of the model.
        code_digits (int): Number of digits in the SIC code.
        candidates_limit (int): Maximum number of candidate SIC codes to return.
    """

    llm_model_name: str
    model_location: str
    code_digits: int
    candidates_limit: int


class LookupsConfig(TypedDict):
    """Configuration for SIC-related lookup tables.

    Attributes:
        sic_index (tuple[str, str]): Path to the SIC index file.
        sic_structure (tuple[str, str]): Path to the SIC structure file.
        sic_condensed (tuple[str, str]): Path to the condensed SIC file.
    """

    sic_index: tuple[str, str]
    sic_structure: tuple[str, str]
    sic_condensed: tuple[str, str]


class FullConfig(TypedDict):
    """Full configuration model for the SIC classification.

    Attributes:
        embedding (EmbeddingConfig): Configuration for embedding model and vector store.
        llm (LLMConfig): Configuration for language and embedding models.
        lookups (LookupsConfig): Configuration for SIC-related lookup tables.
    """

    embedding: EmbeddingConfig
    llm: LLMConfig
    lookups: LookupsConfig
