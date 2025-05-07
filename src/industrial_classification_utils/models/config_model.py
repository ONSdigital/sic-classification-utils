"""This module defines configuration models for the industrial classification utilities.

The models are implemented using Python's `TypedDict` and are used to represent
configuration settings for various components of the system, such as language
models and lookup tables.

Classes:
    LLMConfig: Configuration for language and embedding models.
    LookupsConfig: Configuration for SIC-related lookup tables.
"""

from typing import TypedDict


class LLMConfig(TypedDict):
    """Configuration for language and embedding models and location of
    the vector store.

    Attributes:
        llm_model_name (str): Name of the language model.
        embedding_model_name (str): Name of the embedding model.
        db_dir (str): Directory for the database.
    """

    llm_model_name: str
    embedding_model_name: str
    db_dir: str


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
        llm (LLMConfig): Configuration for language and embedding models.
        lookups (LookupsConfig): Configuration for SIC-related lookup tables.
    """

    llm: LLMConfig
    lookups: LookupsConfig
