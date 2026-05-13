"""Embedding utilities for SIC classification and semantic search."""

from .embedding import EmbeddingHandler, SearchIndexResponse
from .sic_specific_embed import load_embedding_handler_from_sic_index_files

__all__ = [
    "EmbeddingHandler",
    "SearchIndexResponse",
    "load_embedding_handler_from_sic_index_files",
]
