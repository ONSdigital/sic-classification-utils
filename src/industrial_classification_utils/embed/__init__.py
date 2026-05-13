"""Embedding utilities for SIC classification and semantic search."""

from .embedding import EmbeddingHandler
from .sic_specific_embed import load_embedding_handler_from_sic_index_files

__all__ = ["EmbeddingHandler", "load_embedding_handler_from_sic_index_files"]
