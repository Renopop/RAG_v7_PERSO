"""
Chunking module - Découpage intelligent des documents

Ce module gère le découpage des documents en chunks:
- chunking: Chunking générique et EASA-spécifique
- semantic_chunking: Chunking basé sur la sémantique
- easa_sections: Parsing des sections EASA
"""

from .chunking import (
    simple_chunk,
    chunk_easa_sections,
    smart_chunk_generic,
    augment_chunks,
    add_cross_references_to_chunks,
    extract_cross_references,
    add_cross_references_to_chunk,
    get_related_chunks_by_reference,
    expand_chunk_context,
    get_neighboring_chunks,
    _calculate_content_density,
)

from .easa_sections import (
    split_easa_sections,
)

# Semantic chunking - optionnel
try:
    from .semantic_chunking import (
        semantic_chunk,
        AdaptiveSemanticChunker,
        SemanticChunk,
        BoundaryType,
    )
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

__all__ = [
    # chunking
    "simple_chunk",
    "chunk_easa_sections",
    "smart_chunk_generic",
    "augment_chunks",
    "add_cross_references_to_chunks",
    "extract_cross_references",
    "add_cross_references_to_chunk",
    "get_related_chunks_by_reference",
    "expand_chunk_context",
    "get_neighboring_chunks",
    "_calculate_content_density",
    # easa_sections
    "split_easa_sections",
    # semantic_chunking
    "semantic_chunk",
    "AdaptiveSemanticChunker",
    "SemanticChunk",
    "BoundaryType",
    "SEMANTIC_CHUNKING_AVAILABLE",
]
