"""
Ingestion module - Pipeline d'ingestion des documents

Ce module gère l'indexation des documents:
- ingestion_pipeline: Pipeline optimisé d'ingestion
- rag_ingestion: API legacy pour compatibilité
"""

from .ingestion_pipeline import (
    OptimizedIngestionPipeline,
    quick_ingest,
    ingest_csv,
    ingest_documents,
    FileInfo,
    ExtractionResult,
    ChunkInfo,
    PipelineStats,
)

# Legacy API
from .rag_ingestion import (
    ingest_documents as legacy_ingest_documents,
)

__all__ = [
    # ingestion_pipeline
    "OptimizedIngestionPipeline",
    "quick_ingest",
    "ingest_csv",
    "ingest_documents",
    "FileInfo",
    "ExtractionResult",
    "ChunkInfo",
    "PipelineStats",
    # rag_ingestion (legacy)
    "legacy_ingest_documents",
]
