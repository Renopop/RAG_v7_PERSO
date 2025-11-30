"""
Query module - Exécution des requêtes RAG

Ce module gère l'exécution des requêtes:
- rag_query: Fonctions principales de requête RAG
- answer_grounding: Détection d'hallucinations
- rag_metrics: Métriques de qualité RAGAS
"""

from .rag_query import (
    run_rag_query,
    run_multi_collection_rag_query,
    build_store,
    MULTI_COLLECTION_AVAILABLE,
)

# Answer grounding
try:
    from .answer_grounding import (
        AnswerGroundingAnalyzer,
        analyze_grounding,
        get_grounding_warning,
        AnswerGroundingReport,
    )
    ANSWER_GROUNDING_AVAILABLE = True
except ImportError:
    ANSWER_GROUNDING_AVAILABLE = False

# RAG metrics
try:
    from .rag_metrics import (
        RAGEvaluator,
        RAGMetrics,
        quick_evaluate,
        format_metrics_report,
    )
    RAG_METRICS_AVAILABLE = True
except ImportError:
    RAG_METRICS_AVAILABLE = False

__all__ = [
    # rag_query
    "run_rag_query",
    "run_multi_collection_rag_query",
    "build_store",
    "MULTI_COLLECTION_AVAILABLE",
    # answer_grounding
    "AnswerGroundingAnalyzer",
    "analyze_grounding",
    "get_grounding_warning",
    "AnswerGroundingReport",
    "ANSWER_GROUNDING_AVAILABLE",
    # rag_metrics
    "RAGEvaluator",
    "RAGMetrics",
    "quick_evaluate",
    "format_metrics_report",
    "RAG_METRICS_AVAILABLE",
]
