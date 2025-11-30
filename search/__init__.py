"""
Search module - Recherche et ranking

Ce module gère la recherche dans les collections:
- hybrid_search: Recherche hybride BM25 + dense
- advanced_search: Query expansion, reranking BGE
- multi_collection_search: Recherche fédérée inter-bases
- query_understanding: Analyse d'intention et entités
"""

# Hybrid search
try:
    from .hybrid_search import (
        BM25Index,
        should_use_hybrid_search,
        hybrid_search,
        build_or_load_bm25_index,
        get_collection_documents,
        HYBRID_MIN_CHUNKS,
    )
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    HYBRID_MIN_CHUNKS = 1000

# Advanced search
try:
    from .advanced_search import (
        expand_query_with_llm,
        run_multi_query_search,
        apply_reranking_to_sources,
        extract_keywords,
        filter_sources_by_keywords,
        reorder_for_lost_in_middle,
        generate_hypothetical_document,
        search_with_hyde,
        rerank_with_bge,
    )
    ADVANCED_SEARCH_AVAILABLE = True
    BGE_RERANKER_AVAILABLE = True
    KEYWORD_FILTER_AVAILABLE = True
    HYDE_AVAILABLE = True
    LOST_IN_MIDDLE_AVAILABLE = True
except ImportError:
    ADVANCED_SEARCH_AVAILABLE = False
    BGE_RERANKER_AVAILABLE = False
    KEYWORD_FILTER_AVAILABLE = False
    HYDE_AVAILABLE = False
    LOST_IN_MIDDLE_AVAILABLE = False

# Multi-collection search
try:
    from .multi_collection_search import (
        MultiCollectionSearcher,
        create_multi_collection_searcher,
        multi_collection_query,
        format_sources_with_provenance,
    )
    MULTI_COLLECTION_AVAILABLE = True
except ImportError:
    MULTI_COLLECTION_AVAILABLE = False

# Query understanding
try:
    from .query_understanding import (
        QueryAnalyzer,
        QueryIntent,
        QueryComplexity,
        QueryDomain,
        analyze_query,
        get_adaptive_top_k,
        expand_query_for_intent,
        get_hybrid_search_weights,
    )
    QUERY_UNDERSTANDING_AVAILABLE = True
except ImportError:
    QUERY_UNDERSTANDING_AVAILABLE = False

__all__ = [
    # hybrid_search
    "BM25Index",
    "should_use_hybrid_search",
    "hybrid_search",
    "build_or_load_bm25_index",
    "get_collection_documents",
    "HYBRID_MIN_CHUNKS",
    "HYBRID_SEARCH_AVAILABLE",
    # advanced_search
    "expand_query_with_llm",
    "run_multi_query_search",
    "apply_reranking_to_sources",
    "extract_keywords",
    "filter_sources_by_keywords",
    "reorder_for_lost_in_middle",
    "generate_hypothetical_document",
    "search_with_hyde",
    "rerank_with_bge",
    "ADVANCED_SEARCH_AVAILABLE",
    "BGE_RERANKER_AVAILABLE",
    "KEYWORD_FILTER_AVAILABLE",
    "HYDE_AVAILABLE",
    "LOST_IN_MIDDLE_AVAILABLE",
    # multi_collection_search
    "MultiCollectionSearcher",
    "create_multi_collection_searcher",
    "multi_collection_query",
    "format_sources_with_provenance",
    "MULTI_COLLECTION_AVAILABLE",
    # query_understanding
    "QueryAnalyzer",
    "QueryIntent",
    "QueryComplexity",
    "QueryDomain",
    "analyze_query",
    "get_adaptive_top_k",
    "expand_query_for_intent",
    "get_hybrid_search_weights",
    "QUERY_UNDERSTANDING_AVAILABLE",
]
