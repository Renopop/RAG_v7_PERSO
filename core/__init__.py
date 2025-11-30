"""
Core module - Infrastructure de base du RAG

Ce module contient les composants fondamentaux:
- models_utils: Clients API, embeddings, configuration LLM
- faiss_store: Stockage vectoriel FAISS
- semantic_cache: Cache sémantique pour les requêtes
- config_manager: Gestion de la configuration
"""

from .models_utils import (
    make_logger,
    EMBED_MODEL,
    BATCH_SIZE,
    DirectOpenAIEmbeddings,
    embed_in_batches,
    SNOWFLAKE_API_KEY,
    SNOWFLAKE_API_BASE,
    create_http_client,
    call_dallem_chat,
)

from .faiss_store import (
    FaissStore,
    FaissCollection,
    build_faiss_store,
    get_or_create_collection,
    list_all_collections,
)

from .config_manager import (
    load_config,
    save_config,
    StorageConfig,
    is_config_valid,
    validate_all_directories,
    create_directory,
)

# Imports optionnels
try:
    from .semantic_cache import (
        SemanticCache,
        get_semantic_cache,
        invalidate_cache,
    )
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False

__all__ = [
    # models_utils
    "make_logger",
    "EMBED_MODEL",
    "BATCH_SIZE",
    "DirectOpenAIEmbeddings",
    "embed_in_batches",
    "SNOWFLAKE_API_KEY",
    "SNOWFLAKE_API_BASE",
    "create_http_client",
    "call_dallem_chat",
    # faiss_store
    "FaissStore",
    "FaissCollection",
    "build_faiss_store",
    "get_or_create_collection",
    "list_all_collections",
    # config_manager
    "load_config",
    "save_config",
    "StorageConfig",
    "is_config_valid",
    "validate_all_directories",
    "create_directory",
    # semantic_cache
    "SemanticCache",
    "get_semantic_cache",
    "invalidate_cache",
    "SEMANTIC_CACHE_AVAILABLE",
]
