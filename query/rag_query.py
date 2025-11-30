# rag_query.py

import os
import re
import time
from typing import Any, Dict, List, Optional

from core.faiss_store import FaissStore

from core.models_utils import (
    EMBED_MODEL,
    BATCH_SIZE,  # même si non utilisé directement, laissé pour compatibilité
    SNOWFLAKE_API_KEY,
    SNOWFLAKE_API_BASE,
    make_logger,
    create_http_client,
    DirectOpenAIEmbeddings,
    embed_in_batches,
    call_dallem_chat,
)

# Import optionnel du FeedbackStore pour le re-ranking
try:
    from feedback.feedback_store import FeedbackStore
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    FeedbackStore = None

# Import du module de recherche avancée
try:
    from search.advanced_search import (
        expand_query_with_llm,
        run_multi_query_search,
        apply_reranking_to_sources,
        extract_keywords,
        filter_sources_by_keywords,
        # Phase 1 improvements
        reorder_for_lost_in_middle,
        generate_hypothetical_document,
        search_with_hyde,
    )
    from core.models_utils import DALLEM_API_KEY, DALLEM_API_BASE, LLM_MODEL
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

# Import des fonctions de context expansion et cross-references
try:
    from chunking.chunking import (
        extract_cross_references,
        add_cross_references_to_chunk,
        get_related_chunks_by_reference,
        expand_chunk_context,
        get_neighboring_chunks,
    )
    CONTEXT_EXPANSION_AVAILABLE = True
except ImportError:
    CONTEXT_EXPANSION_AVAILABLE = False

# Phase 2: Import recherche hybride BM25
try:
    from search.hybrid_search import (
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

# Phase 2: Import cache sémantique
try:
    from core.semantic_cache import (
        SemanticCache,
        get_semantic_cache,
        invalidate_cache,
    )
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False

# Phase 2: Import métriques RAGAS
try:
    from query.rag_metrics import (
        RAGEvaluator,
        RAGMetrics,
        quick_evaluate,
        format_metrics_report,
    )
    RAG_METRICS_AVAILABLE = True
except ImportError:
    RAG_METRICS_AVAILABLE = False

# Phase 3: Import Answer Grounding (hallucination detection)
try:
    from query.answer_grounding import (
        AnswerGroundingAnalyzer,
        analyze_grounding,
        get_grounding_warning,
        AnswerGroundingReport,
    )
    ANSWER_GROUNDING_AVAILABLE = True
except ImportError:
    ANSWER_GROUNDING_AVAILABLE = False

# Phase 3: Import Query Understanding (intent detection)
try:
    from search.query_understanding import (
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

# Phase 4: Import Multi-Collection Search (federated search)
try:
    from search.multi_collection_search import (
        MultiCollectionSearcher,
        create_multi_collection_searcher,
        multi_collection_query,
        format_sources_with_provenance,
    )
    from core import faiss_store as faiss_store_module
    MULTI_COLLECTION_AVAILABLE = True
except ImportError:
    MULTI_COLLECTION_AVAILABLE = False

logger = make_logger(debug=False)


# =====================================================================
#  FAISS STORE
# =====================================================================

def build_store(db_path: str, use_local_cache: bool = False) -> FaissStore:
    """
    Construit un store FAISS sur le répertoire db_path.
    Pas de retry nécessaire: FAISS fonctionne sans problème sur réseau!

    Args:
        db_path: Chemin de la base FAISS
        use_local_cache: Si True, utilise le cache local pour les lectures
    """
    logger.info(f"[QUERY] Creating FAISS store at: {db_path} (cache={use_local_cache})")
    store = FaissStore(path=db_path, use_local_cache=use_local_cache, lazy_load=True)
    logger.info(f"[QUERY] ✅ FAISS store ready (cache={'enabled' if use_local_cache else 'disabled'})")
    return store


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def _format_section_info(src: Dict[str, Any]) -> str:
    """Format section info for context header (EASA references)."""
    section_id = src.get("section_id") or ""
    section_title = src.get("section_title") or ""

    if not section_id:
        return ""

    section_info = f", section={section_id}"
    if section_title:
        title_preview = f"({section_title[:40]}...)" if len(section_title) > 40 else f"({section_title})"
        section_info += f" {title_preview}"
    return section_info


def _normalize_section_ref(ref: str) -> str:
    """
    Normalise une référence EASA pour la comparaison.

    "CS 25.613" -> "cs25.613"
    "AMC1 25.1309" -> "amc125.1309"
    "CS-APU 25.1309" -> "csapu25.1309"
    """
    if not ref:
        return ""
    # Minuscules, enlever espaces/tirets, garder les points pour la structure
    normalized = ref.lower()
    normalized = re.sub(r'[\s\-]+', '', normalized)
    return normalized


def _extract_section_refs(section_id: str) -> list:
    """
    Extrait toutes les références d'un section_id, gérant les sections fusionnées.

    "CS 25.613" -> ["cs25.613"]
    "CS 25.1 | CS 25.2" -> ["cs25.1", "cs25.2"]
    """
    if not section_id:
        return []

    # Séparer sur | pour les sections fusionnées
    parts = section_id.split("|")
    refs = []
    for part in parts:
        normalized = _normalize_section_ref(part.strip())
        if normalized:
            refs.append(normalized)
    return refs


def _section_matches_entity(section_refs: list, entity: str) -> bool:
    """
    Vérifie si une référence de section correspond à une entité.

    Gère les correspondances:
    - Exactes: "cs25.613" == "cs25.613"
    - Partielles: "cs25" dans "cs25.613" (recherche de chapitre)
    - Sous-paragraphes: "cs25.613" trouve "cs25.613(a)"
    - Fusionnées: "cs25.1" dans ["cs25.1", "cs25.2"]
    """
    if not entity or not section_refs:
        return False

    for ref in section_refs:
        # Match exact
        if entity == ref:
            return True

        # Match partiel (ex: recherche "CS 25" trouve "CS 25.613")
        # Mais éviter les faux positifs: "cs25.1" ne doit pas matcher "cs25.11"
        # On vérifie que le caractère suivant est un séparateur valide
        if ref.startswith(entity):
            remaining = ref[len(entity):]
            # Séparateurs valides: rien, point (sous-section), parenthèse (sous-paragraphe)
            if not remaining or remaining[0] in '.(':
                return True

        # Match inverse: "cs25.613(a)" doit aussi matcher si on cherche "cs25.613"
        if entity.startswith(ref):
            remaining = entity[len(ref):]
            if not remaining or remaining[0] in '.(':
                return True

    return False


def _boost_by_section_id(
    docs: list,
    metas: list,
    dists: list,
    entities: list,
    boost_factor: float = 0.5,
    log=None
) -> tuple:
    """
    Booste les résultats où section_id correspond aux entités EASA détectées.

    Quand un utilisateur demande "Que dit CS 25.613?", les chunks avec
    section_id="CS 25.613" sont remontés en priorité via un boost de distance.

    Gère les cas:
    - Section unique: "CS 25.613"
    - Sections fusionnées: "CS 25.1 | CS 25.2 | CS 25.3"
    - Recherche de chapitre: "CS 25" trouve "CS 25.613"

    Args:
        docs: Liste des documents retournés
        metas: Liste des métadonnées
        dists: Liste des distances L2
        entities: Entités EASA détectées dans la requête (ex: ["CS 25.613", "AMC 25.1309"])
        boost_factor: Réduction de distance pour les matches (0.5 = divise par 2)
        log: Logger optionnel

    Returns:
        (docs, metas, dists) réordonnés avec boost appliqué
    """
    _log = log or logger

    if not entities:
        return docs, metas, dists

    # Normaliser les entités pour la comparaison
    normalized_entities = [_normalize_section_ref(e) for e in entities]
    # Filtrer les entités vides
    normalized_entities = [e for e in normalized_entities if e]

    if not normalized_entities:
        return docs, metas, dists

    _log.debug(f"[BOOST] Looking for section_id matches: {entities}")

    # Calculer les distances boostées
    boosted_results = []
    matches_found = 0

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        section_id = meta.get("section_id", "") if isinstance(meta, dict) else ""

        # Extraire toutes les références (gère les sections fusionnées)
        section_refs = _extract_section_refs(section_id)

        # Vérifier aussi merged_sections si présent
        merged = meta.get("merged_sections", []) if isinstance(meta, dict) else []
        for m in merged:
            section_refs.extend(_extract_section_refs(m))

        # Vérifier si le section_id correspond à une entité
        is_match = False
        for norm_entity in normalized_entities:
            if _section_matches_entity(section_refs, norm_entity):
                is_match = True
                break

        if is_match:
            # Appliquer le boost (réduire la distance)
            boosted_dist = dist * boost_factor
            matches_found += 1
            _log.debug(f"[BOOST] Match found: {section_id} -> dist {dist:.3f} => {boosted_dist:.3f}")
        else:
            boosted_dist = dist

        boosted_results.append((doc, meta, boosted_dist, dist))

    if matches_found > 0:
        _log.info(f"[BOOST] Applied section_id boost to {matches_found} chunks matching {entities}")

    # Trier par distance boostée
    boosted_results.sort(key=lambda x: x[2])

    # Extraire les listes triées
    sorted_docs = [r[0] for r in boosted_results]
    sorted_metas = [r[1] for r in boosted_results]
    sorted_dists = [r[3] for r in boosted_results]  # Distance originale pour l'affichage

    return sorted_docs, sorted_metas, sorted_dists


# =====================================================================
#  RAG SUR UNE SEULE COLLECTION
# =====================================================================

def _run_rag_query_single_collection(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    call_llm: bool = True,
    log=None,
    feedback_store: Optional["FeedbackStore"] = None,
    use_feedback_reranking: bool = False,
    feedback_alpha: float = 0.3,
    use_query_expansion: bool = True,
    num_query_variations: int = 3,
    use_bge_reranker: bool = True,
    use_context_expansion: bool = True,
    use_local_cache: bool = False,
    # Phase 1 improvements
    use_hyde: bool = False,
    use_lost_in_middle: bool = True,
    lost_in_middle_strategy: str = "alternating",
    # Phase 2 improvements
    use_hybrid_search: str = "auto",  # "auto", "always", "never"
    hybrid_dense_weight: float = 0.7,
    use_semantic_cache: bool = True,
    cache_dir: Optional[str] = None,
    compute_metrics: bool = False,
    # Phase 3 improvements
    use_query_understanding: bool = True,
    use_answer_grounding: bool = True,
    grounding_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    RAG sur une seule collection.

    - Si call_llm=True : retrieval + appel LLM
    - Si call_llm=False : retrieval uniquement (retourne context_str & sources, answer vide)
    - Si use_feedback_reranking=True et feedback_store fourni : applique le re-ranking
      basé sur les feedbacks utilisateurs
    - Si use_query_expansion=True : génère des variations de la question et fusionne les résultats
    - Si use_bge_reranker=True : applique le reranking BGE après la recherche initiale
    - Si use_context_expansion=True : enrichit les chunks avec contexte voisin et références
    - Si use_local_cache=True : utilise le cache local pour des lectures plus rapides

    Phase 1 improvements (v1.1):
    - Si use_hyde=True : utilise HyDE (Hypothetical Document Embeddings) pour améliorer le recall
    - Si use_lost_in_middle=True : réordonne les sources pour éviter l'effet "lost in the middle"
    - lost_in_middle_strategy : "alternating", "bookends", ou "reverse_middle"

    Phase 2 improvements (v1.2):
    - use_hybrid_search : "auto" (actif si >1000 chunks), "always", "never"
    - hybrid_dense_weight : poids de la recherche dense vs BM25 (0.7 = 70% dense)
    - use_semantic_cache : cache les réponses pour requêtes similaires
    - cache_dir : répertoire de persistance du cache
    - compute_metrics : calcule les métriques RAGAS de qualité

    Phase 3 improvements (v1.3):
    - use_query_understanding : analyse l'intention de la question pour adapter la recherche
    - use_answer_grounding : vérifie que la réponse est supportée par le contexte
    - grounding_threshold : seuil minimum de grounding (0-1, défaut 0.3)
    """
    _log = log or logger

    question = (question or "").strip()
    if not question:
        raise ValueError("Question vide")

    # ========== PHASE 3: QUERY UNDERSTANDING ==========
    query_analysis = None
    original_top_k = top_k

    if use_query_understanding and QUERY_UNDERSTANDING_AVAILABLE:
        try:
            query_analysis = analyze_query(question)
            _log.info(
                f"[RAG] 🔍 Query Analysis: intent={query_analysis.intent.value} "
                f"({query_analysis.intent_confidence:.0%}), "
                f"domain={query_analysis.domain.value}, "
                f"complexity={query_analysis.complexity.value}"
            )

            # Adapter top_k selon la complexité et l'intention
            top_k = get_adaptive_top_k(original_top_k, query_analysis)
            if top_k != original_top_k:
                _log.info(f"[RAG] Adaptive top_k: {original_top_k} -> {top_k}")

            # Log des entités détectées
            if query_analysis.entities:
                _log.debug(f"[RAG] Detected entities: {query_analysis.entities}")

            # Calculer les poids optimaux pour la recherche hybride selon l'intent
            # Sauf si l'utilisateur a explicitement spécifié une valeur différente du défaut
            if hybrid_dense_weight == 0.7:  # Valeur par défaut
                intent_dense, intent_sparse = get_hybrid_search_weights(query_analysis)
                hybrid_dense_weight = intent_dense
                _log.info(
                    f"[RAG] Intent-aware hybrid weights: "
                    f"dense={intent_dense:.0%}, sparse={intent_sparse:.0%}"
                )

        except Exception as e:
            _log.warning(f"[RAG] Query understanding failed: {e}")

    # ========== PHASE 2: SEMANTIC CACHE CHECK ==========
    semantic_cache = None
    question_embedding = None

    if use_semantic_cache and SEMANTIC_CACHE_AVAILABLE and call_llm:
        try:
            semantic_cache = get_semantic_cache(cache_dir=cache_dir, log=_log)

            # Pré-calculer l'embedding de la question pour le cache
            http_client = create_http_client()
            emb_client = DirectOpenAIEmbeddings(
                model=EMBED_MODEL,
                api_key=SNOWFLAKE_API_KEY,
                base_url=SNOWFLAKE_API_BASE,
                http_client=http_client,
                role_prefix=True,
                logger=_log,
            )
            question_embedding = embed_in_batches(
                texts=[question],
                role="query",
                batch_size=1,
                emb_client=emb_client,
                log=_log,
                dry_run=False,
            )[0].tolist()

            # Vérifier le cache
            cached_result = semantic_cache.get(
                question=question,
                question_embedding=question_embedding,
                collection_name=collection_name,
            )

            if cached_result:
                _log.info(f"[RAG] 🎯 Cache HIT (similarity={cached_result.get('cache_similarity', 0):.2%})")
                return {
                    "answer": cached_result["answer"],
                    "context_str": cached_result.get("context_str", ""),
                    "sources": cached_result.get("sources", []),
                    "cached": True,
                    "cache_similarity": cached_result.get("cache_similarity", 0),
                }
        except Exception as e:
            _log.warning(f"[RAG] Cache check failed: {e}")

    _log.info(
        f"[RAG] (single) db={db_path} | collection={collection_name} | top_k={top_k} | cache={use_local_cache}"
    )

    # 1) FAISS store + collection (avec cache local si activé)
    store = build_store(db_path, use_local_cache=use_local_cache)
    collection = store.get_collection(name=collection_name)

    # Capturer l'état du cache pour l'avertissement
    cache_outdated = getattr(collection, 'cache_outdated', False)
    using_cache = getattr(collection, 'using_cache', False)
    if cache_outdated:
        _log.warning(f"[RAG] ⚠️ Cache obsolète pour {collection_name} - utilisation réseau")

    # 2) Client embeddings Snowflake
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )

    # 3) Recherche avec différentes stratégies

    # Fonction d'embedding pour les recherches
    def embed_query(q: str):
        return embed_in_batches(
            texts=[q],
            role="query",
            batch_size=1,
            emb_client=emb_client,
            log=_log,
            dry_run=False,
        )[0]

    # ========== PHASE 2: HYBRID SEARCH (AUTO) ==========
    # Déterminer si on utilise la recherche hybride
    collection_size = collection.count() if hasattr(collection, 'count') else 0
    should_hybrid = False

    if HYBRID_SEARCH_AVAILABLE:
        if use_hybrid_search == "always":
            should_hybrid = True
        elif use_hybrid_search == "auto":
            should_hybrid = should_use_hybrid_search(collection_size, log=_log)
        # "never" -> should_hybrid reste False

    if should_hybrid:
        _log.info(f"[RAG] 🔀 Mode Hybrid Search activé (collection: {collection_size} chunks)")
        try:
            # Construire ou charger l'index BM25
            docs, doc_ids = get_collection_documents(collection)
            bm25_index = build_or_load_bm25_index(
                db_path=db_path,
                collection_name=collection_name,
                documents=docs,
                doc_ids=doc_ids,
                log=_log
            )

            if bm25_index:
                raw = hybrid_search(
                    query=question,
                    collection=collection,
                    bm25_index=bm25_index,
                    embed_func=embed_query,
                    top_k=top_k,
                    dense_weight=hybrid_dense_weight,
                    sparse_weight=1.0 - hybrid_dense_weight,
                    log=_log,
                )
                _log.info("[RAG] ✅ Hybrid search completed")
            else:
                _log.warning("[RAG] BM25 index unavailable, falling back to dense search")
                should_hybrid = False
        except Exception as e:
            _log.warning(f"[RAG] Hybrid search failed: {e}, falling back to dense search")
            should_hybrid = False

    # Option A: HyDE (Hypothetical Document Embeddings) - si pas de hybrid
    if not should_hybrid and use_hyde and HYDE_AVAILABLE:
        _log.info("[RAG] 🧪 Mode HyDE activé (Hypothetical Document Embeddings)")
        raw = search_with_hyde(
            question=question,
            collection=collection,
            embed_func=embed_query,
            http_client=http_client,
            api_key=DALLEM_API_KEY,
            api_base=DALLEM_API_BASE,
            model=LLM_MODEL,
            top_k=top_k,
            use_both=True,  # Combine HyDE + question originale
            log=_log,
        )
        _log.info("[RAG] ✅ HyDE search completed")

    # Option B: Query Expansion - si pas de hybrid ni HyDE
    elif not should_hybrid and use_query_expansion and ADVANCED_SEARCH_AVAILABLE:
        _log.info(f"[RAG] 🔄 Mode Query Expansion activé ({num_query_variations} variations)")

        # Générer les variations de la question
        queries = expand_query_with_llm(
            question=question,
            http_client=http_client,
            api_key=DALLEM_API_KEY,
            api_base=DALLEM_API_BASE,
            model=LLM_MODEL,
            num_variations=num_query_variations,
            log=_log,
        )

        # Recherche multi-query
        raw = run_multi_query_search(
            collection=collection,
            queries=queries,
            embed_func=embed_query,
            top_k=top_k,
            log=_log,
        )
        _log.info(f"[RAG] ✅ Multi-query search completed ({len(queries)} queries)")

    # Option C: Mode standard (une seule requête) - si aucune autre option
    elif not should_hybrid:
        q_emb = embed_query(question)

        _log.info("[RAG] Querying FAISS index...")
        raw = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        _log.info(f"[RAG] ✅ Query successful")

    # Accès sécurisé aux résultats (évite IndexError si liste vide)
    _docs = raw.get("documents", [[]])
    _metas = raw.get("metadatas", [[]])
    _dists = raw.get("distances", [[]])
    docs = _docs[0] if _docs and _docs[0] else []
    metas = _metas[0] if _metas and _metas[0] else []
    dists = _dists[0] if _dists and _dists[0] else []

    # ========== PHASE 3: SECTION_ID BOOSTING ==========
    # Boost chunks matching EASA references in the query
    if query_analysis and query_analysis.entities:
        docs, metas, dists = _boost_by_section_id(
            docs=docs,
            metas=metas,
            dists=dists,
            entities=query_analysis.entities,
            boost_factor=0.4,  # Boost significatif pour les matches exacts
            log=_log,
        )

    if not docs:
        _log.warning("[RAG] Aucun document retourné par FAISS")
        return {
            "answer": (
                "Aucun contexte trouvé dans la base pour répondre à la question."
                if call_llm
                else ""
            ),
            "context_str": "",
            "raw_results": raw,
            "sources": [],
            "cache_outdated": cache_outdated,
            "using_cache": using_cache,
        }

    # 5) Construction du contexte + liste des sources
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            meta = {}

        source_file = meta.get("source_file", "unknown")
        chunk_id = meta.get("chunk_id", "?")
        section_id = meta.get("section_id") or ""
        section_kind = meta.get("section_kind") or ""
        section_title = meta.get("section_title") or ""
        path = meta.get("path") or ""

        # ID de référence pour matcher avec un CSV global
        ref_id = f"{db_path}|{collection_name}|{path or chunk_id}"

        # Build source dict for helper function
        src_for_header = {"section_id": section_id, "section_title": section_title}
        section_info = _format_section_info(src_for_header)

        header = (
            f"[source={source_file}{section_info}, chunk={chunk_id}, "
            f"dist={float(dist):.3f}]"
        )
        context_blocks.append(f"{header}\n{doc}")

        sources.append(
            {
                "collection": collection_name,
                "source_file": source_file,
                "path": path,
                "chunk_id": chunk_id,
                "distance": float(dist),
                "score": max(0, 1.0 - float(dist) / 2.0),  # L2 normalisé: [0,2] -> [1,0]
                "section_id": section_id,
                "section_kind": section_kind,
                "section_title": section_title,
                "ref_id": ref_id,
                "text": doc,
            }
        )

    # ========== RE-RANKING BASÉ SUR LES FEEDBACKS ==========
    if use_feedback_reranking and feedback_store and FEEDBACK_AVAILABLE:
        _log.info("[RAG] Applying feedback-based re-ranking...")
        try:
            # Extraire le nom de la base depuis db_path
            base_name = os.path.basename(db_path)

            # Appliquer le re-ranking
            sources = feedback_store.compute_reranking_factors(
                sources=sources,
                base_name=base_name,
                collection_name=collection_name,
                question=question,
                alpha=feedback_alpha
            )

            # Reconstruire le contexte avec l'ordre des sources re-rankées
            context_blocks = []
            for src in sources:
                section_info = _format_section_info(src)
                header = (
                    f"[source={src['source_file']}{section_info}, chunk={src['chunk_id']}, "
                    f"score={src['score']:.3f}, boost={src.get('feedback_boost', 0):.3f}]"
                )
                context_blocks.append(f"{header}\n{src['text']}")

            _log.info(f"[RAG] ✅ Re-ranking applied (alpha={feedback_alpha})")
        except Exception as e:
            _log.warning(f"[RAG] Re-ranking failed, using original order: {e}")

    # ========== RE-RANKING BGE (cross-encoder) ==========
    if use_bge_reranker and BGE_RERANKER_AVAILABLE:
        _log.info("[RAG] 🔄 Applying BGE Reranker...")
        try:
            sources = apply_reranking_to_sources(
                query=question,
                sources=sources,
                top_k=top_k,
                http_client=http_client,
                log=_log
            )

            # Reconstruire le contexte avec l'ordre des sources reranked
            context_blocks = []
            for src in sources:
                section_info = _format_section_info(src)
                header = (
                    f"[source={src['source_file']}{section_info}, chunk={src['chunk_id']}, "
                    f"rerank_score={src.get('rerank_score', src.get('score', 0)):.3f}]"
                )
                context_blocks.append(f"{header}\n{src['text']}")

            _log.info(f"[RAG] ✅ BGE Reranking applied")
        except Exception as e:
            _log.warning(f"[RAG] BGE Reranking failed, using previous order: {e}")

    # ========== FILTRAGE PAR MOTS-CLÉS ==========
    if KEYWORD_FILTER_AVAILABLE:
        _log.info("[RAG] 🔑 Applying keyword filtering...")
        try:
            # Extraire les mots-clés de la question
            keywords = extract_keywords(question, min_length=3, log=_log)

            if keywords:
                # Filtrer les sources qui contiennent au moins 1 mot-clé
                sources = filter_sources_by_keywords(
                    sources=sources,
                    keywords=keywords,
                    min_matches=1,
                    log=_log
                )

                # Reconstruire le contexte avec les sources filtrées
                context_blocks = []
                for src in sources:
                    section_info = _format_section_info(src)
                    kw_info = f", keywords={src.get('keyword_count', 0)}" if 'keyword_count' in src else ""
                    header = (
                        f"[source={src['source_file']}{section_info}, chunk={src['chunk_id']}, "
                        f"score={src.get('score', 0):.3f}{kw_info}]"
                    )
                    context_blocks.append(f"{header}\n{src['text']}")

                _log.info(f"[RAG] ✅ Keyword filtering applied ({len(sources)} sources)")
        except Exception as e:
            _log.warning(f"[RAG] Keyword filtering failed: {e}")

    # ========== CONTEXT EXPANSION (Cross-References + Neighbors) ==========
    if use_context_expansion and CONTEXT_EXPANSION_AVAILABLE:
        _log.info("[RAG] 🔗 Applying context expansion...")
        try:
            expanded_context_blocks = []
            seen_chunks = set()  # Pour éviter les doublons

            for src in sources[:15]:  # Limiter aux 15 premiers pour performance
                chunk_id = src.get("chunk_id", "")
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)

                # Extraire les références croisées du chunk
                refs = extract_cross_references(src.get("text", ""))
                ref_ids = [r["ref_id"] for r in refs if r["ref_type"] in ("CS", "AMC", "GM")]

                # Construire le bloc de contexte principal
                section_info = _format_section_info(src)
                main_header = (
                    f"[source={src['source_file']}{section_info}, chunk={chunk_id}, "
                    f"score={src.get('score', 0):.3f}]"
                )
                main_block = f"{main_header}\n{src['text']}"

                # Ajouter les références détectées aux métadonnées
                if ref_ids:
                    src["references_to"] = ref_ids[:5]  # Max 5 références
                    _log.debug(f"[RAG] Chunk {chunk_id} references: {ref_ids[:5]}")

                expanded_context_blocks.append(main_block)

            # Chercher les chunks référencés dans les autres sources
            referenced_chunks = []
            all_refs = set()

            for src in sources[:15]:
                refs_to = src.get("references_to", [])
                for ref in refs_to:
                    ref_normalized = ref.upper().replace(" ", "").replace("-", "")
                    all_refs.add(ref_normalized)

            # Trouver les chunks qui correspondent aux références
            if all_refs:
                for src in sources:
                    section_id = src.get("section_id", "")
                    if section_id:
                        section_normalized = section_id.upper().replace(" ", "").replace("-", "")
                        if section_normalized in all_refs and src.get("chunk_id") not in seen_chunks:
                            referenced_chunks.append(src)
                            seen_chunks.add(src.get("chunk_id"))

                # Ajouter les chunks référencés au contexte
                if referenced_chunks:
                    _log.info(f"[RAG] Adding {len(referenced_chunks)} referenced chunk(s) to context")
                    for ref_src in referenced_chunks[:5]:  # Max 5 chunks référencés
                        ref_header = (
                            f"[REFERENCED: source={ref_src['source_file']}, "
                            f"section={ref_src.get('section_id', '?')}]"
                        )
                        expanded_context_blocks.append(f"{ref_header}\n{ref_src['text']}")

            context_blocks = expanded_context_blocks
            _log.info(f"[RAG] ✅ Context expansion applied ({len(context_blocks)} blocks)")

        except Exception as e:
            _log.warning(f"[RAG] Context expansion failed: {e}")

    # ========== LOST IN THE MIDDLE MITIGATION ==========
    # Apply reordering for 2+ sources (was > 3, but even 2 sources can benefit)
    if use_lost_in_middle and LOST_IN_MIDDLE_AVAILABLE and len(sources) > 1:
        _log.info(f"[RAG] 🔀 Applying Lost in Middle reordering (strategy={lost_in_middle_strategy})...")
        try:
            # Réordonner les sources
            sources = reorder_for_lost_in_middle(sources, strategy=lost_in_middle_strategy)

            # Reconstruire les context_blocks dans le nouvel ordre
            context_blocks = []
            for src in sources:
                section_info = _format_section_info(src)
                header = (
                    f"[source={src['source_file']}{section_info}, chunk={src['chunk_id']}, "
                    f"score={src.get('score', 0):.3f}]"
                )
                context_blocks.append(f"{header}\n{src['text']}")

            _log.info(f"[RAG] ✅ Lost in Middle reordering applied")
        except Exception as e:
            _log.warning(f"[RAG] Lost in Middle reordering failed: {e}")

    full_context = "\n\n".join(context_blocks)

    if not call_llm:
        # Mode "retrieval only"
        return {
            "answer": "",
            "context_str": full_context,
            "raw_results": raw,
            "sources": sources,
            "cache_outdated": cache_outdated,
            "using_cache": using_cache,
        }

    # 6) Appel LLM DALLEM
    answer = call_dallem_chat(
        http_client=http_client,
        question=question,
        context=full_context,
        log=_log,
    )

    # ========== PHASE 3: ANSWER GROUNDING (hallucination detection) ==========
    grounding_report = None
    grounding_warning = None

    if use_answer_grounding and ANSWER_GROUNDING_AVAILABLE and answer:
        try:
            grounding_report = analyze_grounding(
                answer=answer,
                context=full_context,
                min_score=grounding_threshold,
            )

            _log.info(
                f"[RAG] 🔎 Grounding: score={grounding_report.overall_score:.0%}, "
                f"risk={grounding_report.hallucination_risk}, "
                f"grounded={grounding_report.grounded_sentences}/{grounding_report.total_sentences}"
            )

            # Générer un avertissement si risque élevé
            grounding_warning = get_grounding_warning(grounding_report)
            if grounding_warning:
                _log.warning(f"[RAG] ⚠️ {grounding_warning}")

        except Exception as e:
            _log.warning(f"[RAG] Grounding analysis failed: {e}")

    # ========== PHASE 2: CACHE RESPONSE ==========
    if semantic_cache and question_embedding:
        try:
            semantic_cache.put(
                question=question,
                question_embedding=question_embedding,
                collection_name=collection_name,
                answer=answer,
                sources=sources,
                context_str=full_context,
            )
            _log.debug("[RAG] Response cached for future similar queries")
        except Exception as e:
            _log.warning(f"[RAG] Failed to cache response: {e}")

    # ========== PHASE 2: COMPUTE METRICS ==========
    metrics_result = None
    if compute_metrics and RAG_METRICS_AVAILABLE:
        try:
            evaluator = RAGEvaluator(log=_log)
            metrics = evaluator.evaluate(
                question=question,
                answer=answer,
                context=full_context,
                sources=sources,
            )
            metrics_result = metrics.to_dict()
            _log.info(f"[RAG] 📊 Metrics: overall={metrics.overall_score:.0%}, "
                     f"faithfulness={metrics.faithfulness:.0%}, relevance={metrics.answer_relevance:.0%}")
        except Exception as e:
            _log.warning(f"[RAG] Metrics computation failed: {e}")

    result = {
        "answer": answer,
        "context_str": full_context,
        "raw_results": raw,
        "sources": sources,
        "cache_outdated": cache_outdated,
        "using_cache": using_cache,
        "cached": False,
    }

    if metrics_result:
        result["metrics"] = metrics_result

    # Phase 3: Add grounding results
    if grounding_report:
        result["grounding"] = {
            "score": grounding_report.overall_score,
            "is_grounded": grounding_report.is_grounded,
            "risk": grounding_report.hallucination_risk,
            "grounded_sentences": grounding_report.grounded_sentences,
            "total_sentences": grounding_report.total_sentences,
            "flagged_claims": grounding_report.flagged_claims[:5],  # Top 5 flagged
        }
        if grounding_warning:
            result["grounding_warning"] = grounding_warning

    # Phase 3: Add query analysis
    if query_analysis:
        result["query_analysis"] = {
            "intent": query_analysis.intent.value,
            "intent_confidence": query_analysis.intent_confidence,
            "domain": query_analysis.domain.value,
            "complexity": query_analysis.complexity.value,
            "entities": query_analysis.entities,
        }

    return result


# =====================================================================
#  RAG : UNE COLLECTION OU TOUTES LES COLLECTIONS (ALL)
# =====================================================================

def run_rag_query(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    synthesize_all: bool = False,
    log=None,
    feedback_store: Optional["FeedbackStore"] = None,
    use_feedback_reranking: bool = False,
    feedback_alpha: float = 0.3,
    use_query_expansion: bool = True,
    num_query_variations: int = 3,
    use_bge_reranker: bool = True,
    use_context_expansion: bool = True,
    use_local_cache: bool = True,  # Automatique: utilise le cache local si disponible
    # Phase 1 improvements
    use_hyde: bool = False,
    use_lost_in_middle: bool = True,
    lost_in_middle_strategy: str = "alternating",
    # Phase 2 improvements
    use_hybrid_search: str = "auto",  # "auto", "always", "never"
    hybrid_dense_weight: float = 0.7,
    use_semantic_cache: bool = True,
    cache_dir: Optional[str] = None,
    compute_metrics: bool = False,
    # Phase 3 improvements
    use_query_understanding: bool = True,
    use_answer_grounding: bool = True,
    grounding_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    RAG "haut niveau" :

    - Si collection_name != "ALL" :
        → requête sur UNE collection (cf. _run_rag_query_single_collection)

    - Si collection_name == "ALL" :
        - synthesize_all = True :
             → retrieval sur toutes les collections,
               concaténé dans un gros contexte global,
               puis un SEUL appel LLM DALLEM.
        - synthesize_all = False :
             → par sécurité, on se contente de lever une erreur ou
               de déléguer à l'appelant (dans l'interface Streamlit, ce cas
               est géré côté interface, pas ici).

    Options de recherche avancée :
    - use_query_expansion : génère des variations de la question pour améliorer le recall
    - num_query_variations : nombre de variations à générer (défaut: 3)
    - use_bge_reranker : applique le reranking BGE après la recherche (défaut: True)
    - use_context_expansion : enrichit les résultats avec contexte et références (défaut: True)

    Options de re-ranking basé sur les feedbacks :
    - feedback_store : instance de FeedbackStore pour accéder aux feedbacks
    - use_feedback_reranking : activer le re-ranking basé sur les feedbacks
    - feedback_alpha : facteur d'influence (0-1, défaut 0.3)

    Options de performance :
    - use_local_cache : utilise le cache local pour des lectures plus rapides (défaut: False)

    Phase 1 improvements (v1.1):
    - use_hyde : utilise HyDE (Hypothetical Document Embeddings) - améliore le recall de 10-20%
    - use_lost_in_middle : réordonne les sources pour éviter l'effet "lost in the middle"
    - lost_in_middle_strategy : "alternating" (défaut), "bookends", ou "reverse_middle"

    Phase 2 improvements (v1.2):
    - use_hybrid_search : "auto" (actif si >1000 chunks), "always", "never"
    - hybrid_dense_weight : poids de la recherche dense vs BM25 (0.7 = 70% dense)
    - use_semantic_cache : cache les réponses pour requêtes similaires (défaut: True)
    - cache_dir : répertoire de persistance du cache
    - compute_metrics : calcule les métriques RAGAS de qualité

    Phase 3 improvements (v1.3):
    - use_query_understanding : analyse l'intention de la question (définition, procédure, etc.)
    - use_answer_grounding : vérifie que la réponse est supportée par le contexte (détection d'hallucinations)
    - grounding_threshold : seuil minimum de grounding (0-1, défaut 0.3)
    """
    _log = log or logger

    # Cas "ALL" (utilisé par streamlit_RAG quand synthesize_all=True)
    if collection_name == "ALL":
        _log.info(
            f"[RAG] Mode ALL collections | db={db_path} | synthesize_all={synthesize_all} | cache={use_local_cache}"
        )
        store = build_store(db_path, use_local_cache=use_local_cache)
        collections = store.list_collections()  # FAISS retourne directement une liste de noms

        if not collections:
            return {
                "answer": "Aucune collection disponible dans cette base FAISS.",
                "context_str": "",
                "raw_results": {},
                "sources": [],
                "cache_outdated": False,
                "using_cache": False,
            }

        if not synthesize_all:
            # Dans l'interface Streamlit, ce cas est géré côté interface (boucle sur les collections),
            # donc ici on renvoie une erreur explicite pour éviter toute ambiguïté.
            raise ValueError(
                "run_rag_query(collection_name='ALL') appelé avec synthesize_all=False. "
                "Ce mode doit être géré côté interface (une requête par collection)."
            )

        # ---- Mode synthèse globale : un seul appel LLM avec le contexte concaténé ----
        all_sources: List[Dict[str, Any]] = []
        all_context_blocks: List[str] = []
        any_cache_outdated = False  # Track si au moins un cache est obsolète

        for col_name in collections:  # FAISS retourne directement les noms (strings)
            _log.info(f"[RAG-ALL-SYNTH] Retrieval sur collection '{col_name}'")

            try:
                res = _run_rag_query_single_collection(
                    db_path=db_path,
                    collection_name=col_name,
                    question=question,
                    top_k=top_k,
                    call_llm=False,  # pas d'appel LLM ici
                    log=_log,
                    feedback_store=feedback_store,
                    use_feedback_reranking=use_feedback_reranking,
                    feedback_alpha=feedback_alpha,
                    use_query_expansion=use_query_expansion,
                    num_query_variations=num_query_variations,
                    use_bge_reranker=use_bge_reranker,
                    use_context_expansion=use_context_expansion,
                    use_local_cache=use_local_cache,
                    # Phase 1 improvements
                    use_hyde=use_hyde,
                    use_lost_in_middle=use_lost_in_middle,
                    lost_in_middle_strategy=lost_in_middle_strategy,
                    # Phase 2 improvements
                    use_hybrid_search=use_hybrid_search,
                    hybrid_dense_weight=hybrid_dense_weight,
                    use_semantic_cache=False,  # Pas de cache pour ALL (synthèse globale)
                    cache_dir=cache_dir,
                    compute_metrics=False,  # Pas de métriques pour retrieval-only
                    # Phase 3 improvements
                    use_query_understanding=use_query_understanding,
                    use_answer_grounding=False,  # Pas de grounding pour retrieval-only
                    grounding_threshold=grounding_threshold,
                )
            except Exception as e:
                _log.error(
                    f"[RAG-ALL-SYNTH] Erreur pendant la récupération sur '{col_name}' : {e}"
                )
                continue

            context_str = res.get("context_str", "")
            sources = res.get("sources", [])

            # Capturer l'état du cache
            if res.get("cache_outdated", False):
                any_cache_outdated = True

            if context_str:
                all_context_blocks.append(
                    f"=== CONTEXTE {col_name} ===\n{context_str}".strip()
                )

            for s in sources:
                if "collection" not in s:
                    s["collection"] = col_name
                all_sources.append(s)

        global_context = "\n\n".join(all_context_blocks)

        if not global_context.strip():
            return {
                "answer": (
                    "Aucun contexte trouvé dans la base pour répondre à la question (mode ALL)."
                ),
                "context_str": "",
                "raw_results": {},
                "sources": [],
                "cache_outdated": any_cache_outdated,
                "using_cache": False,
            }

        http_client = create_http_client()
        answer = call_dallem_chat(
            http_client=http_client,
            question=(question or "").strip(),
            context=global_context,
            log=_log,
        )

        return {
            "answer": answer,
            "context_str": global_context,
            "raw_results": {},
            "sources": all_sources,
            "cache_outdated": any_cache_outdated,
            "using_cache": False,  # En mode ALL, on agrège donc pas de "using_cache" unique
        }

    # Cas normal : une seule collection
    return _run_rag_query_single_collection(
        db_path=db_path,
        collection_name=collection_name,
        question=question,
        top_k=top_k,
        call_llm=True,
        log=_log,
        feedback_store=feedback_store,
        use_feedback_reranking=use_feedback_reranking,
        feedback_alpha=feedback_alpha,
        use_query_expansion=use_query_expansion,
        num_query_variations=num_query_variations,
        use_bge_reranker=use_bge_reranker,
        use_context_expansion=use_context_expansion,
        use_local_cache=use_local_cache,
        # Phase 1 improvements
        use_hyde=use_hyde,
        use_lost_in_middle=use_lost_in_middle,
        lost_in_middle_strategy=lost_in_middle_strategy,
        # Phase 2 improvements
        use_hybrid_search=use_hybrid_search,
        hybrid_dense_weight=hybrid_dense_weight,
        use_semantic_cache=use_semantic_cache,
        cache_dir=cache_dir,
        compute_metrics=compute_metrics,
        # Phase 3 improvements
        use_query_understanding=use_query_understanding,
        use_answer_grounding=use_answer_grounding,
        grounding_threshold=grounding_threshold,
    )


# =====================================================================
#  RAG MULTI-COLLECTION (FEDERATED SEARCH)
# =====================================================================

def run_multi_collection_rag_query(
    db_path: str,
    collection_names: List[str],
    question: str,
    top_k: int = 30,
    call_llm: bool = True,
    log=None,
    use_hybrid_search: bool = True,
    use_bge_reranker: bool = True,
    use_lost_in_middle: bool = True,
    lost_in_middle_strategy: str = "alternating",
    use_query_understanding: bool = True,
    use_answer_grounding: bool = True,
    grounding_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    RAG sur plusieurs collections simultanément (Federated Search).

    Interroge plusieurs collections en parallèle et fusionne les résultats
    avec indication de la provenance (collection_name) pour chaque source.

    Args:
        db_path: Chemin de la base de données FAISS
        collection_names: Liste des noms de collections à interroger
        question: Question de l'utilisateur
        top_k: Nombre de résultats par collection avant fusion
        call_llm: Si True, appelle le LLM pour générer une réponse
        log: Logger optionnel
        use_hybrid_search: Utiliser la recherche hybride (BM25 + dense)
        use_bge_reranker: Appliquer le reranking BGE
        use_lost_in_middle: Réordonner pour éviter "lost in the middle"
        lost_in_middle_strategy: Stratégie de réordonnancement
        use_query_understanding: Analyser l'intention de la question
        use_answer_grounding: Vérifier le grounding de la réponse
        grounding_threshold: Seuil de grounding

    Returns:
        Dict avec answer, context_str, sources (avec collection_name)
    """
    _log = log or logger

    if not MULTI_COLLECTION_AVAILABLE:
        raise ImportError(
            "Multi-collection search not available. "
            "Please ensure multi_collection_search.py is present."
        )

    question = (question or "").strip()
    if not question:
        raise ValueError("Question vide")

    if not collection_names:
        raise ValueError("Aucune collection spécifiée")

    _log.info(
        f"[RAG-MULTI] Starting federated search on {len(collection_names)} collections: "
        f"{collection_names}"
    )

    # ========== QUERY UNDERSTANDING ==========
    query_analysis = None
    if use_query_understanding and QUERY_UNDERSTANDING_AVAILABLE:
        try:
            query_analysis = analyze_query(question)
            _log.info(
                f"[RAG-MULTI] Query Analysis: intent={query_analysis.intent.value}, "
                f"domain={query_analysis.domain.value}"
            )
        except Exception as e:
            _log.warning(f"[RAG-MULTI] Query understanding failed: {e}")

    # ========== SETUP EMBEDDING CLIENT ==========
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )

    def embed_query(q: str):
        return embed_in_batches(
            texts=[q],
            role="query",
            batch_size=1,
            emb_client=emb_client,
            log=_log,
            dry_run=False,
        )[0]

    # ========== MULTI-COLLECTION SEARCH ==========
    try:
        raw_results, sources = multi_collection_query(
            query=question,
            db_path=db_path,
            collection_names=collection_names,
            embed_func=embed_query,
            faiss_store_module=faiss_store_module,
            top_k=top_k,
            use_hybrid=use_hybrid_search,
            use_reranking=use_bge_reranker,
            http_client=http_client,
            log=_log,
        )
    except Exception as e:
        _log.error(f"[RAG-MULTI] Multi-collection search failed: {e}")
        return {
            "answer": f"Erreur lors de la recherche multi-collection: {e}",
            "context_str": "",
            "sources": [],
            "collections_searched": collection_names,
        }

    if not sources:
        _log.warning("[RAG-MULTI] No results from any collection")
        return {
            "answer": "Aucun contexte trouvé dans les collections pour répondre à la question.",
            "context_str": "",
            "sources": [],
            "collections_searched": collection_names,
        }

    # ========== SECTION_ID BOOSTING ==========
    if query_analysis and query_analysis.entities:
        # Extraire docs, metas, dists des sources pour le boosting
        docs = [s.get("text", "") for s in sources]
        metas = [s for s in sources]  # sources contient déjà les métadonnées
        dists = [s.get("distance", 1.0) for s in sources]

        docs, metas, dists = _boost_by_section_id(
            docs=docs,
            metas=metas,
            dists=dists,
            entities=query_analysis.entities,
            boost_factor=0.4,
            log=_log,
        )

        # Reconstruire sources avec le nouvel ordre
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            src = meta.copy() if isinstance(meta, dict) else {}
            src["text"] = doc
            src["distance"] = dist
            src["rank"] = i + 1
            sources.append(src)

    # ========== LOST IN THE MIDDLE ==========
    if use_lost_in_middle and LOST_IN_MIDDLE_AVAILABLE and len(sources) > 1:
        _log.info(f"[RAG-MULTI] Applying Lost in Middle ({lost_in_middle_strategy})")
        try:
            sources = reorder_for_lost_in_middle(sources, strategy=lost_in_middle_strategy)
        except Exception as e:
            _log.warning(f"[RAG-MULTI] Lost in middle failed: {e}")

    # ========== BUILD CONTEXT ==========
    context_str = format_sources_with_provenance(sources)

    if not call_llm:
        return {
            "answer": "",
            "context_str": context_str,
            "sources": sources,
            "collections_searched": collection_names,
            "raw_results": raw_results,
        }

    # ========== CALL LLM ==========
    _log.info("[RAG-MULTI] Calling LLM for answer generation...")
    answer = call_dallem_chat(
        http_client=http_client,
        question=question,
        context=context_str,
        log=_log,
    )

    # ========== ANSWER GROUNDING ==========
    grounding_report = None
    grounding_warning = None

    if use_answer_grounding and ANSWER_GROUNDING_AVAILABLE and answer:
        try:
            grounding_report = analyze_grounding(
                answer=answer,
                context=context_str,
                min_score=grounding_threshold,
            )
            _log.info(
                f"[RAG-MULTI] Grounding: score={grounding_report.overall_score:.0%}, "
                f"risk={grounding_report.hallucination_risk}"
            )
            grounding_warning = get_grounding_warning(grounding_report)
        except Exception as e:
            _log.warning(f"[RAG-MULTI] Grounding failed: {e}")

    # ========== BUILD RESULT ==========
    result = {
        "answer": answer,
        "context_str": context_str,
        "sources": sources,
        "collections_searched": collection_names,
        "raw_results": raw_results,
    }

    if query_analysis:
        result["query_analysis"] = {
            "intent": query_analysis.intent.value,
            "intent_confidence": query_analysis.intent_confidence,
            "domain": query_analysis.domain.value,
            "complexity": query_analysis.complexity.value,
            "entities": query_analysis.entities,
        }

    if grounding_report:
        result["grounding"] = {
            "score": grounding_report.overall_score,
            "risk": grounding_report.hallucination_risk,
            "grounded_sentences": grounding_report.grounded_sentences,
            "total_sentences": grounding_report.total_sentences,
        }
        if grounding_warning:
            result["grounding_warning"] = grounding_warning

    # Stats de provenance
    collection_stats = {}
    for src in sources:
        coll = src.get("collection_name", "unknown")
        collection_stats[coll] = collection_stats.get(coll, 0) + 1
    result["collection_stats"] = collection_stats

    _log.info(
        f"[RAG-MULTI] ✅ Complete. Sources by collection: {collection_stats}"
    )

    return result
