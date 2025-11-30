"""
Multi-Collection Search Module - Requêtes Inter-Bases

Permet d'interroger plusieurs collections FAISS simultanément
et de fusionner les résultats avec indication de provenance.

v1.0 - Support requêtes inter-bases
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Nombre maximum de workers pour recherche parallèle
MAX_SEARCH_WORKERS = 4

# Constante RRF pour fusion des scores
RRF_K = 60

# Top-K par collection avant fusion
DEFAULT_TOP_K_PER_COLLECTION = 30


# =============================================================================
#  MULTI-COLLECTION SEARCHER
# =============================================================================

class MultiCollectionSearcher:
    """
    Recherche fédérée sur plusieurs collections FAISS.

    Fonctionnalités:
    - Recherche parallèle sur N collections
    - Fusion RRF (Reciprocal Rank Fusion) avec poids égaux
    - Indication de la provenance (collection_name) pour chaque source
    - Support recherche hybride (dense + BM25) multi-collection
    """

    def __init__(
        self,
        collections: Dict[str, Any],
        embed_func,
        bm25_indices: Optional[Dict[str, Any]] = None,
        log: Optional[logging.Logger] = None
    ):
        """
        Args:
            collections: Dict {collection_name: collection_object}
            embed_func: Fonction d'embedding pour les requêtes
            bm25_indices: Dict optionnel {collection_name: BM25Index} pour recherche hybride
            log: Logger optionnel
        """
        self.collections = collections
        self.embed_func = embed_func
        self.bm25_indices = bm25_indices or {}
        self._log = log or logger

        self._log.info(
            f"[MULTI-SEARCH] Initialized with {len(collections)} collections: "
            f"{list(collections.keys())}"
        )

    def _search_single_collection(
        self,
        collection_name: str,
        collection,
        query_embedding: List[float],
        top_k: int,
        use_hybrid: bool = False,
        query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Recherche sur une seule collection.

        Returns:
            Résultats avec collection_name ajouté aux métadonnées
        """
        try:
            # Recherche dense (FAISS)
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Recherche hybride si disponible et activée
            if use_hybrid and collection_name in self.bm25_indices:
                from .hybrid_search import hybrid_search

                bm25_index = self.bm25_indices[collection_name]
                result = hybrid_search(
                    query=query_text,
                    collection=collection,
                    bm25_index=bm25_index,
                    embed_func=self.embed_func,
                    top_k=top_k,
                    log=self._log
                )

            # Ajouter collection_name aux métadonnées
            if result.get("metadatas") and result["metadatas"][0]:
                for meta in result["metadatas"][0]:
                    if meta:
                        meta["collection_name"] = collection_name

            return {
                "collection_name": collection_name,
                "result": result,
                "success": True,
                "error": None
            }

        except Exception as e:
            self._log.error(f"[MULTI-SEARCH] Error searching {collection_name}: {e}")
            return {
                "collection_name": collection_name,
                "result": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "success": False,
                "error": str(e)
            }

    def _merge_results_rrf(
        self,
        all_results: List[Dict[str, Any]],
        max_results: int
    ) -> Dict[str, Any]:
        """
        Fusionne les résultats de plusieurs collections avec RRF.

        Reciprocal Rank Fusion:
        score_rrf(doc) = sum(1 / (K + rank_i)) pour chaque système i

        Args:
            all_results: Liste des résultats par collection
            max_results: Nombre max de résultats fusionnés

        Returns:
            Résultats fusionnés au format FAISS standard
        """
        doc_scores: Dict[str, Dict[str, Any]] = {}

        for coll_result in all_results:
            if not coll_result["success"]:
                continue

            collection_name = coll_result["collection_name"]
            result = coll_result["result"]

            # Accès sécurisé
            docs = result.get("documents", [[]])[0] or []
            metas = result.get("metadatas", [[]])[0] or []
            dists = result.get("distances", [[]])[0] or []

            for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                # Créer une clé unique pour le document
                # Utiliser chunk_id si disponible, sinon hash du contenu
                doc_key = None
                if meta:
                    doc_key = meta.get("chunk_id") or meta.get("faiss_id")
                if not doc_key:
                    doc_text = doc[:200] if doc else ""
                    doc_key = hashlib.md5(
                        doc_text.encode('utf-8', errors='ignore')
                    ).hexdigest()[:16]

                # Clé composite incluant la collection pour éviter collisions
                composite_key = f"{collection_name}::{doc_key}"

                if composite_key not in doc_scores:
                    doc_scores[composite_key] = {
                        "document": doc,
                        "metadata": meta or {},
                        "collection_name": collection_name,
                        "ranks": {},
                        "distances": {},
                        "rrf_score": 0.0,
                    }

                # Score RRF pour cette collection
                rrf_contribution = 1.0 / (RRF_K + rank)
                doc_scores[composite_key]["rrf_score"] += rrf_contribution
                doc_scores[composite_key]["ranks"][collection_name] = rank
                doc_scores[composite_key]["distances"][collection_name] = dist

        # Trier par score RRF décroissant
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:max_results]

        # Formater les résultats
        documents = []
        metadatas = []
        distances = []

        for item in sorted_docs:
            documents.append(item["document"])

            # Enrichir les métadonnées avec provenance
            meta = item["metadata"].copy()
            meta["collection_name"] = item["collection_name"]
            meta["multi_search_rrf_score"] = item["rrf_score"]
            meta["multi_search_ranks"] = item["ranks"]
            metadatas.append(meta)

            # Distance moyenne pondérée (pour compatibilité)
            if item["distances"]:
                avg_dist = sum(item["distances"].values()) / len(item["distances"])
            else:
                avg_dist = 1.0  # Distance par défaut si aucune
            distances.append(avg_dist)

        self._log.info(
            f"[MULTI-SEARCH] Merged {len(doc_scores)} unique docs from "
            f"{len(all_results)} collections -> {len(documents)} results"
        )

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_PER_COLLECTION,
        use_hybrid: bool = False,
        collection_names: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Recherche fédérée sur plusieurs collections.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de résultats par collection avant fusion
            use_hybrid: Utiliser la recherche hybride (BM25 + dense)
            collection_names: Liste des collections à interroger (None = toutes)
            parallel: Exécuter les recherches en parallèle

        Returns:
            Résultats fusionnés au format FAISS standard avec provenance
        """
        # Sélectionner les collections
        if collection_names:
            selected = {
                name: coll for name, coll in self.collections.items()
                if name in collection_names
            }
        else:
            selected = self.collections

        if not selected:
            self._log.warning("[MULTI-SEARCH] No collections selected")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        self._log.info(
            f"[MULTI-SEARCH] Searching {len(selected)} collections: "
            f"{list(selected.keys())} | top_k={top_k} | hybrid={use_hybrid}"
        )

        # Générer l'embedding de la requête (une seule fois)
        query_embedding = self.embed_func(query).tolist()

        all_results = []

        if parallel and len(selected) > 1:
            # Recherche parallèle
            with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
                futures = {
                    executor.submit(
                        self._search_single_collection,
                        name, coll, query_embedding, top_k, use_hybrid, query
                    ): name
                    for name, coll in selected.items()
                }

                for future in as_completed(futures):
                    coll_name = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        self._log.debug(
                            f"[MULTI-SEARCH] {coll_name}: "
                            f"{'OK' if result['success'] else 'FAILED'}"
                        )
                    except Exception as e:
                        self._log.error(f"[MULTI-SEARCH] {coll_name} exception: {e}")
                        all_results.append({
                            "collection_name": coll_name,
                            "result": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                            "success": False,
                            "error": str(e)
                        })
        else:
            # Recherche séquentielle
            for name, coll in selected.items():
                result = self._search_single_collection(
                    name, coll, query_embedding, top_k, use_hybrid, query
                )
                all_results.append(result)

        # Fusionner les résultats
        merged = self._merge_results_rrf(all_results, max_results=top_k * 2)

        # Ajouter des statistiques
        merged["_multi_search_stats"] = {
            "collections_searched": list(selected.keys()),
            "collections_succeeded": [
                r["collection_name"] for r in all_results if r["success"]
            ],
            "collections_failed": [
                r["collection_name"] for r in all_results if not r["success"]
            ],
            "total_results": len(merged["documents"][0]),
        }

        return merged

    def search_with_reranking(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_PER_COLLECTION,
        final_top_k: int = 10,
        use_hybrid: bool = False,
        collection_names: Optional[List[str]] = None,
        http_client=None
    ) -> Dict[str, Any]:
        """
        Recherche fédérée avec reranking BGE sur les résultats fusionnés.

        Args:
            query: Question
            top_k: Top-K par collection avant fusion
            final_top_k: Nombre final de résultats après reranking
            use_hybrid: Utiliser recherche hybride
            collection_names: Collections à interroger
            http_client: Client HTTP pour l'API de reranking

        Returns:
            Résultats fusionnés et rerankés
        """
        # Recherche multi-collection
        results = self.search(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid,
            collection_names=collection_names
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return results

        # Reranking
        try:
            from .advanced_search import rerank_with_bge

            reranked = rerank_with_bge(
                query=query,
                documents=docs,
                top_k=final_top_k,
                http_client=http_client,
                log=self._log
            )

            # Réordonner selon le reranking
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            new_docs = []
            new_metas = []
            new_dists = []

            for r in reranked:
                idx = r["index"]
                if idx < len(docs):
                    new_docs.append(docs[idx])
                    new_metas.append(metas[idx] if idx < len(metas) else {})
                    new_dists.append(dists[idx] if idx < len(dists) else 1.0)
                    # Ajouter le score de reranking
                    if new_metas[-1]:
                        new_metas[-1]["rerank_score"] = r["score"]

            results["documents"] = [new_docs]
            results["metadatas"] = [new_metas]
            results["distances"] = [new_dists]

            self._log.info(
                f"[MULTI-SEARCH] Reranked to {len(new_docs)} results"
            )

        except Exception as e:
            self._log.warning(f"[MULTI-SEARCH] Reranking failed: {e}")

        return results


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def create_multi_collection_searcher(
    db_path: str,
    collection_names: List[str],
    embed_func,
    faiss_store_module,
    log: Optional[logging.Logger] = None
) -> MultiCollectionSearcher:
    """
    Crée un MultiCollectionSearcher à partir des noms de collections.

    Args:
        db_path: Chemin vers la base de données FAISS
        collection_names: Liste des noms de collections à charger
        embed_func: Fonction d'embedding
        faiss_store_module: Module faiss_store pour charger les collections
        log: Logger optionnel

    Returns:
        MultiCollectionSearcher configuré
    """
    _log = log or logger

    collections = {}
    bm25_indices = {}

    for name in collection_names:
        try:
            # Charger la collection FAISS
            collection = faiss_store_module.get_or_create_collection(
                db_path=db_path,
                collection_name=name,
                log=_log
            )
            collections[name] = collection

            # Essayer de charger l'index BM25 si disponible
            try:
                from .hybrid_search import build_or_load_bm25_index, get_collection_documents

                docs, doc_ids = get_collection_documents(collection)
                if len(docs) > 0:
                    bm25_index = build_or_load_bm25_index(
                        db_path=db_path,
                        collection_name=name,
                        documents=docs,
                        doc_ids=doc_ids,
                        log=_log
                    )
                    if bm25_index:
                        bm25_indices[name] = bm25_index
            except Exception as e:
                _log.debug(f"[MULTI-SEARCH] BM25 not available for {name}: {e}")

            _log.info(f"[MULTI-SEARCH] Loaded collection: {name}")

        except Exception as e:
            _log.error(f"[MULTI-SEARCH] Failed to load collection {name}: {e}")

    return MultiCollectionSearcher(
        collections=collections,
        embed_func=embed_func,
        bm25_indices=bm25_indices,
        log=_log
    )


def multi_collection_query(
    query: str,
    db_path: str,
    collection_names: List[str],
    embed_func,
    faiss_store_module,
    top_k: int = 30,
    use_hybrid: bool = True,
    use_reranking: bool = True,
    http_client=None,
    log: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Fonction utilitaire pour requête multi-collection complète.

    Args:
        query: Question
        db_path: Chemin base de données
        collection_names: Collections à interroger
        embed_func: Fonction d'embedding
        faiss_store_module: Module faiss_store
        top_k: Nombre de résultats
        use_hybrid: Utiliser recherche hybride
        use_reranking: Utiliser le reranking BGE
        http_client: Client HTTP
        log: Logger

    Returns:
        Tuple (résultats bruts, sources formatées avec provenance)
    """
    _log = log or logger

    # Créer le searcher
    searcher = create_multi_collection_searcher(
        db_path=db_path,
        collection_names=collection_names,
        embed_func=embed_func,
        faiss_store_module=faiss_store_module,
        log=_log
    )

    # Recherche
    if use_reranking:
        results = searcher.search_with_reranking(
            query=query,
            top_k=top_k,
            final_top_k=top_k,
            use_hybrid=use_hybrid,
            http_client=http_client
        )
    else:
        results = searcher.search(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid
        )

    # Formater les sources avec provenance
    sources = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        source = {
            "rank": i + 1,
            "text": doc,
            "score": 1.0 - (dist / 2.0) if dist else 0.5,
            "distance": dist,
            "collection_name": meta.get("collection_name", "unknown"),
            "path": meta.get("path", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "section_id": meta.get("section_id", ""),
            "page": meta.get("page", ""),
        }

        # Ajouter scores supplémentaires si disponibles
        if "rerank_score" in meta:
            source["rerank_score"] = meta["rerank_score"]
        if "multi_search_rrf_score" in meta:
            source["rrf_score"] = meta["multi_search_rrf_score"]

        sources.append(source)

    return results, sources


def format_sources_with_provenance(sources: List[Dict[str, Any]]) -> str:
    """
    Formate les sources pour affichage avec indication de provenance.

    Args:
        sources: Liste des sources avec collection_name

    Returns:
        Texte formaté pour le contexte LLM
    """
    if not sources:
        return ""

    formatted_parts = []

    for src in sources:
        collection = src.get("collection_name", "unknown")
        path = src.get("path", "")
        section_id = src.get("section_id", "")
        text = src.get("text", "")

        # Construire le header de source
        header_parts = [f"[Collection: {collection}]"]
        if path:
            header_parts.append(f"[File: {path}]")
        if section_id:
            header_parts.append(f"[Section: {section_id}]")

        header = " ".join(header_parts)
        formatted_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted_parts)
