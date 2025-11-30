"""
Semantic Cache Module - Cache intelligent basé sur la similarité sémantique

Évite de recalculer les mêmes requêtes (ou requêtes similaires) en utilisant
la similarité cosinus entre les embeddings des questions.

Phase 2 improvements v1.2
"""

import logging
import time
import json
import os
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Seuil de similarité pour considérer une requête comme "identique"
DEFAULT_SIMILARITY_THRESHOLD = 0.95

# Durée de vie du cache (en secondes)
DEFAULT_CACHE_TTL = 3600  # 1 heure

# Taille maximum du cache (nombre d'entrées)
DEFAULT_MAX_CACHE_SIZE = 1000

# Fichier de persistance du cache
DEFAULT_CACHE_FILE = "semantic_cache.json"


# =============================================================================
#  CACHE ENTRY
# =============================================================================

@dataclass
class CacheEntry:
    """Une entrée du cache sémantique."""
    question: str
    question_embedding: List[float]
    collection_name: str
    answer: str
    sources: List[Dict[str, Any]]
    context_str: str
    timestamp: float
    hit_count: int = 0
    ttl: float = DEFAULT_CACHE_TTL

    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour la sérialisation."""
        return {
            "question": self.question,
            "question_embedding": self.question_embedding,
            "collection_name": self.collection_name,
            "answer": self.answer,
            "sources": self.sources,
            "context_str": self.context_str,
            "timestamp": self.timestamp,
            "hit_count": self.hit_count,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Crée une entrée depuis un dictionnaire."""
        return cls(
            question=data["question"],
            question_embedding=data["question_embedding"],
            collection_name=data["collection_name"],
            answer=data["answer"],
            sources=data.get("sources", []),
            context_str=data.get("context_str", ""),
            timestamp=data["timestamp"],
            hit_count=data.get("hit_count", 0),
            ttl=data.get("ttl", DEFAULT_CACHE_TTL),
        )


# =============================================================================
#  SEMANTIC CACHE
# =============================================================================

class SemanticCache:
    """
    Cache sémantique pour les requêtes RAG.

    Utilise la similarité cosinus pour trouver des requêtes similaires
    et éviter de recalculer les mêmes réponses.

    Features:
    - Recherche par similarité sémantique (pas juste exact match)
    - TTL configurable par entrée
    - Persistance sur disque
    - Statistiques d'utilisation
    - Auto-nettoyage des entrées expirées
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_size: int = DEFAULT_MAX_CACHE_SIZE,
        default_ttl: float = DEFAULT_CACHE_TTL,
        cache_dir: Optional[str] = None,
        log=None
    ):
        """
        Args:
            similarity_threshold: Seuil de similarité (0-1) pour considérer un hit
            max_size: Nombre maximum d'entrées dans le cache
            default_ttl: Durée de vie par défaut des entrées (secondes)
            cache_dir: Répertoire pour la persistance (None = pas de persistance)
            log: Logger optionnel
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_dir = cache_dir
        self._log = log or logger

        # Cache en mémoire: collection_name -> [CacheEntry, ...]
        self._cache: Dict[str, List[CacheEntry]] = {}

        # Statistiques
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        # Charger le cache persistant si disponible
        if cache_dir:
            self._load_cache()

        self._log.info(
            f"[CACHE] Initialized: threshold={similarity_threshold}, "
            f"max_size={max_size}, ttl={default_ttl}s"
        )

    def _compute_similarity(
        self,
        emb1: List[float],
        emb2: List[float]
    ) -> float:
        """Calcule la similarité cosinus entre deux embeddings."""
        v1 = np.array(emb1)
        v2 = np.array(emb2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get(
        self,
        question: str,
        question_embedding: List[float],
        collection_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Recherche une réponse en cache pour une question.

        Args:
            question: Question de l'utilisateur
            question_embedding: Embedding de la question
            collection_name: Nom de la collection

        Returns:
            Résultat en cache ou None si pas de hit
        """
        if collection_name not in self._cache:
            self._stats["misses"] += 1
            return None

        entries = self._cache[collection_name]
        best_match: Optional[CacheEntry] = None
        best_similarity = 0.0

        # Chercher la meilleure correspondance
        valid_entries = []
        for entry in entries:
            # Vérifier l'expiration
            if entry.is_expired():
                self._stats["expirations"] += 1
                continue

            valid_entries.append(entry)

            # Calculer la similarité
            similarity = self._compute_similarity(
                question_embedding,
                entry.question_embedding
            )

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry

        # Mettre à jour la liste des entrées valides
        self._cache[collection_name] = valid_entries

        if best_match:
            # Cache hit
            best_match.hit_count += 1
            self._stats["hits"] += 1

            self._log.info(
                f"[CACHE] ✅ HIT: similarity={best_similarity:.3f}, "
                f"hits={best_match.hit_count}, question='{question[:50]}...'"
            )

            return {
                "answer": best_match.answer,
                "sources": best_match.sources,
                "context_str": best_match.context_str,
                "cached": True,
                "cache_similarity": best_similarity,
                "cache_question": best_match.question,
            }

        self._stats["misses"] += 1
        self._log.debug(f"[CACHE] MISS: question='{question[:50]}...'")
        return None

    def put(
        self,
        question: str,
        question_embedding: List[float],
        collection_name: str,
        answer: str,
        sources: List[Dict[str, Any]],
        context_str: str,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Ajoute une entrée au cache.

        Args:
            question: Question de l'utilisateur
            question_embedding: Embedding de la question
            collection_name: Nom de la collection
            answer: Réponse générée
            sources: Sources utilisées
            context_str: Contexte utilisé
            ttl: Durée de vie (secondes), None = default
        """
        if collection_name not in self._cache:
            self._cache[collection_name] = []

        # Vérifier si une entrée similaire existe déjà
        entries = self._cache[collection_name]
        for i, entry in enumerate(entries):
            similarity = self._compute_similarity(
                question_embedding,
                entry.question_embedding
            )
            if similarity >= self.similarity_threshold:
                # Mettre à jour l'entrée existante
                entries[i] = CacheEntry(
                    question=question,
                    question_embedding=question_embedding,
                    collection_name=collection_name,
                    answer=answer,
                    sources=sources,
                    context_str=context_str,
                    timestamp=time.time(),
                    hit_count=entry.hit_count,
                    ttl=ttl or self.default_ttl,
                )
                self._log.debug(f"[CACHE] Updated existing entry")
                return

        # Vérifier la taille du cache
        total_entries = sum(len(e) for e in self._cache.values())
        if total_entries >= self.max_size:
            self._evict_oldest()

        # Ajouter la nouvelle entrée
        entry = CacheEntry(
            question=question,
            question_embedding=question_embedding,
            collection_name=collection_name,
            answer=answer,
            sources=sources,
            context_str=context_str,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
        )
        entries.append(entry)

        self._log.debug(f"[CACHE] Added new entry: question='{question[:50]}...'")

        # Sauvegarder si persistance activée
        if self.cache_dir:
            self._save_cache()

    def _evict_oldest(self) -> None:
        """Supprime les entrées les plus anciennes/moins utilisées."""
        # Collecter toutes les entrées avec leur score
        all_entries: List[Tuple[str, int, CacheEntry, float]] = []
        for collection_name, entries in self._cache.items():
            for i, entry in enumerate(entries):
                # Score = hit_count / age (privilégie les entrées récentes et populaires)
                age = time.time() - entry.timestamp
                score = entry.hit_count / (age + 1)
                all_entries.append((collection_name, i, entry, score))

        # Trier par score croissant (les moins utiles en premier)
        all_entries.sort(key=lambda x: x[3])

        # Supprimer 10% des entrées les moins utiles
        to_remove = max(1, len(all_entries) // 10)

        for collection_name, idx, entry, _ in all_entries[:to_remove]:
            if collection_name in self._cache:
                try:
                    self._cache[collection_name].remove(entry)
                    self._stats["evictions"] += 1
                except ValueError:
                    # Entry already removed (possible race condition)
                    self._log.debug(f"[CACHE] Entry already removed during eviction")

        self._log.info(f"[CACHE] Evicted {to_remove} entries")

    def invalidate(self, collection_name: Optional[str] = None) -> int:
        """
        Invalide le cache (après ingestion par exemple).

        Args:
            collection_name: Collection à invalider, ou None pour tout

        Returns:
            Nombre d'entrées supprimées
        """
        if collection_name:
            count = len(self._cache.get(collection_name, []))
            self._cache[collection_name] = []
            self._log.info(f"[CACHE] Invalidated {count} entries for '{collection_name}'")
        else:
            count = sum(len(e) for e in self._cache.values())
            self._cache = {}
            self._log.info(f"[CACHE] Invalidated all {count} entries")

        if self.cache_dir:
            self._save_cache()

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total_entries = sum(len(e) for e in self._cache.values())
        hit_rate = (
            self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
            if (self._stats["hits"] + self._stats["misses"]) > 0
            else 0.0
        )

        return {
            **self._stats,
            "total_entries": total_entries,
            "collections": len(self._cache),
            "hit_rate": round(hit_rate, 3),
            "threshold": self.similarity_threshold,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }

    def _get_cache_path(self) -> str:
        """Retourne le chemin du fichier de cache."""
        if not self.cache_dir:
            return ""
        return os.path.join(self.cache_dir, DEFAULT_CACHE_FILE)

    def _save_cache(self) -> None:
        """Sauvegarde le cache sur disque."""
        if not self.cache_dir:
            return

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = self._get_cache_path()

            data = {
                collection_name: [entry.to_dict() for entry in entries]
                for collection_name, entries in self._cache.items()
            }

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._log.debug(f"[CACHE] Saved to {path}")

        except Exception as e:
            self._log.warning(f"[CACHE] Failed to save: {e}")

    def _load_cache(self) -> None:
        """Charge le cache depuis le disque."""
        if not self.cache_dir:
            return

        path = self._get_cache_path()
        if not os.path.exists(path):
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for collection_name, entries_data in data.items():
                entries = []
                for entry_data in entries_data:
                    entry = CacheEntry.from_dict(entry_data)
                    # Ne pas charger les entrées expirées
                    if not entry.is_expired():
                        entries.append(entry)

                if entries:
                    self._cache[collection_name] = entries

            total = sum(len(e) for e in self._cache.values())
            self._log.info(f"[CACHE] Loaded {total} entries from {path}")

        except Exception as e:
            self._log.warning(f"[CACHE] Failed to load: {e}")

    def cleanup_expired(self) -> int:
        """
        Nettoie les entrées expirées.

        Returns:
            Nombre d'entrées supprimées
        """
        count = 0
        for collection_name in list(self._cache.keys()):
            entries = self._cache[collection_name]
            valid_entries = [e for e in entries if not e.is_expired()]
            count += len(entries) - len(valid_entries)
            self._cache[collection_name] = valid_entries

        if count > 0:
            self._log.info(f"[CACHE] Cleaned up {count} expired entries")
            if self.cache_dir:
                self._save_cache()

        return count


# =============================================================================
#  GLOBAL CACHE INSTANCE
# =============================================================================

_global_cache: Optional[SemanticCache] = None


def get_semantic_cache(
    cache_dir: Optional[str] = None,
    **kwargs
) -> SemanticCache:
    """
    Retourne l'instance globale du cache sémantique.

    Args:
        cache_dir: Répertoire de persistance
        **kwargs: Arguments pour SemanticCache

    Returns:
        Instance du cache
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = SemanticCache(cache_dir=cache_dir, **kwargs)

    return _global_cache


def invalidate_cache(collection_name: Optional[str] = None) -> int:
    """
    Invalide le cache global.

    Args:
        collection_name: Collection à invalider, ou None pour tout

    Returns:
        Nombre d'entrées supprimées
    """
    global _global_cache

    if _global_cache is None:
        return 0

    return _global_cache.invalidate(collection_name)
