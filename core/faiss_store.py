"""
FAISS-based vector store
Optimisé pour les partages réseau Windows (pas de SQLite)

Architecture:
- Un dossier par base de données (db_path)
- Dans chaque base: sous-dossiers pour chaque collection
- Dans chaque collection: index.faiss + metadata.json

Avantages:
- Pas de SQLite (pas de problèmes de verrouillage réseau)
- Fichiers simples qui se synchronisent bien
- Rapide
- Compatible partages réseau Windows

Fonctionnalités v2:
- Cache local pour accès rapide (évite latence réseau)
- Lazy loading (chargement différé de l'index FAISS)
"""

import os
import json
import shutil
import hashlib
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

# Import FAISS
try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS n'est pas installé. Installez-le avec:\n"
        "  pip install faiss-cpu\n"
        "ou pour GPU:\n"
        "  pip install faiss-gpu"
    )

logger = logging.getLogger(__name__)


# =====================================================================
#  GESTIONNAIRE DE CACHE LOCAL
# =====================================================================

class LocalCacheManager:
    """
    Gestionnaire de cache local pour les bases FAISS.
    Copie les fichiers depuis le réseau vers un répertoire local temporaire
    pour accélérer les lectures.
    """

    # Répertoire de cache par défaut
    DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "faiss_local_cache")

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Répertoire de cache local. Par défaut: %TEMP%/faiss_local_cache
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache_info_file = os.path.join(self.cache_dir, "_cache_info.json")
        self._cache_info = self._load_cache_info()
        logger.info(f"[CACHE] Initialisé dans: {self.cache_dir}")

    def _load_cache_info(self) -> Dict[str, Any]:
        """Charge les informations de cache."""
        if os.path.exists(self._cache_info_file):
            try:
                with open(self._cache_info_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[CACHE] Erreur lecture cache info: {e}")
        return {"collections": {}}

    def _save_cache_info(self):
        """Sauvegarde les informations de cache."""
        try:
            with open(self._cache_info_file, "w", encoding="utf-8") as f:
                json.dump(self._cache_info, f, indent=2)
        except Exception as e:
            logger.error(f"[CACHE] Erreur sauvegarde cache info: {e}")

    def _get_collection_key(self, network_path: str) -> str:
        """Génère une clé unique pour une collection basée sur son chemin réseau."""
        return hashlib.md5(network_path.encode()).hexdigest()[:12]

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """Calcule le hash MD5 d'un fichier pour détecter les changements."""
        if not os.path.exists(file_path):
            return None
        try:
            # Pour les gros fichiers, on utilise taille + mtime comme "hash rapide"
            stat = os.stat(file_path)
            return f"{stat.st_size}_{stat.st_mtime}"
        except Exception:
            return None

    def get_local_path(self, network_collection_path: str) -> str:
        """
        Retourne le chemin local correspondant à un chemin réseau.
        Ne copie pas les fichiers, retourne juste le chemin.
        """
        key = self._get_collection_key(network_collection_path)
        return os.path.join(self.cache_dir, key)

    def is_cached(self, network_collection_path: str) -> bool:
        """Vérifie si une collection est en cache local."""
        key = self._get_collection_key(network_collection_path)
        local_path = os.path.join(self.cache_dir, key)

        # Vérifier que les fichiers existent
        index_exists = os.path.exists(os.path.join(local_path, "index.faiss"))
        meta_exists = os.path.exists(os.path.join(local_path, "metadata.json"))

        return index_exists and meta_exists

    def is_cache_valid(self, network_collection_path: str) -> bool:
        """
        Vérifie si le cache est à jour par rapport à la source réseau.
        Compare les hashes des fichiers.
        """
        if not self.is_cached(network_collection_path):
            return False

        key = self._get_collection_key(network_collection_path)
        local_path = os.path.join(self.cache_dir, key)

        # Comparer les hashes
        for filename in ["index.faiss", "metadata.json"]:
            network_file = os.path.join(network_collection_path, filename)
            local_file = os.path.join(local_path, filename)

            network_hash = self._get_file_hash(network_file)
            cached_hash = self._cache_info.get("collections", {}).get(key, {}).get(f"{filename}_hash")

            if network_hash != cached_hash:
                logger.info(f"[CACHE] Fichier modifié sur réseau: {filename}")
                return False

        return True

    def copy_to_cache(
        self,
        network_collection_path: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Copie une collection du réseau vers le cache local.

        Args:
            network_collection_path: Chemin réseau de la collection
            progress_callback: Fonction callback(progress, message) pour progression

        Returns:
            Chemin local de la collection cachée
        """
        key = self._get_collection_key(network_collection_path)
        local_path = os.path.join(self.cache_dir, key)

        # Créer le répertoire local
        os.makedirs(local_path, exist_ok=True)

        files_to_copy = ["index.faiss", "metadata.json"]
        total_files = len(files_to_copy)
        hashes = {}

        for i, filename in enumerate(files_to_copy):
            network_file = os.path.join(network_collection_path, filename)
            local_file = os.path.join(local_path, filename)

            if os.path.exists(network_file):
                if progress_callback:
                    file_size = os.path.getsize(network_file) / (1024 * 1024)  # MB
                    progress_callback(
                        (i / total_files) * 100,
                        f"Copie {filename} ({file_size:.1f} MB)..."
                    )

                logger.info(f"[CACHE] Copie {network_file} -> {local_file}")
                shutil.copy2(network_file, local_file)
                hashes[f"{filename}_hash"] = self._get_file_hash(network_file)

        # Sauvegarder les infos de cache
        self._cache_info["collections"][key] = {
            "network_path": network_collection_path,
            "local_path": local_path,
            "cached_at": datetime.now().isoformat(),
            **hashes
        }
        self._save_cache_info()

        if progress_callback:
            progress_callback(100, "Cache local créé ✓")

        logger.info(f"[CACHE] Collection cachée: {network_collection_path} -> {local_path}")
        return local_path

    def invalidate_cache(self, network_collection_path: str):
        """Invalide le cache pour une collection (après modification)."""
        key = self._get_collection_key(network_collection_path)
        local_path = os.path.join(self.cache_dir, key)

        # Supprimer les fichiers locaux
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=True)

        # Mettre à jour les infos
        if key in self._cache_info.get("collections", {}):
            del self._cache_info["collections"][key]
            self._save_cache_info()

        logger.info(f"[CACHE] Cache invalidé pour: {network_collection_path}")

    def get_cache_status(self) -> Dict[str, Any]:
        """Retourne le statut global du cache."""
        total_size = 0
        collections = []

        for key, info in self._cache_info.get("collections", {}).items():
            local_path = info.get("local_path", "")
            if os.path.exists(local_path):
                size = sum(
                    os.path.getsize(os.path.join(local_path, f))
                    for f in os.listdir(local_path)
                    if os.path.isfile(os.path.join(local_path, f))
                )
                total_size += size
                collections.append({
                    "network_path": info.get("network_path"),
                    "local_path": local_path,
                    "size_mb": size / (1024 * 1024),
                    "cached_at": info.get("cached_at"),
                    "valid": self.is_cache_valid(info.get("network_path", ""))
                })

        return {
            "cache_dir": self.cache_dir,
            "total_size_mb": total_size / (1024 * 1024),
            "collections_count": len(collections),
            "collections": collections
        }

    def clear_all_cache(self):
        """Supprime tout le cache local."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)
        self._cache_info = {"collections": {}}
        self._save_cache_info()
        logger.info("[CACHE] Cache entièrement vidé")


# Instance globale du gestionnaire de cache
_global_cache_manager: Optional[LocalCacheManager] = None


def get_cache_manager(cache_dir: Optional[str] = None) -> LocalCacheManager:
    """Retourne l'instance globale du gestionnaire de cache."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = LocalCacheManager(cache_dir)
    return _global_cache_manager


# =====================================================================
#  COLLECTION FAISS AVEC LAZY LOADING
# =====================================================================

class FaissCollection:
    """Collection FAISS pour stocker et rechercher des embeddings (avec lazy loading)"""

    def __init__(
        self,
        collection_path: str,
        name: str,
        dimension: int = 1024,
        use_local_cache: bool = False,
        lazy_load: bool = True
    ):
        """
        Args:
            collection_path: Chemin du dossier de la collection (réseau ou local)
            name: Nom de la collection
            dimension: Dimension des embeddings (1024 pour Snowflake Arctic)
            use_local_cache: Si True, utilise le cache local pour les lectures
            lazy_load: Si True, charge l'index seulement au premier accès
        """
        self.network_path = collection_path  # Chemin réseau original
        self.name = name
        self.dimension = dimension
        self.use_local_cache = use_local_cache
        self._lazy_load = lazy_load
        self.cache_outdated = False  # Flag pour signaler un cache obsolète
        self.using_cache = False  # Flag pour indiquer si on utilise le cache

        # Déterminer le chemin effectif (local ou réseau)
        if use_local_cache:
            cache_mgr = get_cache_manager()
            if cache_mgr.is_cached(collection_path):
                # Vérifier si le cache est à jour
                if cache_mgr.is_cache_valid(collection_path):
                    # Cache valide → utiliser le cache local
                    self.collection_path = cache_mgr.get_local_path(collection_path)
                    self.using_cache = True
                    logger.info(f"[FAISS] ✅ Cache local valide: {self.collection_path}")
                else:
                    # Cache obsolète → utiliser le réseau et signaler
                    self.collection_path = collection_path
                    self.cache_outdated = True
                    logger.warning(f"[FAISS] ⚠️ Cache obsolète pour {name}, utilisation réseau")
            else:
                self.collection_path = collection_path
                logger.info(f"[FAISS] Cache local non disponible, utilisation réseau: {collection_path}")
        else:
            self.collection_path = collection_path

        # Chemins des fichiers
        self.index_path = os.path.join(self.collection_path, "index.faiss")
        self.metadata_path = os.path.join(self.collection_path, "metadata.json")

        # Créer le dossier si nécessaire
        os.makedirs(self.collection_path, exist_ok=True)

        # Lazy loading: index chargé à la demande
        self._index: Optional[faiss.Index] = None
        self._metadata: Optional[List[Dict]] = None
        self._ids: Optional[List[str]] = None
        self._idx_to_meta: Dict[int, Dict] = {}  # Lookup O(1) pour faiss_idx -> metadata
        self._loaded = False

        # Si lazy_load=False, charger immédiatement
        if not lazy_load:
            self._ensure_loaded()

    def _ensure_loaded(self):
        """Charge l'index et les métadonnées si pas encore fait (lazy loading)."""
        if self._loaded:
            return

        # Charger ou créer l'index FAISS
        if os.path.exists(self.index_path):
            logger.info(f"[FAISS] Loading existing index: {self.index_path}")
            self._index = faiss.read_index(self.index_path)
        else:
            logger.info(f"[FAISS] Creating new index with dimension {self.dimension}")
            self._index = faiss.IndexFlatL2(self.dimension)

        # Charger ou créer les métadonnées
        if os.path.exists(self.metadata_path):
            logger.info(f"[FAISS] Loading metadata: {self.metadata_path}")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._metadata = data.get("metadata", [])
                self._ids = data.get("ids", [])
        else:
            logger.info("[FAISS] Creating new metadata store")
            self._metadata = []
            self._ids = []

        # Construire le lookup faiss_idx -> metadata pour queries O(1)
        self._idx_to_meta = {
            m.get("faiss_idx"): m for m in self._metadata if "faiss_idx" in m
        }
        logger.debug(f"[FAISS] Built idx_to_meta lookup with {len(self._idx_to_meta)} entries")

        self._loaded = True

    @property
    def index(self) -> faiss.Index:
        """Accès à l'index FAISS (lazy loading)."""
        self._ensure_loaded()
        return self._index

    @index.setter
    def index(self, value):
        """Setter pour l'index."""
        self._index = value

    @property
    def metadata(self) -> List[Dict]:
        """Accès aux métadonnées (lazy loading)."""
        self._ensure_loaded()
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Setter pour les métadonnées."""
        self._metadata = value

    @property
    def ids(self) -> List[str]:
        """Accès aux IDs (lazy loading)."""
        self._ensure_loaded()
        return self._ids

    @ids.setter
    def ids(self, value):
        """Setter pour les IDs."""
        self._ids = value

    def is_loaded(self) -> bool:
        """Retourne True si l'index a été chargé."""
        return self._loaded

    def preload(self):
        """Force le chargement de l'index (utile pour pré-charger en arrière-plan)."""
        self._ensure_loaded()

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Ajoute des documents avec leurs embeddings à la collection.

        Args:
            ids: Liste des IDs uniques
            embeddings: Liste des vecteurs embeddings
            documents: Liste des textes des documents
            metadatas: Liste optionnelle de métadonnées (dicts)
        """
        # Vérifier que toutes les listes ont la même longueur
        if not (len(ids) == len(embeddings) == len(documents)):
            raise ValueError(
                f"ids, embeddings et documents doivent avoir la même taille. "
                f"Reçu: ids={len(ids)}, embeddings={len(embeddings)}, documents={len(documents)}"
            )

        if metadatas is None:
            metadatas = [{} for _ in ids]

        if len(metadatas) != len(ids):
            raise ValueError("metadatas doit avoir la même taille que ids")

        # Convertir embeddings en numpy array
        emb_array = np.array(embeddings, dtype=np.float32)

        # Vérifier la dimension
        if emb_array.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension des embeddings ({emb_array.shape[1]}) "
                f"ne correspond pas à la dimension de l'index ({self.dimension})"
            )

        # Ajouter à l'index FAISS
        start_idx = self.index.ntotal
        self.index.add(emb_array)

        # Stocker les métadonnées
        for i, (id_, doc, meta) in enumerate(zip(ids, documents, metadatas)):
            # Enrichir les métadonnées avec le document
            faiss_idx = start_idx + i
            full_meta = {
                "id": id_,
                "document": doc,
                "faiss_idx": faiss_idx,  # Index dans FAISS
                **meta
            }
            self.metadata.append(full_meta)
            self.ids.append(id_)
            # Mettre à jour le lookup O(1)
            self._idx_to_meta[faiss_idx] = full_meta

        logger.info(f"[FAISS] Added {len(ids)} documents (total: {self.index.ntotal})")

        # Sauvegarder automatiquement
        self._save()

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recherche les documents les plus similaires.

        Args:
            query_embeddings: Liste des vecteurs de requête
            n_results: Nombre de résultats à retourner
            include: Liste des champs à inclure

        Returns:
            Dict: {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        # Convertir en numpy
        query_array = np.array(query_embeddings, dtype=np.float32)

        # Vérifier qu'on a des données
        if self.index.ntotal == 0:
            logger.warning("[FAISS] Index is empty, returning no results")
            n_queries = len(query_embeddings)
            return {
                "ids": [[] for _ in range(n_queries)],
                "documents": [[] for _ in range(n_queries)],
                "metadatas": [[] for _ in range(n_queries)],
                "distances": [[] for _ in range(n_queries)]
            }

        # Limiter n_results au nombre de vecteurs disponibles
        n_results = min(n_results, self.index.ntotal)

        # Recherche FAISS
        distances, indices = self.index.search(query_array, n_results)

        # Formater les résultats
        results = {"ids": [], "documents": [], "metadatas": [], "distances": []}

        for i in range(len(query_embeddings)):
            query_ids = []
            query_docs = []
            query_metas = []
            query_dists = []

            for j, idx in enumerate(indices[i]):
                if idx == -1:  # FAISS retourne -1 si pas assez de résultats
                    continue

                # Trouver les métadonnées correspondantes - lookup O(1)
                meta = self._idx_to_meta.get(idx)

                if meta:
                    query_ids.append(meta.get("id", str(idx)))

                    if "documents" in include:
                        query_docs.append(meta.get("document", ""))

                    if "metadatas" in include:
                        # Copier les métadonnées sans les champs internes
                        clean_meta = {k: v for k, v in meta.items()
                                     if k not in ["id", "document", "faiss_idx"]}
                        # Ajouter le nom de la collection pour traçabilité
                        clean_meta["collection_name"] = self.name
                        query_metas.append(clean_meta)

                    if "distances" in include:
                        query_dists.append(float(distances[i][j]))

            results["ids"].append(query_ids)
            results["documents"].append(query_docs)
            results["metadatas"].append(query_metas)
            results["distances"].append(query_dists)

        logger.info(f"[FAISS] Query returned {len(results['ids'][0])} results")
        return results

    def get(
        self,
        ids: Optional[List[str]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Récupère tous les documents ou une sélection par IDs.
        Compatible avec l'interface ChromaDB pour la recherche hybride.

        Args:
            ids: Liste optionnelle d'IDs à récupérer. Si None, retourne tous les documents.
            include: Liste des champs à inclure ("documents", "metadatas")

        Returns:
            Dict: {"ids": [...], "documents": [...], "metadatas": [...]}
        """
        if include is None:
            include = ["documents", "metadatas"]

        self._ensure_loaded()

        result = {"ids": [], "documents": [], "metadatas": []}

        if ids is not None:
            # Récupérer uniquement les IDs spécifiés
            for id_ in ids:
                if id_ in self.ids:
                    idx = self.ids.index(id_)
                    meta = self.metadata[idx] if idx < len(self.metadata) else {}
                    result["ids"].append(id_)
                    if "documents" in include:
                        result["documents"].append(meta.get("document", ""))
                    if "metadatas" in include:
                        clean_meta = {k: v for k, v in meta.items()
                                     if k not in ["id", "document", "faiss_idx"]}
                        clean_meta["collection_name"] = self.name
                        result["metadatas"].append(clean_meta)
        else:
            # Récupérer tous les documents
            for i, id_ in enumerate(self.ids):
                meta = self.metadata[i] if i < len(self.metadata) else {}
                result["ids"].append(id_)
                if "documents" in include:
                    result["documents"].append(meta.get("document", ""))
                if "metadatas" in include:
                    clean_meta = {k: v for k, v in meta.items()
                                 if k not in ["id", "document", "faiss_idx"]}
                    clean_meta["collection_name"] = self.name
                    result["metadatas"].append(clean_meta)

        logger.debug(f"[FAISS] get() returned {len(result['ids'])} documents")
        return result

    def count(self) -> int:
        """Retourne le nombre de vecteurs dans l'index"""
        return self.index.ntotal

    def delete(self):
        """Supprime la collection (fichiers sur disque)"""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        logger.info(f"[FAISS] Deleted collection: {self.name}")

    def _save(self):
        """Sauvegarde l'index et les métadonnées sur disque"""
        # IMPORTANT: Toujours sauvegarder sur le chemin RÉSEAU (pas le cache local)
        save_path = self.network_path
        index_save_path = os.path.join(save_path, "index.faiss")
        metadata_save_path = os.path.join(save_path, "metadata.json")

        logger.info(f"[FAISS] Saving to: {save_path}")
        logger.info(f"[FAISS] Index path: {index_save_path}")
        logger.info(f"[FAISS] Metadata path: {metadata_save_path}")

        # S'assurer que le répertoire existe (critique pour les partages réseau Windows)
        os.makedirs(save_path, exist_ok=True)

        try:
            # Sauvegarder l'index FAISS
            faiss.write_index(self.index, index_save_path)
            logger.info(f"[FAISS] ✅ Index saved successfully")
        except Exception as e:
            logger.error(f"[FAISS] ❌ Failed to save index: {e}")
            raise

        try:
            # Sauvegarder les métadonnées
            with open(metadata_save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": self.metadata,
                    "ids": self.ids
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"[FAISS] ✅ Metadata saved successfully")
        except Exception as e:
            logger.error(f"[FAISS] ❌ Failed to save metadata: {e}")
            raise

        logger.info(f"[FAISS] Saved index and metadata to {save_path}")

        # Invalider le cache local si on utilise le cache
        if self.use_local_cache:
            cache_mgr = get_cache_manager()
            cache_mgr.invalidate_cache(self.network_path)
            logger.info(f"[FAISS] Cache local invalidé après sauvegarde")


class FaissStore:
    """Store FAISS pour gérer plusieurs collections (avec support cache local)"""

    def __init__(
        self,
        path: str,
        use_local_cache: bool = False,
        lazy_load: bool = True
    ):
        """
        Args:
            path: Chemin du répertoire de la base de données (réseau)
            use_local_cache: Si True, utilise le cache local pour les lectures
            lazy_load: Si True, charge les index seulement au premier accès
        """
        self.path = path
        self.use_local_cache = use_local_cache
        self.lazy_load = lazy_load
        os.makedirs(path, exist_ok=True)
        logger.info(f"[FAISS] Store initialized at: {path} (cache={use_local_cache}, lazy={lazy_load})")

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 1024
    ) -> FaissCollection:
        """
        Récupère ou crée une collection.

        Args:
            name: Nom de la collection
            dimension: Dimension des embeddings

        Returns:
            FaissCollection
        """
        collection_path = os.path.join(self.path, name)
        return FaissCollection(
            collection_path,
            name,
            dimension,
            use_local_cache=self.use_local_cache,
            lazy_load=self.lazy_load
        )

    def list_collections(self) -> List[str]:
        """Liste les noms de toutes les collections"""
        if not os.path.exists(self.path):
            return []

        collections = []
        for item in os.listdir(self.path):
            item_path = os.path.join(self.path, item)
            if os.path.isdir(item_path):
                # Vérifier qu'il y a bien un index ou des métadonnées
                if (os.path.exists(os.path.join(item_path, "index.faiss")) or
                    os.path.exists(os.path.join(item_path, "metadata.json"))):
                    collections.append(item)

        return collections

    def delete_collection(self, name: str):
        """Supprime une collection"""
        collection_path = os.path.join(self.path, name)
        if os.path.exists(collection_path):
            collection = FaissCollection(collection_path, name, lazy_load=False)
            collection.delete()
            # Supprimer le dossier s'il est vide
            try:
                os.rmdir(collection_path)
            except OSError:
                pass  # Le dossier n'est pas vide, on le laisse

            # Invalider le cache local si existant
            if self.use_local_cache:
                cache_mgr = get_cache_manager()
                cache_mgr.invalidate_cache(collection_path)

            logger.info(f"[FAISS] Deleted collection: {name}")

    def get_collection(self, name: str) -> FaissCollection:
        """
        Récupère une collection existante.

        Args:
            name: Nom de la collection

        Returns:
            FaissCollection

        Raises:
            ValueError si la collection n'existe pas
        """
        collection_path = os.path.join(self.path, name)
        if not os.path.exists(collection_path):
            raise ValueError(f"Collection '{name}' does not exist")

        return FaissCollection(
            collection_path,
            name,
            use_local_cache=self.use_local_cache,
            lazy_load=self.lazy_load
        )

    def cache_collection(
        self,
        name: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Copie une collection vers le cache local.

        Args:
            name: Nom de la collection
            progress_callback: Fonction callback(progress, message)

        Returns:
            Chemin local de la collection cachée
        """
        collection_path = os.path.join(self.path, name)
        if not os.path.exists(collection_path):
            raise ValueError(f"Collection '{name}' does not exist")

        cache_mgr = get_cache_manager()
        return cache_mgr.copy_to_cache(collection_path, progress_callback)

    def is_collection_cached(self, name: str) -> bool:
        """Vérifie si une collection est en cache local."""
        collection_path = os.path.join(self.path, name)
        cache_mgr = get_cache_manager()
        return cache_mgr.is_cached(collection_path)

    def get_collection_cache_info(self, name: str) -> Dict[str, Any]:
        """Retourne les infos de cache pour une collection."""
        collection_path = os.path.join(self.path, name)
        cache_mgr = get_cache_manager()
        key = cache_mgr._get_collection_key(collection_path)
        return cache_mgr._cache_info.get("collections", {}).get(key, {})


def build_faiss_store(
    path: str,
    use_local_cache: bool = False,
    lazy_load: bool = True
) -> FaissStore:
    """
    Crée un store FAISS.

    Args:
        path: Chemin du répertoire de la base de données
        use_local_cache: Si True, utilise le cache local pour les lectures
        lazy_load: Si True, charge les index seulement au premier accès

    Returns:
        FaissStore configuré
    """
    return FaissStore(path, use_local_cache=use_local_cache, lazy_load=lazy_load)


def get_or_create_collection(
    db_path: str,
    collection_name: str,
    dimension: int = 1024,
    use_local_cache: bool = False,
    lazy_load: bool = True,
    log: Optional[logging.Logger] = None
) -> FaissCollection:
    """
    Fonction utilitaire pour récupérer ou créer une collection FAISS.

    Args:
        db_path: Chemin de la base de données
        collection_name: Nom de la collection
        dimension: Dimension des embeddings (1024 pour Snowflake Arctic)
        use_local_cache: Utiliser le cache local
        lazy_load: Chargement différé de l'index
        log: Logger optionnel

    Returns:
        FaissCollection configurée
    """
    _log = log or logger
    store = FaissStore(db_path, use_local_cache=use_local_cache, lazy_load=lazy_load)
    collection = store.get_or_create_collection(collection_name, dimension)
    _log.info(f"[FAISS] Collection '{collection_name}' ready (items: {collection.count() if collection.is_loaded() else 'lazy'})")
    return collection


def list_all_collections(db_path: str) -> List[str]:
    """
    Liste toutes les collections disponibles dans une base de données.

    Args:
        db_path: Chemin de la base de données

    Returns:
        Liste des noms de collections
    """
    store = FaissStore(db_path, lazy_load=True)
    return store.list_collections()
