# offline_models.py
"""
Module de gestion des modeles locaux pour le mode offline.

Modeles supportes:
- LLM: Mistral-7B-Instruct-v0.3
- Embeddings: BGE-M3 (1024 dimensions)
- Reranker: BGE-Reranker-v2-m3
- OCR: Donut-base

Features:
- Detection automatique GPU NVIDIA (CUDA)
- Optimisation VRAM selon la carte (RTX 4060, 4090, etc.)
- Fallback automatique CPU si pas de GPU
- Chargement lazy des modeles (a la demande)
- Singleton pattern pour eviter les chargements multiples
"""

import os
import sys
import logging
import threading
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURATION CHEMINS MODELES OFFLINE
# =============================================================================

OFFLINE_MODELS_CONFIG = {
    "llm": {
        "path": r"D:\IA_Test\models\mistralai\Mistral-7B-Instruct-v0.3",
        "name": "Mistral-7B-Instruct-v0.3",
        "type": "causal_lm",
    },
    "embeddings": {
        "path": r"D:\IA_Test\models\BAAI\bge-m3",
        "name": "BGE-M3",
        "type": "embeddings",
        "dimension": 1024,
    },
    "reranker": {
        "path": r"D:\IA_Test\models\BAAI\bge-reranker-v2-m3",
        "name": "BGE-Reranker-v2-m3",
        "type": "reranker",
    },
    "ocr": {
        "path": r"D:\IA_Test\models\donut-base",
        "name": "Donut-base",
        "type": "vision_encoder_decoder",
    },
}

# Chemins de stockage avec fallback - importes depuis config_manager pour eviter duplication
try:
    from core.config_manager import PRIMARY_NETWORK_BASE, FALLBACK_LOCAL_BASE
    STORAGE_PATHS = {
        "primary": PRIMARY_NETWORK_BASE,
        "fallback": FALLBACK_LOCAL_BASE,
    }
except ImportError:
    # Fallback si config_manager n'est pas accessible
    STORAGE_PATHS = {
        "primary": r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE",
        "fallback": r"D:\FAISS_DATABASE",
    }


# =============================================================================
#  GPU DETECTION & CONFIGURATION
# =============================================================================

class GPUTier(Enum):
    """Tiers de GPU pour la configuration automatique."""
    NONE = "none"           # Pas de GPU -> CPU only
    LOW = "low"             # 4-6GB VRAM (GTX 1650, RTX 3050)
    MEDIUM = "medium"       # 8GB VRAM (RTX 4060, RTX 3070)
    HIGH = "high"           # 12-16GB VRAM (RTX 4070, RTX 3090)
    ULTRA = "ultra"         # 24GB+ VRAM (RTX 4090, A100)


@dataclass
class GPUInfo:
    """Informations sur le GPU detecte."""
    available: bool
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: str
    compute_capability: Tuple[int, int]
    tier: GPUTier

    def __str__(self) -> str:
        if not self.available:
            return "GPU: Non disponible (mode CPU)"
        return (
            f"GPU: {self.device_name} | "
            f"VRAM: {self.vram_free_gb:.1f}/{self.vram_total_gb:.1f} GB | "
            f"CUDA: {self.cuda_version} | "
            f"Tier: {self.tier.value}"
        )


def detect_gpu() -> GPUInfo:
    """
    Detecte le GPU NVIDIA et ses capacites.

    Returns:
        GPUInfo avec les informations du GPU ou mode CPU
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("[GPU] CUDA non disponible - mode CPU")
            return GPUInfo(
                available=False,
                device_name="CPU",
                vram_total_gb=0,
                vram_free_gb=0,
                cuda_version="N/A",
                compute_capability=(0, 0),
                tier=GPUTier.NONE,
            )

        # Informations GPU
        device_id = 0
        device_name = torch.cuda.get_device_name(device_id)

        # VRAM
        vram_total = torch.cuda.get_device_properties(device_id).total_memory
        vram_free = vram_total - torch.cuda.memory_allocated(device_id)
        vram_total_gb = vram_total / (1024**3)
        vram_free_gb = vram_free / (1024**3)

        # CUDA version
        cuda_version = torch.version.cuda or "Unknown"

        # Compute capability
        props = torch.cuda.get_device_properties(device_id)
        compute_capability = (props.major, props.minor)

        # Determiner le tier
        tier = _determine_gpu_tier(vram_total_gb, device_name)

        gpu_info = GPUInfo(
            available=True,
            device_name=device_name,
            vram_total_gb=vram_total_gb,
            vram_free_gb=vram_free_gb,
            cuda_version=cuda_version,
            compute_capability=compute_capability,
            tier=tier,
        )

        logger.info(f"[GPU] Detecte: {gpu_info}")
        return gpu_info

    except ImportError:
        logger.warning("[GPU] PyTorch non installe - mode CPU")
        return GPUInfo(
            available=False,
            device_name="CPU (PyTorch non installe)",
            vram_total_gb=0,
            vram_free_gb=0,
            cuda_version="N/A",
            compute_capability=(0, 0),
            tier=GPUTier.NONE,
        )
    except Exception as e:
        logger.error(f"[GPU] Erreur detection: {e}")
        return GPUInfo(
            available=False,
            device_name=f"CPU (Erreur: {e})",
            vram_total_gb=0,
            vram_free_gb=0,
            cuda_version="N/A",
            compute_capability=(0, 0),
            tier=GPUTier.NONE,
        )


def _determine_gpu_tier(vram_gb: float, device_name: str) -> GPUTier:
    """Determine le tier GPU en fonction de la VRAM et du nom."""
    # Detection par nom pour certains GPU specifiques
    name_lower = device_name.lower()

    if "4090" in name_lower or "a100" in name_lower or "h100" in name_lower:
        return GPUTier.ULTRA
    elif "4080" in name_lower or "3090" in name_lower or "4070 ti" in name_lower:
        return GPUTier.HIGH
    elif "4070" in name_lower or "3080" in name_lower:
        return GPUTier.HIGH
    elif "4060" in name_lower or "3070" in name_lower or "3060 ti" in name_lower:
        return GPUTier.MEDIUM
    elif "3060" in name_lower or "3050" in name_lower:
        return GPUTier.LOW

    # Fallback sur la VRAM
    if vram_gb >= 20:
        return GPUTier.ULTRA
    elif vram_gb >= 12:
        return GPUTier.HIGH
    elif vram_gb >= 8:
        return GPUTier.MEDIUM
    elif vram_gb >= 4:
        return GPUTier.LOW
    else:
        return GPUTier.NONE


def get_optimal_config(gpu_info: GPUInfo) -> Dict[str, Any]:
    """
    Retourne la configuration optimale selon le GPU.

    Args:
        gpu_info: Informations GPU

    Returns:
        Dict avec les parametres optimaux
    """
    configs = {
        GPUTier.ULTRA: {
            "dtype": "float16",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "max_batch_size": 64,
            "max_length": 4096,
            "device_map": "auto",
            "use_flash_attention": True,
        },
        GPUTier.HIGH: {
            "dtype": "float16",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "max_batch_size": 32,
            "max_length": 2048,
            "device_map": "auto",
            "use_flash_attention": True,
        },
        GPUTier.MEDIUM: {
            "dtype": "float16",
            "load_in_8bit": False,
            "load_in_4bit": True,  # Quantification 4-bit pour 8GB
            "max_batch_size": 16,
            "max_length": 2048,
            "device_map": "auto",
            "use_flash_attention": False,
        },
        GPUTier.LOW: {
            "dtype": "float16",
            "load_in_8bit": True,
            "load_in_4bit": False,
            "max_batch_size": 8,
            "max_length": 1024,
            "device_map": "auto",
            "use_flash_attention": False,
        },
        GPUTier.NONE: {
            "dtype": "float32",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "max_batch_size": 4,
            "max_length": 1024,
            "device_map": "cpu",
            "use_flash_attention": False,
        },
    }

    return configs.get(gpu_info.tier, configs[GPUTier.NONE])


# =============================================================================
#  STORAGE PATH MANAGEMENT
# =============================================================================

def get_storage_path(subdir: str = "") -> str:
    """
    Retourne le chemin de stockage avec fallback automatique.

    Verifie d'abord le chemin primaire (N:\), sinon utilise le fallback (D:\).

    Args:
        subdir: Sous-repertoire optionnel

    Returns:
        Chemin accessible
    """
    primary = STORAGE_PATHS["primary"]
    fallback = STORAGE_PATHS["fallback"]

    # Tester le chemin primaire
    if _is_path_accessible(primary):
        base_path = primary
        logger.debug(f"[STORAGE] Utilisation chemin primaire: {primary}")
    else:
        base_path = fallback
        logger.info(f"[STORAGE] Chemin primaire inaccessible, fallback vers: {fallback}")

    if subdir:
        return os.path.join(base_path, subdir)
    return base_path


def _is_path_accessible(path: str) -> bool:
    """Verifie si un chemin est accessible en lecture."""
    try:
        if not path:
            return False
        # Normaliser le chemin pour Windows
        path_upper = path.upper()
        # Verifier chemin reseau Windows (\\server\share ou //server/share)
        # ou lettre de lecteur (N:, D:, etc.)
        is_windows_path = (
            path.startswith(r"\\") or
            path.startswith("//") or
            (len(path) >= 2 and path_upper[1] == ":")
        )
        if is_windows_path:
            return os.path.exists(path) and os.access(path, os.R_OK)
        return os.path.exists(path) and os.access(path, os.R_OK)
    except (OSError, PermissionError, TypeError):
        return False


def check_offline_models_available() -> Dict[str, bool]:
    """
    Verifie la disponibilite des modeles offline.

    Returns:
        Dict avec le statut de chaque modele
    """
    status = {}
    for model_key, config in OFFLINE_MODELS_CONFIG.items():
        path = config["path"]
        status[model_key] = os.path.exists(path) and os.path.isdir(path)
        if status[model_key]:
            logger.info(f"[OFFLINE] Modele {model_key} disponible: {path}")
        else:
            logger.warning(f"[OFFLINE] Modele {model_key} NON TROUVE: {path}")
    return status


# =============================================================================
#  OFFLINE EMBEDDINGS (BGE-M3)
# =============================================================================

class OfflineEmbeddings:
    """
    Client embeddings offline utilisant BGE-M3.

    Produit des embeddings de dimension 1024, compatible avec Snowflake Arctic.
    Thread-safe avec verrou pour les appels concurrents.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern pour eviter les chargements multiples."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, log=None):
        # Double-check locking pour thread-safety
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._log = log or logger
            self.model_path = model_path or OFFLINE_MODELS_CONFIG["embeddings"]["path"]
            self.model = None
            self.tokenizer = None
            self.device = None
            self.dimension = OFFLINE_MODELS_CONFIG["embeddings"].get("dimension", 1024)
            self._model_lock = threading.Lock()
            self._initialized = True

    def _ensure_loaded(self):
        """Charge le modele si pas deja fait (lazy loading)."""
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            # Verifier que le chemin existe
            if not os.path.exists(self.model_path):
                self._log.error(f"[OFFLINE-EMB] Chemin modele introuvable: {self.model_path}")
                raise FileNotFoundError(
                    f"Modele BGE-M3 introuvable: {self.model_path}\n"
                    f"Verifiez que le modele est telecharge dans le bon repertoire."
                )

            self._log.info(f"[OFFLINE-EMB] Chargement BGE-M3 depuis {self.model_path}...")

            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                # Detecter le device optimal
                gpu_info = detect_gpu()
                config = get_optimal_config(gpu_info)

                self.device = "cuda" if gpu_info.available else "cpu"

                self._log.info(f"[OFFLINE-EMB] Device: {self.device}, GPU tier: {gpu_info.tier.value}")

                # Charger le tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                except Exception as tok_err:
                    self._log.error(f"[OFFLINE-EMB] Erreur chargement tokenizer: {tok_err}")
                    raise RuntimeError(f"Erreur tokenizer BGE-M3: {tok_err}")

                # Charger le modele avec la configuration optimale
                dtype = torch.float16 if config["dtype"] == "float16" and gpu_info.available else torch.float32

                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                    )
                except Exception as model_err:
                    self._log.error(f"[OFFLINE-EMB] Erreur chargement modele: {model_err}")
                    raise RuntimeError(f"Erreur modele BGE-M3: {model_err}")

                self.model = self.model.to(self.device)
                self.model.eval()

                self._log.info(f"[OFFLINE-EMB] BGE-M3 charge sur {self.device} ({dtype})")

            except ImportError as ie:
                self._log.error(f"[OFFLINE-EMB] Dependances manquantes: {ie}")
                raise RuntimeError(
                    f"Dependances manquantes pour le mode offline: {ie}\n"
                    f"Installez les dependances avec: pip install -r requirements-offline.txt"
                )
            except Exception as e:
                self._log.error(f"[OFFLINE-EMB] Erreur chargement BGE-M3: {e}")
                raise RuntimeError(f"Impossible de charger BGE-M3: {e}")

    def _apply_prefix(self, texts: List[str], role: str) -> List[str]:
        """Applique le prefixe selon le role (query/passage)."""
        if role == "query":
            return [f"query: {t}" for t in texts]
        else:
            return [f"passage: {t}" for t in texts]

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Genere les embeddings pour une liste de textes."""
        import torch

        self._ensure_loaded()

        try:
            with self._model_lock:
                with torch.no_grad():
                    # Tokenizer avec padding
                    inputs = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=8192,
                        return_tensors="pt",
                    )

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward pass
                    outputs = self.model(**inputs)

                    # Mean pooling sur les embeddings
                    attention_mask = inputs["attention_mask"]
                    embeddings = outputs.last_hidden_state

                    # Masquer les tokens de padding
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)

                    mean_embeddings = sum_embeddings / sum_mask

                    # Normaliser L2
                    normalized = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

                    return normalized.cpu().numpy().astype(np.float32)

        except RuntimeError as e:
            # Gestion CUDA OOM
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._log.error(f"[OFFLINE-EMB] CUDA OOM lors de l'embedding: {e}")
                # Vider le cache GPU
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # Retourner des embeddings zeros en fallback
                self._log.warning(f"[OFFLINE-EMB] Fallback: embeddings zeros pour {len(texts)} textes")
                return np.zeros((len(texts), self.dimension), dtype=np.float32)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genere les embeddings pour des documents.

        Args:
            texts: Liste de textes

        Returns:
            Liste d'embeddings (listes de floats, dimension 1024)
        """
        if not texts:
            return []

        try:
            prefixed = self._apply_prefix(texts, role="passage")
            embeddings = self._embed(prefixed)

            # Verifier la dimension
            if embeddings.shape[1] != self.dimension:
                self._log.warning(
                    f"[OFFLINE-EMB] Dimension inattendue: {embeddings.shape[1]} vs {self.dimension}"
                )

            return embeddings.tolist()
        except Exception as e:
            self._log.error(f"[OFFLINE-EMB] Erreur embed_documents: {e}")
            raise

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """
        Genere les embeddings pour des requetes.

        Args:
            texts: Liste de requetes

        Returns:
            Liste d'embeddings (listes de floats, dimension 1024)
        """
        if not texts:
            return []

        try:
            prefixed = self._apply_prefix(texts, role="query")
            embeddings = self._embed(prefixed)

            # Verifier la dimension
            if embeddings.shape[1] != self.dimension:
                self._log.warning(
                    f"[OFFLINE-EMB] Dimension inattendue: {embeddings.shape[1]} vs {self.dimension}"
                )

            return embeddings.tolist()
        except Exception as e:
            self._log.error(f"[OFFLINE-EMB] Erreur embed_queries: {e}")
            raise

    def embed_single(self, text: str, role: str = "query") -> np.ndarray:
        """
        Genere l'embedding pour un seul texte.

        Args:
            text: Texte a encoder
            role: "query" ou "passage"

        Returns:
            Embedding numpy array (float32, dimension 1024)
        """
        # Verifier que le texte n'est pas None ou vide
        if not text:
            self._log.warning("[OFFLINE-EMB] embed_single appele avec texte vide/None")
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            if role == "query":
                emb = np.array(self.embed_queries([text])[0], dtype=np.float32)
            else:
                emb = np.array(self.embed_documents([text])[0], dtype=np.float32)

            # Verifier la dimension
            if len(emb) != self.dimension:
                self._log.warning(
                    f"[OFFLINE-EMB] Dimension embedding: {len(emb)} vs attendu {self.dimension}"
                )

            return emb
        except Exception as e:
            self._log.error(f"[OFFLINE-EMB] Erreur embed_single: {e}")
            raise


# =============================================================================
#  OFFLINE RERANKER (BGE-Reranker-v2-m3)
# =============================================================================

class OfflineReranker:
    """
    Reranker offline utilisant BGE-Reranker-v2-m3.

    Thread-safe avec singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, log=None):
        # Double-check locking pour thread-safety
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._log = log or logger
            self.model_path = model_path or OFFLINE_MODELS_CONFIG["reranker"]["path"]
            self.model = None
            self.tokenizer = None
            self.device = None
            self._model_lock = threading.Lock()
            self._initialized = True

    def _ensure_loaded(self):
        """Charge le modele si pas deja fait."""
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            # Verifier que le chemin existe
            if not os.path.exists(self.model_path):
                self._log.error(f"[OFFLINE-RR] Chemin modele introuvable: {self.model_path}")
                raise FileNotFoundError(
                    f"Modele BGE-Reranker introuvable: {self.model_path}\n"
                    f"Verifiez que le modele est telecharge dans le bon repertoire."
                )

            self._log.info(f"[OFFLINE-RR] Chargement BGE-Reranker depuis {self.model_path}...")

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                gpu_info = detect_gpu()
                config = get_optimal_config(gpu_info)

                self.device = "cuda" if gpu_info.available else "cpu"
                self._log.info(f"[OFFLINE-RR] Device: {self.device}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )

                dtype = torch.float16 if config["dtype"] == "float16" and gpu_info.available else torch.float32

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )

                self.model = self.model.to(self.device)
                self.model.eval()

                self._log.info(f"[OFFLINE-RR] BGE-Reranker charge sur {self.device}")

            except ImportError as ie:
                self._log.error(f"[OFFLINE-RR] Dependances manquantes: {ie}")
                raise RuntimeError(f"Dependances manquantes: {ie}")
            except Exception as e:
                self._log.error(f"[OFFLINE-RR] Erreur chargement: {e}")
                raise RuntimeError(f"Impossible de charger BGE-Reranker: {e}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Reranke les documents selon leur pertinence pour la requete.

        Args:
            query: Requete
            documents: Liste de documents a reranker
            top_k: Nombre de resultats a retourner (None = tous)

        Returns:
            Liste de (index_original, score) triee par score decroissant
        """
        import torch

        if not documents:
            return []

        self._ensure_loaded()

        try:
            with self._model_lock:
                with torch.no_grad():
                    # Preparer les paires query-document
                    pairs = [[query, doc] for doc in documents]

                    # Tokenizer
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Scores
                    outputs = self.model(**inputs)
                    scores = outputs.logits.squeeze(-1).cpu().numpy()

                    # Si une seule dimension, convertir en array
                    if scores.ndim == 0:
                        scores = np.array([scores])

                    # Creer les paires (index, score) et trier
                    results = [(i, float(s)) for i, s in enumerate(scores)]
                    results.sort(key=lambda x: x[1], reverse=True)
        except RuntimeError as e:
            # Gestion CUDA OOM
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._log.error(f"[OFFLINE-RR] CUDA OOM lors du reranking: {e}")
                # Vider le cache GPU
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # Retourner les documents dans l'ordre original avec score 0
                self._log.warning("[OFFLINE-RR] Fallback: retour ordre original")
                return [(i, 0.0) for i in range(len(documents))]
            raise
        except Exception as e:
            self._log.error(f"[OFFLINE-RR] Erreur reranking: {e}")
            # Retourner les documents dans l'ordre original
            return [(i, 0.0) for i in range(len(documents))]

        if top_k:
            results = results[:top_k]

        return results


# =============================================================================
#  OFFLINE LLM (Mistral-7B-Instruct)
# =============================================================================

class OfflineLLM:
    """
    LLM offline utilisant Mistral-7B-Instruct.

    Thread-safe avec singleton pattern.
    Supporte la generation de texte avec streaming.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, log=None):
        # Double-check locking pour thread-safety
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._log = log or logger
            self.model_path = model_path or OFFLINE_MODELS_CONFIG["llm"]["path"]
            self.model = None
            self.tokenizer = None
            self.device = None
            self.gpu_info = None
            self.config = None
            self._model_lock = threading.Lock()
            self._initialized = True

    def _ensure_loaded(self):
        """Charge le modele si pas deja fait."""
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            # Verifier que le chemin existe
            if not os.path.exists(self.model_path):
                self._log.error(f"[OFFLINE-LLM] Chemin modele introuvable: {self.model_path}")
                raise FileNotFoundError(
                    f"Modele Mistral-7B introuvable: {self.model_path}\n"
                    f"Verifiez que le modele est telecharge dans le bon repertoire."
                )

            self._log.info(f"[OFFLINE-LLM] Chargement Mistral-7B depuis {self.model_path}...")

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                self.gpu_info = detect_gpu()
                self.config = get_optimal_config(self.gpu_info)

                self.device = "cuda" if self.gpu_info.available else "cpu"

                # Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Configuration de quantification si necessaire
                quantization_config = None

                if self.config.get("load_in_4bit") and self.gpu_info.available:
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        self._log.info("[OFFLINE-LLM] Quantification 4-bit activee")
                    except Exception as e:
                        self._log.warning(f"[OFFLINE-LLM] Quantification 4-bit non disponible: {e}")
                        quantization_config = None

                elif self.config.get("load_in_8bit") and self.gpu_info.available:
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                        self._log.info("[OFFLINE-LLM] Quantification 8-bit activee")
                    except Exception as e:
                        self._log.warning(f"[OFFLINE-LLM] Quantification 8-bit non disponible: {e}")
                        quantization_config = None

                # Charger le modele
                load_kwargs = {
                    "local_files_only": True,
                    "trust_remote_code": True,
                    "device_map": self.config.get("device_map", "auto") if self.gpu_info.available else None,
                }

                if quantization_config:
                    load_kwargs["quantization_config"] = quantization_config
                elif self.gpu_info.available:
                    load_kwargs["torch_dtype"] = torch.float16

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs,
                )

                if not self.gpu_info.available:
                    self.model = self.model.to("cpu")

                self.model.eval()

                self._log.info(f"[OFFLINE-LLM] Mistral-7B charge sur {self.device}")

            except Exception as e:
                self._log.error(f"[OFFLINE-LLM] Erreur chargement: {e}")
                raise RuntimeError(f"Impossible de charger Mistral-7B: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2000,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Genere du texte a partir d'un prompt.

        Args:
            prompt: Prompt d'entree
            max_new_tokens: Nombre max de tokens a generer
            temperature: Temperature de sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Si True, utilise le sampling

        Returns:
            Texte genere
        """
        import torch

        self._ensure_loaded()

        try:
            with self._model_lock:
                with torch.no_grad():
                    # Tokenizer
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.get("max_length", 2048),
                    )

                    if self.gpu_info and self.gpu_info.available:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decoder
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )

                    return generated_text.strip()

        except RuntimeError as e:
            # Gestion CUDA OOM
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._log.error(f"[OFFLINE-LLM] CUDA OOM lors de la generation: {e}")
                # Vider le cache GPU
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return "[ERREUR] Memoire GPU insuffisante pour generer une reponse. Essayez avec un prompt plus court."
            raise
        except Exception as e:
            self._log.error(f"[OFFLINE-LLM] Erreur generation: {e}")
            return f"[ERREUR] Impossible de generer une reponse: {e}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """
        Generation en mode chat avec messages structures.

        Args:
            messages: Liste de messages [{"role": "user/assistant/system", "content": "..."}]
            max_new_tokens: Nombre max de tokens
            temperature: Temperature

        Returns:
            Reponse generee
        """
        # Formater les messages pour Mistral Instruct
        formatted_prompt = self._format_chat_messages(messages)
        return self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Formate les messages au format Mistral Instruct."""
        formatted = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted += f"<s>[INST] {content}\n"
            elif role == "user":
                if formatted:
                    formatted += f" {content} [/INST]"
                else:
                    formatted += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>"

        # Ajouter le debut de la reponse assistant si necessaire
        if not formatted.endswith("[/INST]") and not formatted.endswith("</s>"):
            pass  # La generation commencera automatiquement

        return formatted


# =============================================================================
#  OFFLINE OCR (Donut-base)
# =============================================================================

class OfflineOCR:
    """
    OCR offline utilisant Donut-base.

    Thread-safe avec singleton pattern.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None, log=None):
        # Double-check locking pour thread-safety
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._log = log or logger
            self.model_path = model_path or OFFLINE_MODELS_CONFIG["ocr"]["path"]
            self.model = None
            self.processor = None
            self.device = None
            self._model_lock = threading.Lock()
            self._initialized = True

    def _ensure_loaded(self):
        """Charge le modele si pas deja fait."""
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            self._log.info(f"[OFFLINE-OCR] Chargement Donut-base depuis {self.model_path}...")

            try:
                import torch
                from transformers import DonutProcessor, VisionEncoderDecoderModel

                gpu_info = detect_gpu()
                self.device = "cuda" if gpu_info.available else "cpu"

                self.processor = DonutProcessor.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )

                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )

                self.model = self.model.to(self.device)
                self.model.eval()

                self._log.info(f"[OFFLINE-OCR] Donut-base charge sur {self.device}")

            except Exception as e:
                self._log.error(f"[OFFLINE-OCR] Erreur chargement: {e}")
                raise RuntimeError(f"Impossible de charger Donut-base: {e}")

    def extract_text(self, image) -> str:
        """
        Extrait le texte d'une image.

        Args:
            image: Image PIL ou chemin vers l'image

        Returns:
            Texte extrait
        """
        import torch
        from PIL import Image

        self._ensure_loaded()

        # Charger l'image si c'est un chemin
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not hasattr(image, "convert"):
            raise ValueError("L'image doit etre un objet PIL.Image ou un chemin")
        else:
            image = image.convert("RGB")

        try:
            with self._model_lock:
                with torch.no_grad():
                    # Preparer l'image
                    pixel_values = self.processor(
                        images=image,
                        return_tensors="pt",
                    ).pixel_values

                    pixel_values = pixel_values.to(self.device)

                    # Generer
                    task_prompt = "<s_cord-v2>"
                    decoder_input_ids = self.processor.tokenizer(
                        task_prompt,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids.to(self.device)

                    outputs = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.model.decoder.config.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    )

                    # Decoder
                    sequence = self.processor.batch_decode(outputs)[0]
                    sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
                    sequence = sequence.replace(self.processor.tokenizer.pad_token, "")

                    # Parser le resultat JSON si possible
                    try:
                        sequence = self.processor.token2json(sequence)
                        if isinstance(sequence, dict):
                            # Extraire le texte du JSON
                            text_parts = []
                            self._extract_text_from_dict(sequence, text_parts)
                            return " ".join(text_parts)
                    except Exception:
                        pass

                    return sequence.strip()

        except RuntimeError as e:
            # Gestion CUDA OOM
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                self._log.error(f"[OFFLINE-OCR] CUDA OOM lors de l'OCR: {e}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return "[ERREUR] Memoire GPU insuffisante pour l'OCR"
            raise
        except Exception as e:
            self._log.error(f"[OFFLINE-OCR] Erreur extraction texte: {e}")
            return ""

    def _extract_text_from_dict(self, d: Dict, result: List[str]):
        """Extrait recursivement le texte d'un dict."""
        for key, value in d.items():
            if isinstance(value, str):
                result.append(value)
            elif isinstance(value, dict):
                self._extract_text_from_dict(value, result)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        result.append(item)
                    elif isinstance(item, dict):
                        self._extract_text_from_dict(item, result)


# =============================================================================
#  FACTORY FUNCTIONS
# =============================================================================

def get_offline_embeddings(log=None) -> OfflineEmbeddings:
    """Retourne l'instance singleton des embeddings offline."""
    return OfflineEmbeddings(log=log)


def get_offline_reranker(log=None) -> OfflineReranker:
    """Retourne l'instance singleton du reranker offline."""
    return OfflineReranker(log=log)


def get_offline_llm(log=None) -> OfflineLLM:
    """Retourne l'instance singleton du LLM offline."""
    return OfflineLLM(log=log)


def get_offline_ocr(log=None) -> OfflineOCR:
    """Retourne l'instance singleton de l'OCR offline."""
    return OfflineOCR(log=log)


def cleanup_models():
    """Libere la memoire des modeles charges."""
    import gc

    # Nettoyer explicitement les modeles avant de reset les singletons
    for cls in [OfflineEmbeddings, OfflineReranker, OfflineLLM, OfflineOCR]:
        if cls._instance is not None:
            try:
                instance = cls._instance
                # Liberer les attributs du modele
                if hasattr(instance, 'model') and instance.model is not None:
                    del instance.model
                    instance.model = None
                if hasattr(instance, 'tokenizer') and instance.tokenizer is not None:
                    del instance.tokenizer
                    instance.tokenizer = None
                if hasattr(instance, 'processor') and instance.processor is not None:
                    del instance.processor
                    instance.processor = None
            except Exception as e:
                logger.warning(f"[OFFLINE] Erreur nettoyage {cls.__name__}: {e}")

    # Reset singletons
    OfflineEmbeddings._instance = None
    OfflineReranker._instance = None
    OfflineLLM._instance = None
    OfflineOCR._instance = None

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache si disponible
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    logger.info("[OFFLINE] Modeles decharges et memoire liberee")


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Retourne les informations sur la memoire GPU.

    Returns:
        Dict avec memoire totale, utilisee et libre en GB
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - allocated

            return {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "free_gb": round(free, 2),
            }
    except Exception:
        pass

    return {"total_gb": 0, "allocated_gb": 0, "cached_gb": 0, "free_gb": 0}


def clear_gpu_cache():
    """
    Vide le cache GPU pour liberer de la memoire.
    Utile avant de charger un gros modele.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("[GPU] Cache GPU vide")
    except Exception as e:
        logger.warning(f"[GPU] Erreur vidage cache: {e}")


def check_gpu_memory_sufficient(required_gb: float = 4.0) -> bool:
    """
    Verifie si la memoire GPU disponible est suffisante.

    Args:
        required_gb: Memoire requise en GB

    Returns:
        True si suffisant
    """
    info = get_gpu_memory_info()
    return info.get("free_gb", 0) >= required_gb


def load_model_with_fallback(
    load_func,
    model_name: str,
    required_vram_gb: float = 4.0,
    log=None,
):
    """
    Charge un modele avec fallback automatique vers CPU si GPU insuffisant.

    Args:
        load_func: Fonction de chargement du modele (prend device en parametre)
        model_name: Nom du modele pour les logs
        required_vram_gb: VRAM minimum requise
        log: Logger

    Returns:
        Modele charge sur le device optimal
    """
    _log = log or logger

    # Essayer d'abord sur GPU
    gpu_info = detect_gpu()

    if gpu_info.available:
        # Verifier la VRAM disponible
        if gpu_info.vram_free_gb >= required_vram_gb:
            try:
                _log.info(f"[OFFLINE] Chargement {model_name} sur GPU...")
                model = load_func(device="cuda")
                _log.info(f"[OFFLINE] {model_name} charge sur GPU avec succes")
                return model, "cuda"
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    _log.warning(f"[OFFLINE] OOM GPU, fallback vers CPU: {e}")
                    clear_gpu_cache()
                else:
                    raise
        else:
            _log.warning(
                f"[OFFLINE] VRAM insuffisante ({gpu_info.vram_free_gb:.1f}GB < {required_vram_gb}GB), "
                f"fallback vers CPU"
            )

    # Fallback CPU
    _log.info(f"[OFFLINE] Chargement {model_name} sur CPU...")
    try:
        model = load_func(device="cpu")
        _log.info(f"[OFFLINE] {model_name} charge sur CPU")
        return model, "cpu"
    except Exception as e:
        _log.error(f"[OFFLINE] Erreur chargement {model_name} sur CPU: {e}")
        raise


# =============================================================================
#  DIMENSION VERIFICATION
# =============================================================================

# Dimension attendue pour la compatibilite avec Snowflake Arctic
EXPECTED_EMBEDDING_DIMENSION = 1024


def verify_embedding_dimension(embeddings: np.ndarray, log=None) -> bool:
    """
    Verifie que les embeddings ont la bonne dimension.

    Args:
        embeddings: Array numpy d'embeddings
        log: Logger

    Returns:
        True si dimension correcte
    """
    _log = log or logger

    if embeddings.ndim != 2:
        _log.error(f"[EMB-DIM] Attendu 2D, obtenu {embeddings.ndim}D")
        return False

    dim = embeddings.shape[1]
    if dim != EXPECTED_EMBEDDING_DIMENSION:
        _log.error(
            f"[EMB-DIM] Dimension incorrecte: {dim} vs attendu {EXPECTED_EMBEDDING_DIMENSION}"
        )
        return False

    return True


def pad_or_truncate_embedding(
    embedding: np.ndarray,
    target_dim: int = EXPECTED_EMBEDDING_DIMENSION,
) -> np.ndarray:
    """
    Ajuste la dimension d'un embedding par padding ou troncature.

    Args:
        embedding: Embedding source
        target_dim: Dimension cible

    Returns:
        Embedding ajuste
    """
    current_dim = len(embedding)

    if current_dim == target_dim:
        return embedding
    elif current_dim > target_dim:
        # Tronquer
        return embedding[:target_dim]
    else:
        # Padding avec zeros
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:current_dim] = embedding
        return padded


# =============================================================================
#  HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def embed_in_batches_offline(
    texts: List[str],
    role: str,
    batch_size: int,
    log=None,
    timeout_per_batch: float = 60.0,
) -> np.ndarray:
    """
    Version offline de embed_in_batches compatible avec models_utils.py.

    Args:
        texts: Liste de textes a encoder
        role: "query" ou "passage"
        batch_size: Taille des batchs (utilise pour le logging)
        log: Logger
        timeout_per_batch: Timeout par batch en secondes

    Returns:
        Array numpy d'embeddings normalises (n_texts, 1024)
    """
    import time

    _log = log or logger

    if not texts:
        return np.array([], dtype=np.float32).reshape(0, EXPECTED_EMBEDDING_DIMENSION)

    # Valider batch_size pour eviter division par zero ou boucle infinie
    if batch_size <= 0:
        _log.warning(f"[OFFLINE-EMB] batch_size invalide ({batch_size}), utilisation de 32")
        batch_size = 32

    try:
        embeddings_client = get_offline_embeddings(log=_log)
    except Exception as e:
        _log.error(f"[OFFLINE-EMB] Erreur initialisation client: {e}")
        raise RuntimeError(f"Impossible d'initialiser les embeddings offline: {e}")

    _log.info(f"[OFFLINE-EMB] Embedding {len(texts)} textes en mode offline (role={role})")

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    failed_batches = 0
    failed_texts = 0

    # Traiter par batchs pour eviter les problemes memoire
    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_start = time.time()

        try:
            if role == "query":
                batch_emb = embeddings_client.embed_queries(batch)
            else:
                batch_emb = embeddings_client.embed_documents(batch)

            elapsed = time.time() - batch_start

            if elapsed > timeout_per_batch:
                _log.warning(
                    f"[OFFLINE-EMB] Batch {batch_idx + 1}/{total_batches} lent: {elapsed:.1f}s"
                )

            all_embeddings.extend(batch_emb)
            _log.debug(
                f"[OFFLINE-EMB] Batch {batch_idx + 1}/{total_batches}: "
                f"{len(batch)} textes en {elapsed:.2f}s"
            )

        except Exception as e:
            failed_batches += 1
            failed_texts += len(batch)
            _log.error(f"[OFFLINE-EMB] Erreur batch {batch_idx + 1}: {e}")
            # Ajouter des embeddings vides pour ce batch (fallback)
            for _ in batch:
                all_embeddings.append([0.0] * EXPECTED_EMBEDDING_DIMENSION)

    # Verifier le taux d'echec
    if failed_batches > 0:
        failure_rate = failed_texts / len(texts)
        _log.warning(
            f"[OFFLINE-EMB] {failed_batches}/{total_batches} batches echoues "
            f"({failed_texts}/{len(texts)} textes, {failure_rate:.1%})"
        )
        # Si plus de 50% des textes ont echoue, lever une exception
        if failure_rate > 0.5:
            raise RuntimeError(
                f"Trop d'erreurs d'embedding: {failed_texts}/{len(texts)} textes echoues ({failure_rate:.1%})"
            )

    result = np.array(all_embeddings, dtype=np.float32)

    # Verifier la dimension
    if not verify_embedding_dimension(result, log=_log):
        _log.warning("[OFFLINE-EMB] Dimension incorrecte, tentative de correction")
        # Tenter de corriger
        if result.ndim == 2 and result.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
            corrected = np.zeros((result.shape[0], EXPECTED_EMBEDDING_DIMENSION), dtype=np.float32)
            min_dim = min(result.shape[1], EXPECTED_EMBEDDING_DIMENSION)
            corrected[:, :min_dim] = result[:, :min_dim]
            result = corrected

    # Verifier la normalisation L2
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(np.abs(norms - 1.0) > 0.01):
        _log.debug("[OFFLINE-EMB] Re-normalisation L2 des embeddings")
        result = result / (norms + 1e-12)

    _log.info(f"[OFFLINE-EMB] Termine: shape={result.shape}")
    return result


def call_llm_offline(
    question: str,
    context: str,
    log=None,
) -> str:
    """
    Version offline de call_dallem_chat compatible avec models_utils.py.

    Args:
        question: Question de l'utilisateur
        context: Contexte documentaire
        log: Logger

    Returns:
        Reponse generee
    """
    _log = log or logger

    if not question:
        _log.warning("[OFFLINE-LLM] Question vide")
        return "Veuillez poser une question."

    if not context:
        _log.warning("[OFFLINE-LLM] Contexte vide")
        context = "Aucun contexte fourni."

    llm = get_offline_llm(log=_log)

    system_msg = (
        "Tu es un assistant expert en reglementation aeronautique EASA. "
        "Tu maitrises parfaitement les documents de certification:\n"
        "- CS (Certification Specifications): exigences reglementaires obligatoires\n"
        "- AMC (Acceptable Means of Compliance): moyens acceptables pour demontrer la conformite\n"
        "- GM (Guidance Material): explications et interpretations non contraignantes\n"
        "Tu dois repondre en te basant UNIQUEMENT sur le CONTEXTE fourni."
    )

    user_msg = f"""
=== CONTEXTE DOCUMENTAIRE ===
{context}
=== FIN DU CONTEXTE ===

QUESTION : {question}

INSTRUCTIONS :
- Reponds dans la meme langue que la question (francais si question en francais, anglais sinon)
- Cite TOUJOURS les references exactes (ex: "CS 25.613", "AMC1 25.1309") presentes dans le contexte
- Distingue clairement ce qui est une exigence (CS/shall) de ce qui est un moyen de conformite (AMC) ou une guidance (GM)
- Pour les termes "shall", "must", "may", "should": precise le niveau d'obligation
- Si le contexte ne contient AUCUNE information pertinente, reponds : "Je n'ai pas l'information pour repondre a cette question."
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    _log.info("[OFFLINE-LLM] Generation de reponse RAG en mode offline")

    try:
        response = llm.chat(messages, max_new_tokens=2000, temperature=0.3)
        _log.info(f"[OFFLINE-LLM] Reponse generee: {len(response)} caracteres")
        return response
    except Exception as e:
        _log.error(f"[OFFLINE-LLM] Erreur generation: {e}")
        return f"Erreur lors de la generation de la reponse: {e}"


def rerank_offline(
    query: str,
    sources: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    log=None,
) -> List[Dict[str, Any]]:
    """
    Version offline du reranking compatible avec advanced_search.py.

    Args:
        query: Requete
        sources: Liste de sources avec "text"
        top_k: Nombre de resultats
        log: Logger

    Returns:
        Sources reordonnees avec "rerank_score"
    """
    _log = log or logger

    if not sources:
        return sources

    if not query:
        _log.warning("[OFFLINE-RR] Query vide, retour sources sans reranking")
        return sources

    reranker = get_offline_reranker(log=_log)

    documents = [s.get("text", "") for s in sources]

    _log.info(f"[OFFLINE-RR] Reranking {len(documents)} documents")

    results = reranker.rerank(query, documents, top_k=top_k)

    # Reconstruire les sources avec les scores
    reranked = []
    for idx, score in results:
        src = sources[idx].copy()
        src["rerank_score"] = score
        src["original_rank"] = idx
        reranked.append(src)

    _log.info(f"[OFFLINE-RR] Reranking termine: {len(reranked)} resultats")
    return reranked


# =============================================================================
#  STATUS & INFO
# =============================================================================

def get_offline_status() -> Dict[str, Any]:
    """
    Retourne le statut complet du mode offline.

    Returns:
        Dict avec GPU info, modeles disponibles, chemins
    """
    gpu_info = detect_gpu()
    models_status = check_offline_models_available()
    storage_path = get_storage_path()

    # Enrichir les infos modeles avec noms et chemins
    models_details = {}
    for model_key, available in models_status.items():
        config = OFFLINE_MODELS_CONFIG.get(model_key, {})
        models_details[model_key] = {
            "available": available,
            "name": config.get("name", model_key),
            "path": config.get("path", "N/A"),
            "type": config.get("type", "unknown"),
        }

    return {
        "gpu": {
            "available": gpu_info.available,
            "device": gpu_info.device_name,
            "vram_total_gb": gpu_info.vram_total_gb,
            "vram_free_gb": gpu_info.vram_free_gb,
            "cuda_version": gpu_info.cuda_version,
            "tier": gpu_info.tier.value,
        },
        "models": models_status,
        "models_details": models_details,
        "storage": {
            "active_path": storage_path,
            "primary_accessible": _is_path_accessible(STORAGE_PATHS["primary"]),
            "fallback_accessible": _is_path_accessible(STORAGE_PATHS["fallback"]),
        },
        "ready": all(models_status.values()),
    }


def print_offline_status(log=None):
    """Affiche le statut du mode offline."""
    _log = log or logger

    status = get_offline_status()

    print("\n" + "=" * 60)
    print("MODE OFFLINE - STATUT")
    print("=" * 60)

    # GPU
    gpu = status["gpu"]
    if gpu["available"]:
        print(f"\n GPU: {gpu['device']}")
        print(f"   VRAM: {gpu['vram_free_gb']:.1f}/{gpu['vram_total_gb']:.1f} GB")
        print(f"   CUDA: {gpu['cuda_version']}")
        print(f"   Tier: {gpu['tier']}")
    else:
        print(f"\n GPU: Non disponible (mode CPU)")

    # Modeles
    print(f"\n MODELES:")
    for model, available in status["models"].items():
        icon = "[OK]" if available else "[X]"
        path = OFFLINE_MODELS_CONFIG[model]["path"]
        print(f"   {icon} {model}: {path}")

    # Stockage
    print(f"\n STOCKAGE:")
    print(f"   Chemin actif: {status['storage']['active_path']}")
    print(f"   N:\\ accessible: {status['storage']['primary_accessible']}")
    print(f"   D:\\ accessible: {status['storage']['fallback_accessible']}")

    # Status global
    print(f"\n{'=' * 60}")
    if status["ready"]:
        print(" MODE OFFLINE: PRET")
    else:
        print(" MODE OFFLINE: NON DISPONIBLE (modeles manquants)")
    print("=" * 60 + "\n")

    return status


if __name__ == "__main__":
    # Test du module
    logging.basicConfig(level=logging.INFO)
    print_offline_status()
