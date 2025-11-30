"""
Ingestion Pipeline - Orchestration optimisée du traitement des fichiers

Architecture en 2 phases pour minimiser les appels réseau:

PHASE 1 - LOCALE (0 appel réseau sauf OCR si nécessaire):
  1. Téléchargement/copie des fichiers vers répertoire TEMP local
  2. Extraction des pièces jointes vers le même répertoire local
  3. Extraction texte (pdfplumber/pymupdf/pdfminer)
  4. Évaluation qualité + fallback OCR DALLEM si nécessaire
  5. Stockage des fichiers OCR dans le répertoire TEMP
  6. Chunking sémantique

PHASE 2 - RÉSEAU (appels minimisés):
  7. Embeddings en batch (1 appel / 32 chunks)
  8. Insertion FAISS (local après phase réseau)

Usage:
    from ingestion_pipeline import OptimizedIngestionPipeline

    pipeline = OptimizedIngestionPipeline(
        db_path="/path/to/faiss/db",
        collection_name="my_collection",
    )

    # Ingestion depuis un CSV de fichiers
    result = pipeline.ingest_from_csv(
        csv_path="/path/to/files.csv",
        file_column="path",
    )

    # Ou ingestion directe de fichiers
    result = pipeline.ingest_files(
        file_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
    )
"""

import os
import gc
import csv
import uuid
import time
import shutil
import tempfile
import logging
import requests
from urllib.parse import urlparse, unquote
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import multiprocessing

# Détection RAM pour configuration automatique
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
    TOTAL_RAM_GB = psutil.virtual_memory().total / (1024 ** 3)
    AVAILABLE_RAM_GB = psutil.virtual_memory().available / (1024 ** 3)
except ImportError:
    # psutil non disponible - utiliser valeur conservatrice par défaut
    TOTAL_RAM_GB = 8  # Valeur conservatrice si psutil non disponible
    AVAILABLE_RAM_GB = 4
    print("⚠️  psutil non installé - mode conservateur activé (8 Go par défaut)")
    print("   Pour une détection précise: pip install psutil")


def get_optimal_config(ram_gb: float = None) -> dict:
    """
    Calcule la configuration optimale en fonction de la RAM disponible.

    Utilise le MINIMUM entre RAM totale et RAM disponible * 2 pour être prudent.

    Args:
        ram_gb: RAM en Go (si None, utilise la RAM système détectée)

    Returns:
        Dict avec les paramètres optimaux
    """
    if ram_gb is None:
        # Utiliser le minimum entre RAM totale et RAM disponible * 2
        # Cela permet de tenir compte des autres applications en cours
        ram_gb = min(TOTAL_RAM_GB, AVAILABLE_RAM_GB * 2)

    # Configuration par paliers de RAM
    if ram_gb <= 8:
        # PC très limité (8 Go) - Mode ultra conservateur
        return {
            "low_memory": True,
            "max_workers": 1,
            "embedding_batch_size": 4,
            "streaming_mode": True,
            "aggressive_gc": True,
            "description": "Ultra-conservateur (≤8 Go RAM)",
        }
    elif ram_gb <= 12:
        # PC limité (8-12 Go) - Mode conservateur
        return {
            "low_memory": True,
            "max_workers": 2,
            "embedding_batch_size": 8,
            "streaming_mode": True,
            "aggressive_gc": True,
            "description": "Conservateur (8-12 Go RAM)",
        }
    elif ram_gb <= 16:
        # PC standard (12-16 Go) - Mode équilibré
        return {
            "low_memory": False,
            "max_workers": 4,
            "embedding_batch_size": 16,
            "streaming_mode": False,
            "aggressive_gc": True,
            "description": "Équilibré (12-16 Go RAM)",
        }
    elif ram_gb <= 32:
        # PC performant (16-32 Go) - Mode performance
        return {
            "low_memory": False,
            "max_workers": 6,
            "embedding_batch_size": 32,
            "streaming_mode": False,
            "aggressive_gc": False,
            "description": "Performance (16-32 Go RAM)",
        }
    else:
        # Workstation (32+ Go) - Mode maximum
        return {
            "low_memory": False,
            "max_workers": min(8, multiprocessing.cpu_count()),
            "embedding_batch_size": 64,
            "streaming_mode": False,
            "aggressive_gc": False,
            "description": "Maximum (32+ Go RAM)",
        }


# Configuration optimale pré-calculée au démarrage
OPTIMAL_CONFIG = get_optimal_config()

from langdetect import detect

from core.models_utils import (
    make_logger,
    EMBED_MODEL,
    BATCH_SIZE,
    DirectOpenAIEmbeddings,
    embed_in_batches,
    SNOWFLAKE_API_KEY,
    SNOWFLAKE_API_BASE,
    create_http_client,
)

from core.faiss_store import FaissStore, build_faiss_store
from processing.pdf_processing import (
    extract_text_from_pdf,
    extract_attachments_from_pdf,
    extract_text_with_dallem,
    assess_extraction_quality,
    log_extraction_quality,
    LLM_OCR_AVAILABLE,
)
from processing.docx_processing import extract_text_from_docx
from processing.xml_processing import extract_text_from_xml, XMLParseConfig

# PPTX Processing (Phase 4)
try:
    from processing.pptx_processing import (
        extract_pptx,
        process_pptx_with_attachments,
        is_pptx_processing_available,
    )
    PPTX_AVAILABLE = is_pptx_processing_available()
except ImportError:
    PPTX_AVAILABLE = False
from chunking.chunking import (
    simple_chunk,
    chunk_easa_sections,
    smart_chunk_generic,
    augment_chunks,
    _calculate_content_density,
    add_cross_references_to_chunks,
)
from chunking.easa_sections import split_easa_sections

# Chunking sémantique (Phase 3)
try:
    from chunking.semantic_chunking import semantic_chunk, AdaptiveSemanticChunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class FileInfo:
    """Informations sur un fichier à traiter."""
    original_path: str
    local_path: Optional[str] = None
    filename: str = ""
    extension: str = ""
    size: int = 0
    is_attachment: bool = False
    parent_file: Optional[str] = None

    def __post_init__(self):
        if not self.filename:
            self.filename = os.path.basename(self.original_path)
        if not self.extension:
            self.extension = os.path.splitext(self.filename)[1].lower()


@dataclass
class ExtractionResult:
    """Résultat de l'extraction de texte d'un fichier."""
    file_info: FileInfo
    text: str = ""
    language: str = ""
    quality_score: float = 1.0
    method: str = "classic"
    llm_ocr_used: bool = False
    error: Optional[str] = None

    # Métadonnées additionnelles
    attachments: List['ExtractionResult'] = field(default_factory=list)


@dataclass
class ChunkInfo:
    """Information sur un chunk préparé pour l'embedding."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    faiss_id: str


@dataclass
class PipelineStats:
    """Statistiques du pipeline."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_attachments: int = 0
    ocr_files: int = 0
    total_chunks: int = 0
    embedding_batches: int = 0
    total_time: float = 0.0
    download_time: float = 0.0
    extraction_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    faiss_time: float = 0.0


# =============================================================================
#  TYPE ALIASES
# =============================================================================

ProgressCallback = Callable[[float, str], None]


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def _make_progress_bar(progress: float, width: int = 30, filled: str = "█", empty: str = "░") -> str:
    """
    Crée une barre de progression ASCII.

    Args:
        progress: Valeur entre 0 et 1
        width: Largeur de la barre en caractères
        filled: Caractère pour la partie remplie
        empty: Caractère pour la partie vide

    Returns:
        Barre de progression formatée
    """
    progress = max(0, min(1, progress))
    filled_width = int(progress * width)
    bar = filled * filled_width + empty * (width - filled_width)
    pct = int(progress * 100)
    return f"[{bar}] {pct:3d}%"


def _format_duration(seconds: float) -> str:
    """Formate une durée en format lisible."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _format_size(size_bytes: int) -> str:
    """Formate une taille en format lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# =============================================================================
#  OPTIMIZED INGESTION PIPELINE
# =============================================================================

class OptimizedIngestionPipeline:
    """
    Pipeline d'ingestion optimisé pour minimiser les appels réseau.

    Architecture:
    - Phase 1 (locale): Téléchargement → Extraction PJ → Extraction texte → OCR fallback → Chunking
    - Phase 2 (réseau): Embeddings batch → FAISS insert

    Attributes:
        db_path: Chemin de la base FAISS
        collection_name: Nom de la collection
        temp_dir: Répertoire temporaire pour le traitement local
        quality_threshold: Seuil de qualité pour déclencher l'OCR LLM (0-1)
        chunk_size: Taille cible des chunks
        use_semantic_chunking: Utiliser le chunking sémantique (Phase 3)
        use_easa_sections: Détecter et découper selon les sections EASA
        force_ocr: Forcer l'OCR LLM pour tous les PDF
        low_memory: Mode faible mémoire pour PC 8 Go RAM
        max_workers: Nombre de workers parallèles (1-8)
        embedding_batch_size: Taille des batches d'embeddings (8-64)
        streaming_mode: Traiter fichier par fichier (libère la mémoire)
        aggressive_gc: Libération mémoire agressive après chaque étape
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        temp_dir: Optional[str] = None,
        quality_threshold: float = 0.5,
        chunk_size: int = 1500,
        use_semantic_chunking: bool = True,
        use_easa_sections: bool = True,
        force_ocr: bool = False,
        log=None,
        # Options low memory (None = détection automatique basée sur RAM)
        low_memory: Optional[bool] = None,
        max_workers: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        streaming_mode: Optional[bool] = None,
        aggressive_gc: Optional[bool] = None,
        # Mapping des chemins logiques (pour Confluence, URLs, etc.)
        logical_paths: Optional[Dict[str, str]] = None,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.quality_threshold = quality_threshold
        self.chunk_size = chunk_size
        self.use_semantic_chunking = use_semantic_chunking and SEMANTIC_CHUNKING_AVAILABLE
        self.use_easa_sections = use_easa_sections
        self.force_ocr = force_ocr
        self.log = log or make_logger(debug=False)
        # Mapping {chemin_fichier_réel: chemin_logique_ou_URL}
        self.logical_paths = logical_paths or {}

        # Configuration automatique basée sur la RAM disponible
        # Utilise OPTIMAL_CONFIG calculé au démarrage, sauf si explicitement spécifié
        self._auto_config = OPTIMAL_CONFIG.copy()
        self._config_mode = self._auto_config["description"]

        # Appliquer les valeurs : explicites > auto-détection
        self.low_memory = low_memory if low_memory is not None else self._auto_config["low_memory"]
        self.max_workers = max_workers if max_workers is not None else self._auto_config["max_workers"]
        self.embedding_batch_size = embedding_batch_size if embedding_batch_size is not None else self._auto_config["embedding_batch_size"]
        self.streaming_mode = streaming_mode if streaming_mode is not None else self._auto_config["streaming_mode"]
        self.aggressive_gc = aggressive_gc if aggressive_gc is not None else self._auto_config["aggressive_gc"]

        # Log de la configuration auto-détectée
        print(f"\n🔧 AUTO-CONFIG: {self._config_mode}")
        print(f"   RAM détectée: {TOTAL_RAM_GB:.1f} Go (disponible: {AVAILABLE_RAM_GB:.1f} Go)")

        # Créer le répertoire temporaire
        if temp_dir:
            self.temp_dir = temp_dir
            self._owns_temp_dir = False
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="rag_ingestion_")
            self._owns_temp_dir = True

        self.temp_files_dir = os.path.join(self.temp_dir, "files")
        self.temp_attachments_dir = os.path.join(self.temp_dir, "attachments")
        self.temp_ocr_dir = os.path.join(self.temp_dir, "ocr_cache")

        os.makedirs(self.temp_files_dir, exist_ok=True)
        os.makedirs(self.temp_attachments_dir, exist_ok=True)
        os.makedirs(self.temp_ocr_dir, exist_ok=True)

        self.log.info(f"[PIPELINE] Temp directory: {self.temp_dir}")

        # Diagnostic OCR LLM
        self._print_ocr_diagnostic()

        # Stats
        self.stats = PipelineStats()

    def _print_ocr_diagnostic(self):
        """Affiche l'état de la configuration OCR LLM et mémoire."""
        # Config mémoire
        print(f"\n{'─'*60}")
        print(f"💾 CONFIGURATION MÉMOIRE AUTO-DÉTECTÉE")
        print(f"{'─'*60}")
        print(f"  📊 Mode: {self._config_mode}")
        if PSUTIL_AVAILABLE:
            print(f"  💻 RAM totale: {TOTAL_RAM_GB:.1f} Go (détectée via psutil)")
        else:
            print(f"  💻 RAM totale: {TOTAL_RAM_GB:.1f} Go (valeur par défaut - psutil non installé)")
        print(f"  📈 RAM disponible: {AVAILABLE_RAM_GB:.1f} Go")
        print(f"{'─'*60}")
        print(f"  ⚙️  Paramètres appliqués:")
        print(f"     • Max workers: {self.max_workers}")
        print(f"     • Batch embeddings: {self.embedding_batch_size}")
        print(f"     • Streaming: {'✅ Oui' if self.streaming_mode else '❌ Non'}")
        print(f"     • GC agressif: {'✅ Oui' if self.aggressive_gc else '❌ Non'}")

        # Config OCR
        print(f"{'─'*60}")
        print(f"🔍 OCR LLM DIAGNOSTIC")
        print(f"{'─'*60}")
        print(f"  LLM_OCR_AVAILABLE: {LLM_OCR_AVAILABLE}")
        print(f"  Quality threshold: {self.quality_threshold}")
        print(f"  Force OCR: {self.force_ocr}")

        if LLM_OCR_AVAILABLE:
            try:
                from ocr.llm_ocr import DALLEM_CONFIG_AVAILABLE
                print(f"  DALLEM_CONFIG_AVAILABLE: {DALLEM_CONFIG_AVAILABLE}")
                if DALLEM_CONFIG_AVAILABLE:
                    from core.models_utils import DALLEM_API_BASE, DALLEM_API_KEY
                    print(f"  DALLEM_API_BASE: {DALLEM_API_BASE[:50]}...")
                    print(f"  DALLEM_API_KEY: {'✅ Set' if DALLEM_API_KEY and DALLEM_API_KEY != 'EMPTY' else '❌ Not set'}")
                    print(f"  ✅ OCR LLM is READY")
                else:
                    print(f"  ❌ DALLEM config not available in llm_ocr.py")
            except ImportError as e:
                print(f"  ❌ Import error: {e}")
        else:
            print(f"  ❌ LLM OCR module not available")
            print(f"     Check if llm_ocr.py is present and has no import errors")

        print(f"{'─'*60}\n")

    def cleanup(self):
        """Nettoie le répertoire temporaire avec rapport détaillé."""
        if self._owns_temp_dir and os.path.exists(self.temp_dir):
            # Compter les fichiers avant suppression
            files_count = 0
            total_size = 0
            for root, dirs, files in os.walk(self.temp_dir):
                for f in files:
                    files_count += 1
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

            size_mb = total_size / (1024 * 1024)
            self.log.info(
                f"[PIPELINE] 🧹 Cleaning up temp directory:\n"
                f"  📁 Location: {self.temp_dir}\n"
                f"  📄 Files: {files_count}\n"
                f"  💾 Size: {size_mb:.2f} MB"
            )

            try:
                shutil.rmtree(self.temp_dir)
                self.log.info(f"[PIPELINE] ✅ Temp directory cleaned successfully")
            except Exception as e:
                self.log.warning(f"[PIPELINE] ⚠️ Failed to cleanup temp dir: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    # =========================================================================
    #  PHASE 1: LOCAL PROCESSING
    # =========================================================================

    def _is_url(self, path: str) -> bool:
        """Vérifie si le chemin est une URL."""
        try:
            parsed = urlparse(path)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False

    def _download_url(self, url: str, local_path: str, timeout: int = 60) -> bool:
        """
        Télécharge un fichier depuis une URL vers un chemin local.

        Args:
            url: URL du fichier
            local_path: Chemin de destination
            timeout: Timeout en secondes

        Returns:
            True si téléchargement réussi
        """
        try:
            with requests.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            self.log.debug(f"[PIPELINE] URL downloaded: {url} -> {local_path}")
            return True

        except requests.RequestException as e:
            self.log.error(f"[PIPELINE] Failed to download URL {url}: {e}")
            return False

    def _get_filename_from_url(self, url: str) -> str:
        """Extrait le nom de fichier depuis une URL."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = os.path.basename(path)
        if not filename or "." not in filename:
            # Générer un nom si non extractible
            filename = f"download_{uuid.uuid4().hex[:8]}.pdf"
        return filename

    def download_to_temp(
        self,
        file_paths: List[str],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[FileInfo]:
        """
        ÉTAPE 1: Copie/télécharge les fichiers vers le répertoire TEMP local.

        Gère les chemins locaux, les chemins réseau et les URLs (http/https).

        Args:
            file_paths: Liste des chemins de fichiers (locaux, réseau ou URLs)
            progress_cb: Callback de progression

        Returns:
            Liste de FileInfo avec chemins locaux
        """
        # Afficher le header
        print(f"\n{'='*60}")
        print(f"📥 PHASE 1.1: DOWNLOAD TO TEMP")
        print(f"{'='*60}")
        print(f"  📁 Files to download: {len(file_paths)}")
        print(f"  📂 Temp directory: {self.temp_files_dir}")
        print(f"{'─'*60}")

        self.log.info(
            f"[PIPELINE] 📥 Phase 1.1: DOWNLOAD TO TEMP\n"
            f"[PIPELINE]   Files to download: {len(file_paths)}\n"
            f"[PIPELINE]   Temp directory: {self.temp_files_dir}"
        )
        start_time = time.time()

        file_infos: List[FileInfo] = []
        total = len(file_paths)
        url_count = 0
        local_count = 0
        total_size = 0

        if total == 0:
            print("  ⚠️ No files to download")
            self.log.info("[PIPELINE] ✅ No files to download")
            self.stats.download_time = 0
            return file_infos

        for i, path in enumerate(file_paths):
            progress = (i + 1) / total
            bar = _make_progress_bar(progress, width=25)

            if progress_cb:
                progress_cb(progress * 0.1, f"📥 Download {bar} {i+1}/{total}")

            try:
                # Déterminer le nom de fichier
                if self._is_url(path):
                    filename = self._get_filename_from_url(path)
                else:
                    filename = os.path.basename(path)

                local_path = os.path.join(self.temp_files_dir, filename)

                # Éviter les doublons
                if os.path.exists(local_path):
                    base, ext = os.path.splitext(filename)
                    local_path = os.path.join(self.temp_files_dir, f"{base}_{uuid.uuid4().hex[:6]}{ext}")

                # Télécharger ou copier selon le type de chemin
                if self._is_url(path):
                    # Télécharger depuis URL
                    print(f"  📡 [{i+1}/{total}] Downloading URL: {filename}...", end=" ", flush=True)
                    success = self._download_url(path, local_path)
                    if not success:
                        print("❌ FAILED")
                        file_infos.append(FileInfo(original_path=path, local_path=None))
                        continue
                    url_count += 1
                    file_size = os.path.getsize(local_path)
                    print(f"✅ ({_format_size(file_size)})")
                else:
                    # Copier depuis chemin local/réseau
                    if not os.path.exists(path):
                        print(f"  ❌ [{i+1}/{total}] File not found: {path}")
                        self.log.error(f"[PIPELINE] ❌ File not found: {path}")
                        file_infos.append(FileInfo(original_path=path, local_path=None))
                        continue

                    print(f"  📁 [{i+1}/{total}] Copying: {filename}...", end=" ", flush=True)
                    shutil.copy2(path, local_path)
                    local_count += 1
                    file_size = os.path.getsize(local_path)
                    print(f"✅ ({_format_size(file_size)})")

                file_info = FileInfo(
                    original_path=path,
                    local_path=local_path,
                    size=os.path.getsize(local_path),
                )
                file_infos.append(file_info)
                total_size += file_info.size

                self.log.debug(f"[PIPELINE] ✓ {filename} ({_format_size(file_info.size)})")

            except Exception as e:
                print(f"  ❌ [{i+1}/{total}] Error: {path} - {e}")
                self.log.error(f"[PIPELINE] Failed to download {path}: {e}")
                # Créer un FileInfo avec erreur
                file_infos.append(FileInfo(original_path=path, local_path=None))

        self.stats.download_time = time.time() - start_time
        success_count = len([f for f in file_infos if f.local_path])
        failed_count = len([f for f in file_infos if not f.local_path])

        # Afficher le récapitulatif
        print(f"{'─'*60}")
        print(f"✅ DOWNLOAD COMPLETED")
        print(f"{'─'*60}")
        print(f"  {_make_progress_bar(1.0, width=35)}")
        print(f"  ✓ Success: {success_count}/{total} files")
        print(f"  ✗ Failed:  {failed_count} files")
        print(f"  📡 URLs downloaded: {url_count}")
        print(f"  📁 Local files copied: {local_count}")
        print(f"  💾 Total size: {_format_size(total_size)}")
        print(f"  ⏱️  Duration: {_format_duration(self.stats.download_time)}")
        print(f"{'='*60}\n")

        self.log.info(
            f"[PIPELINE] ✅ Download completed: {success_count}/{total} files "
            f"({_format_size(total_size)}) in {_format_duration(self.stats.download_time)}"
        )

        return file_infos

    def extract_attachments(
        self,
        file_infos: List[FileInfo],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[FileInfo]:
        """
        ÉTAPE 2: Extrait les pièces jointes des PDF vers le répertoire local.

        Args:
            file_infos: Liste des fichiers à traiter
            progress_cb: Callback de progression

        Returns:
            Liste mise à jour avec les pièces jointes ajoutées
        """
        pdf_files = [f for f in file_infos if f.local_path and f.extension == ".pdf"]

        # Afficher le header
        print(f"\n{'='*60}")
        print(f"📎 PHASE 1.2: EXTRACT ATTACHMENTS")
        print(f"{'='*60}")
        print(f"  📄 Total files: {len(file_infos)}")
        print(f"  📕 PDFs to scan: {len(pdf_files)}")
        print(f"  📂 Output directory: {self.temp_attachments_dir}")
        print(f"{'─'*60}")

        self.log.info(
            f"[PIPELINE] 📎 Phase 1.2: EXTRACT ATTACHMENTS\n"
            f"[PIPELINE]   Total files: {len(file_infos)}, PDFs to scan: {len(pdf_files)}"
        )

        all_files: List[FileInfo] = []
        total = len(file_infos)

        if total == 0:
            print("  ⚠️ No files to process for attachments")
            self.log.info("[PIPELINE] ✅ No files to process for attachments")
            return all_files

        pdf_counter = 0  # Compteur O(1) au lieu de recalculer à chaque itération

        for i, file_info in enumerate(file_infos):
            progress = 0.1 + ((i + 1) / total) * 0.1
            bar = _make_progress_bar((i + 1) / total, width=20)
            if progress_cb:
                progress_cb(progress, f"📎 Attachments {bar} {i+1}/{total}")

            all_files.append(file_info)

            if file_info.local_path and file_info.extension == ".pdf":
                pdf_counter += 1
                pdf_idx = pdf_counter
                print(f"  🔍 [{pdf_idx}/{len(pdf_files)}] Scanning: {file_info.filename}...", end=" ", flush=True)

                try:
                    result = extract_attachments_from_pdf(
                        pdf_path=file_info.local_path,
                        output_dir=self.temp_attachments_dir,
                    )

                    attachments = result.get("attachments_paths", [])

                    if attachments:
                        print(f"📎 {len(attachments)} attachment(s)")
                        for att_path in attachments:
                            att_size = os.path.getsize(att_path) if os.path.exists(att_path) else 0
                            att_name = os.path.basename(att_path)
                            att_info = FileInfo(
                                original_path=att_path,
                                local_path=att_path,
                                is_attachment=True,
                                parent_file=file_info.original_path,
                                size=att_size,
                            )
                            all_files.append(att_info)
                            self.stats.total_attachments += 1
                            print(f"      ├─ {att_name} ({_format_size(att_size)})")
                    else:
                        print("✓ no attachments")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    self.log.warning(f"[PIPELINE] ⚠️ Failed to extract attachments: {e}")

        # Compter les PDFs avec/sans attachments
        pdfs_with_att = len([f for f in file_infos if f.local_path and f.extension == ".pdf" and
                           any(a.parent_file == f.original_path for a in all_files if a.is_attachment)])
        pdfs_without_att = len(pdf_files) - pdfs_with_att

        # Afficher le récapitulatif
        print(f"{'─'*60}")
        print(f"✅ ATTACHMENTS EXTRACTION COMPLETED")
        print(f"{'─'*60}")
        print(f"  {_make_progress_bar(1.0, width=35)}")
        print(f"  📕 PDFs scanned: {len(pdf_files)}")
        print(f"  📎 PDFs with attachments: {pdfs_with_att}")
        print(f"  📄 PDFs without attachments: {pdfs_without_att}")
        print(f"  📎 Total attachments extracted: {self.stats.total_attachments}")
        print(f"  📁 Total files to process: {len(all_files)}")
        print(f"{'='*60}\n")

        self.log.info(
            f"[PIPELINE] ✅ Attachments: {self.stats.total_attachments} extracted from {pdfs_with_att} PDFs"
        )
        return all_files

    def extract_text_local(
        self,
        file_infos: List[FileInfo],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[ExtractionResult]:
        """
        ÉTAPE 3-4: Extrait le texte des fichiers avec fallback OCR si nécessaire.

        Tout se passe localement sauf les appels OCR LLM si la qualité est < seuil.

        Args:
            file_infos: Liste des fichiers à traiter
            progress_cb: Callback de progression

        Returns:
            Liste de ExtractionResult
        """
        valid_files = [f for f in file_infos if f.local_path and os.path.exists(f.local_path)]
        # Utiliser max_workers configuré (mode low_memory = 1)
        max_workers = max(1, min(self.max_workers, len(valid_files)))

        self.log.info(
            f"[PIPELINE] ═══════════════════════════════════════════════════════\n"
            f"[PIPELINE] 📄 Phase 1.3: EXTRACT TEXT\n"
            f"[PIPELINE] ───────────────────────────────────────────────────────\n"
            f"[PIPELINE]   Files to process: {len(valid_files)}/{len(file_infos)}\n"
            f"[PIPELINE]   Parallel workers: {max_workers} {'(low memory)' if self.low_memory else ''}\n"
            f"[PIPELINE]   OCR threshold: {self.quality_threshold}\n"
            f"[PIPELINE]   Force OCR: {self.force_ocr}"
        )
        start_time = time.time()

        results: List[ExtractionResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._extract_single_file, f): f
                for f in valid_files
            }

            completed = 0
            total_valid = len(valid_files) or 1  # Évite division par zéro
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                completed += 1

                progress = 0.2 + (completed / total_valid) * 0.3
                bar = _make_progress_bar(completed / total_valid, width=20)
                if progress_cb:
                    progress_cb(progress, f"📄 Extract {bar} {completed}/{len(valid_files)}")

                try:
                    result = future.result()
                    results.append(result)

                    if result.llm_ocr_used:
                        self.stats.ocr_files += 1

                except Exception as e:
                    self.log.error(f"[PIPELINE] Extraction failed for {file_info.filename}: {e}")
                    results.append(ExtractionResult(
                        file_info=file_info,
                        error=str(e)
                    ))

                # Libération mémoire agressive après chaque fichier
                if self.aggressive_gc:
                    gc.collect()

        # Ajouter les fichiers qui n'avaient pas de chemin local
        for f in file_infos:
            if not f.local_path or not os.path.exists(f.local_path):
                results.append(ExtractionResult(
                    file_info=f,
                    error="File not found or not downloaded"
                ))

        self.stats.extraction_time = time.time() - start_time
        success_count = len([r for r in results if r.text])
        error_count = len([r for r in results if r.error])

        self.log.info(
            f"[PIPELINE] ✅ Text extraction completed:\n"
            f"[PIPELINE]   {_make_progress_bar(1.0, width=30)}\n"
            f"[PIPELINE]   ✓ Success: {success_count}/{len(file_infos)} files\n"
            f"[PIPELINE]   ✗ Errors: {error_count} files\n"
            f"[PIPELINE]   🔍 OCR used: {self.stats.ocr_files} files\n"
            f"[PIPELINE]   ⏱️ Duration: {_format_duration(self.stats.extraction_time)}"
        )

        return results

    def _extract_single_file(self, file_info: FileInfo) -> ExtractionResult:
        """Extrait le texte d'un seul fichier avec fallback OCR."""
        ext = file_info.extension
        path = file_info.local_path

        try:
            text = ""
            method = "classic"
            llm_ocr_used = False
            quality_score = 1.0

            if ext == ".pdf":
                # Pour les PDF: extraction avec évaluation qualité + fallback OCR
                if self.force_ocr and LLM_OCR_AVAILABLE:
                    # Mode forcé: toujours utiliser DALLEM OCR
                    result = extract_text_with_dallem(
                        path=path,
                        force_llm_ocr=True,
                        log=self.log,
                    )
                    text = result.get("text", "")
                    method = result.get("method", "dallem_ocr")
                    llm_ocr_used = True
                    quality_score = result.get("quality_score") or 0.8

                else:
                    # Mode intelligent: extraction classique puis OCR si nécessaire
                    text = extract_text_from_pdf(path)
                    quality = assess_extraction_quality(text)
                    quality_score = quality.get("quality_score", 1.0) if quality else 1.0

                    # Log la qualité
                    log_extraction_quality(text, file_info.filename, log=self.log)

                    # Si qualité insuffisante, utiliser OCR LLM
                    if quality_score < self.quality_threshold:
                        print(f"  ⚠️ Low quality ({quality_score:.0%}) for {file_info.filename}")
                        if LLM_OCR_AVAILABLE:
                            print(f"  🔍 Launching LLM OCR...")
                            self.log.info(
                                f"[PIPELINE] Low quality ({quality_score:.0%}) for {file_info.filename}, "
                                f"using LLM OCR..."
                            )
                            result = extract_text_with_dallem(
                                path=path,
                                quality_threshold=self.quality_threshold,
                                force_llm_ocr=False,
                                log=self.log,
                            )

                            if result.get("llm_ocr_used"):
                                text = result.get("text", text)  # Garder l'ancien si OCR échoue
                                method = result.get("method", "hybrid")
                                llm_ocr_used = True
                                print(f"  ✅ OCR completed: {len(text)} chars extracted")

                                # Sauvegarder le résultat OCR en cache
                                self._cache_ocr_result(file_info, text)
                            else:
                                print(f"  ⚠️ OCR did not improve quality, keeping original")
                        else:
                            print(f"  ❌ LLM OCR not available! Install llm_ocr module.")
                            self.log.warning(
                                f"[PIPELINE] LLM OCR not available for {file_info.filename}. "
                                f"Quality is {quality_score:.0%} but OCR module is missing."
                            )

            elif ext in (".doc", ".docx"):
                text = extract_text_from_docx(path)

            elif ext == ".xml":
                text = extract_text_from_xml(path)

            elif ext == ".pptx":
                # PowerPoint avec extraction complète (images, embedded, etc.)
                if PPTX_AVAILABLE:
                    self.log.info(f"[PIPELINE] Processing PPTX: {file_info.filename}")

                    # Fonction OCR pour images: crée un fichier temp et utilise extract_text_with_dallem
                    ocr_func = None
                    if LLM_OCR_AVAILABLE:
                        def _ocr_image_data(img_data: bytes) -> str:
                            """OCR sur données image via fichier temporaire."""
                            import tempfile
                            import os

                            # Vérifier que img_data est valide
                            if not img_data or len(img_data) < 8:
                                self.log.warning("[PPTX-OCR] Image data empty or too small")
                                return ""

                            # Détecter le format de l'image
                            if img_data[:8] == b'\x89PNG\r\n\x1a\n':
                                suffix = '.png'
                            elif img_data[:2] == b'\xff\xd8':
                                suffix = '.jpg'
                            else:
                                suffix = '.png'  # Défaut

                            try:
                                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                                    tmp.write(img_data)
                                    tmp_path = tmp.name

                                # Utiliser extract_text_with_dallem avec force_llm_ocr
                                result = extract_text_with_dallem(
                                    path=tmp_path,
                                    force_llm_ocr=True,
                                    log=self.log,
                                )
                                return result.get("text", "")
                            except Exception as e:
                                self.log.warning(f"[PPTX-OCR] OCR failed: {e}")
                                return ""
                            finally:
                                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                    os.unlink(tmp_path)

                        ocr_func = _ocr_image_data

                    # Fonction PDF si disponible
                    pdf_func = lambda pdf_path: extract_text_from_pdf(pdf_path)

                    try:
                        pptx_result = process_pptx_with_attachments(
                            pptx_path=path,
                            ocr_func=ocr_func,
                            pdf_processor=pdf_func,
                            log=self.log,
                        )

                        text = pptx_result.get("full_text", "")
                        method = "pptx_full"

                        # Log des stats
                        n_slides = pptx_result.get("total_slides", 0)
                        n_embedded = len(pptx_result.get("embedded_files", []))
                        n_images = len(pptx_result.get("images_for_ocr", []))

                        self.log.info(
                            f"[PIPELINE] PPTX extracted: {n_slides} slides, "
                            f"{n_images} images OCR'd, {n_embedded} embedded files"
                        )

                        if n_images > 0 and ocr_func:
                            llm_ocr_used = True

                    except Exception as e:
                        self.log.warning(f"[PIPELINE] PPTX full extraction failed: {e}")
                        # Fallback: extraction texte simple
                        from processing.pptx_processing import pptx_to_text
                        text = pptx_to_text(path, include_metadata=True)
                        method = "pptx_simple"
                else:
                    self.log.warning(
                        f"[PIPELINE] PPTX processing not available (python-pptx missing): "
                        f"{file_info.filename}"
                    )
                    text = ""

            elif ext in (".txt", ".md"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            elif ext == ".csv":
                # Les CSV sont généralement des listes de fichiers, pas du texte à indexer
                # Si vous voulez indexer le contenu d'un CSV, convertissez-le en .txt d'abord
                self.log.info(
                    f"[PIPELINE] CSV file skipped (use ingest_from_csv for file lists): "
                    f"{file_info.filename}"
                )
                text = ""

            else:
                self.log.warning(f"[PIPELINE] Unsupported format: {ext} ({file_info.filename})")
                text = ""

            # Détection de langue
            language = self._detect_language(text)

            return ExtractionResult(
                file_info=file_info,
                text=text,
                language=language,
                quality_score=quality_score,
                method=method,
                llm_ocr_used=llm_ocr_used,
            )

        except Exception as e:
            self.log.error(f"[PIPELINE] Error extracting {file_info.filename}: {e}")
            return ExtractionResult(
                file_info=file_info,
                error=str(e)
            )

    def _cache_ocr_result(self, file_info: FileInfo, text: str):
        """Sauvegarde le résultat OCR dans le cache local."""
        try:
            cache_file = os.path.join(
                self.temp_ocr_dir,
                f"{os.path.splitext(file_info.filename)[0]}_ocr.txt"
            )
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
            self.log.debug(f"[PIPELINE] Cached OCR result: {cache_file}")
        except Exception as e:
            self.log.warning(f"[PIPELINE] Failed to cache OCR result: {e}")

    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte."""
        if not text or len(text) < 50:
            return "unk"
        try:
            return detect(text[:5000])  # Limiter pour performance
        except Exception:
            return "unk"

    def chunk_documents(
        self,
        extraction_results: List[ExtractionResult],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[ChunkInfo]:
        """
        ÉTAPE 5: Découpe les documents en chunks.

        Utilise le chunking sémantique si disponible, sinon chunking EASA/générique.

        Args:
            extraction_results: Résultats d'extraction
            progress_cb: Callback de progression

        Returns:
            Liste de ChunkInfo prêts pour l'embedding
        """
        valid_results = [r for r in extraction_results if r.text and not r.error]

        self.log.info(
            f"[PIPELINE] ═══════════════════════════════════════════════════════\n"
            f"[PIPELINE] ✂️ Phase 1.4: CHUNKING\n"
            f"[PIPELINE] ───────────────────────────────────────────────────────\n"
            f"[PIPELINE]   Documents to chunk: {len(valid_results)}/{len(extraction_results)}\n"
            f"[PIPELINE]   Chunk size: {self.chunk_size}\n"
            f"[PIPELINE]   Semantic chunking: {self.use_semantic_chunking}\n"
            f"[PIPELINE]   EASA sections: {self.use_easa_sections}"
        )
        start_time = time.time()

        all_chunks: List[ChunkInfo] = []
        total_valid = len(valid_results)

        for i, result in enumerate(valid_results):
            progress = 0.5 + (i / total_valid) * 0.1 if total_valid > 0 else 0.5
            bar = _make_progress_bar(i / total_valid if total_valid > 0 else 1, width=20)
            if progress_cb:
                progress_cb(progress, f"✂️ Chunk {bar} {i+1}/{total_valid}")

            try:
                chunks = self._chunk_single_document(result)
                all_chunks.extend(chunks)
                self.log.debug(f"[PIPELINE] {result.file_info.filename}: {len(chunks)} chunks")

            except Exception as e:
                import traceback
                self.log.error(f"[PIPELINE] Chunking failed for {result.file_info.filename}: {e}")
                self.log.error(f"[PIPELINE] Traceback: {traceback.format_exc()}")
                print(f"  ❌ Chunking error for {result.file_info.filename}: {e}")
                print(f"      {traceback.format_exc()}")

        self.stats.chunking_time = time.time() - start_time
        self.stats.total_chunks = len(all_chunks)

        # Calculer la taille moyenne des chunks
        avg_chunk_size = sum(len(c.text) for c in all_chunks) / len(all_chunks) if all_chunks else 0

        self.log.info(
            f"[PIPELINE] ✅ Chunking completed:\n"
            f"[PIPELINE]   {_make_progress_bar(1.0, width=30)}\n"
            f"[PIPELINE]   ✂️ Total chunks: {len(all_chunks)}\n"
            f"[PIPELINE]   📊 Avg chunk size: {int(avg_chunk_size)} chars\n"
            f"[PIPELINE]   ⏱️ Duration: {_format_duration(self.stats.chunking_time)}"
        )

        return all_chunks

    def _chunk_single_document(self, result: ExtractionResult) -> List[ChunkInfo]:
        """Découpe un document en chunks."""
        text = result.text
        file_info = result.file_info
        base_name = file_info.filename

        self.log.debug(f"[CHUNK] Starting chunking for {base_name} ({len(text)} chars)")

        chunks: List[ChunkInfo] = []

        # Analyser la densité pour adapter la taille
        try:
            density_info = _calculate_content_density(text)
            density_type = density_info["density_type"]
            density_score = density_info["density_score"]
            self.log.debug(f"[CHUNK] Density: {density_type} ({density_score:.2f})")
        except Exception as e:
            self.log.warning(f"[CHUNK] Density calculation failed: {e}, using defaults")
            density_type = "normal"
            density_score = 0.5

        # Calculer la taille adaptative
        base_sizes = {"very_dense": 800, "dense": 1200, "normal": 1500, "sparse": 2000}
        recommended = base_sizes.get(density_type, 1500)
        ratio = recommended / 1500
        adapted_chunk_size = max(600, min(int(self.chunk_size * ratio), 2000))
        self.log.debug(f"[CHUNK] Adapted chunk size: {adapted_chunk_size}")

        # Utiliser le chunking sémantique si disponible
        if self.use_semantic_chunking and SEMANTIC_CHUNKING_AVAILABLE:
            from chunking.semantic_chunking import semantic_chunk

            semantic_chunks = semantic_chunk(
                text=text,
                target_size=adapted_chunk_size,
                adaptive=True,
            )

            for chunk in semantic_chunks:
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                chunk_idx = chunk.index if hasattr(chunk, 'index') else len(chunks)

                chunk_id = f"{base_name}_semantic_{chunk_idx}"
                faiss_id = f"{chunk_id}__{uuid.uuid4().hex[:8]}"

                # Utiliser le chemin logique (URL Confluence, etc.) si disponible
                display_path = self.logical_paths.get(file_info.original_path, file_info.original_path)
                metadata = {
                    "source_file": base_name,
                    "path": display_path,
                    "parent_file": file_info.parent_file,
                    "is_attachment": file_info.is_attachment,
                    "chunk_id": chunk_id,
                    "chunk_type": "semantic",
                    "language": result.language,
                    "density_type": density_type,
                    "density_score": density_score,
                    "extraction_method": result.method,
                    "llm_ocr_used": result.llm_ocr_used,
                }

                if hasattr(chunk, 'boundary_type'):
                    # Convertir en string pour la sérialisation JSON
                    bt = chunk.boundary_type
                    metadata["boundary_type"] = bt.value if hasattr(bt, 'value') else str(bt)

                chunks.append(ChunkInfo(
                    text=chunk_text,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    faiss_id=faiss_id,
                ))

        else:
            # Chunking EASA ou générique
            self.log.debug(f"[CHUNK] Using EASA/generic chunking (use_easa_sections={self.use_easa_sections})")

            sections = []
            if self.use_easa_sections:
                try:
                    sections = split_easa_sections(text)
                    self.log.debug(f"[CHUNK] EASA sections found: {len(sections)}")
                except Exception as e:
                    self.log.warning(f"[CHUNK] EASA section split failed: {e}")
                    sections = []

            if sections:
                # Sections EASA détectées
                self.log.debug(f"[CHUNK] Chunking {len(sections)} EASA sections...")
                smart_chunks = chunk_easa_sections(
                    sections,
                    max_chunk_size=adapted_chunk_size + 500,
                    min_chunk_size=200,
                    merge_small_sections=True,
                    add_context_prefix=True,
                )
            else:
                # Chunking générique
                self.log.debug(f"[CHUNK] Using generic chunking...")
                smart_chunks = smart_chunk_generic(
                    text,
                    source_file=base_name,
                    chunk_size=adapted_chunk_size + 300,
                    min_chunk_size=200,
                    overlap=100,
                )

            self.log.debug(f"[CHUNK] Got {len(smart_chunks)} raw chunks, augmenting...")

            # Augmentation des chunks
            try:
                smart_chunks = augment_chunks(
                    smart_chunks,
                    add_keywords=True,
                    add_key_phrases=True,
                    add_density_info=False,
                )
                self.log.debug(f"[CHUNK] Augmentation done")
            except Exception as e:
                self.log.warning(f"[CHUNK] Augmentation failed: {e}, using raw chunks")

            for chunk_data in smart_chunks:
                chunk_text = chunk_data.get("text", "")
                if not chunk_text:
                    continue

                chunk_idx = chunk_data.get("chunk_index", len(chunks))
                sec_id = chunk_data.get("section_id", "")

                safe_sec_id = sec_id.replace(" ", "_").replace("|", "_")[:50] if sec_id else f"chunk"
                # Toujours inclure chunk_idx pour éviter les collisions si section divisée
                chunk_id = f"{base_name}_{safe_sec_id}_{chunk_idx}"
                faiss_id = f"{chunk_id}__{uuid.uuid4().hex[:8]}"

                # Utiliser le chemin logique (URL Confluence, etc.) si disponible
                display_path = self.logical_paths.get(file_info.original_path, file_info.original_path)
                metadata = {
                    "source_file": base_name,
                    "path": display_path,
                    "parent_file": file_info.parent_file,
                    "is_attachment": file_info.is_attachment,
                    "chunk_id": chunk_id,
                    "section_id": sec_id,
                    "section_kind": chunk_data.get("section_kind", ""),
                    "section_title": chunk_data.get("section_title", ""),
                    "language": result.language,
                    "keywords": chunk_data.get("keywords", [])[:10],
                    "density_type": density_type,
                    "density_score": density_score,
                    "extraction_method": result.method,
                    "llm_ocr_used": result.llm_ocr_used,
                }

                chunks.append(ChunkInfo(
                    text=chunk_text,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    faiss_id=faiss_id,
                ))

        return chunks

    # =========================================================================
    #  PHASE 2: NETWORK OPERATIONS
    # =========================================================================

    def batch_embed_and_insert(
        self,
        chunks: List[ChunkInfo],
        rebuild: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        ÉTAPE 6-7: Génère les embeddings en batch et insère dans FAISS.

        C'est ici que se font les appels réseau (minimisés par batch).

        Args:
            chunks: Liste des chunks à indexer
            rebuild: Si True, recrée la collection
            progress_cb: Callback de progression

        Returns:
            Rapport d'ingestion
        """
        if not chunks:
            self.log.warning("[PIPELINE] ⚠️ No chunks to embed")
            return {"total_chunks": 0, "status": "empty"}

        # Utiliser embedding_batch_size configuré (mode low_memory = 8)
        batch_size = self.embedding_batch_size
        num_batches = (len(chunks) + batch_size - 1) // batch_size

        self.log.info(
            f"[PIPELINE] ═══════════════════════════════════════════════════════\n"
            f"[PIPELINE] 🌐 Phase 2: NETWORK OPERATIONS\n"
            f"[PIPELINE] ═══════════════════════════════════════════════════════\n"
            f"[PIPELINE] 🧠 Phase 2.1: EMBEDDINGS\n"
            f"[PIPELINE] ───────────────────────────────────────────────────────\n"
            f"[PIPELINE]   Chunks to embed: {len(chunks)}\n"
            f"[PIPELINE]   Batch size: {batch_size} {'(low memory)' if self.low_memory else ''}\n"
            f"[PIPELINE]   API calls needed: {num_batches}"
        )
        embed_start = time.time()

        if progress_cb:
            progress_cb(0.6, f"🧠 Embedding {_make_progress_bar(0, width=20)} 0/{num_batches} batches")

        # Créer le client d'embeddings
        http_client = create_http_client()
        emb_client = DirectOpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=SNOWFLAKE_API_KEY,
            base_url=SNOWFLAKE_API_BASE,
            http_client=http_client,
            role_prefix=True,
            logger=self.log,
        )

        # Extraire les textes
        texts = [c.text for c in chunks]

        # Embeddings en batch
        embeddings = embed_in_batches(
            texts=texts,
            role="doc",
            batch_size=batch_size,
            emb_client=emb_client,
            log=self.log,
            dry_run=False,
        )

        # Fermer le client HTTP et libérer la mémoire après embeddings
        try:
            http_client.close()
        except Exception:
            pass  # Ignorer les erreurs de fermeture

        if self.aggressive_gc:
            gc.collect()

        self.stats.embedding_time = time.time() - embed_start
        self.stats.embedding_batches = num_batches

        self.log.info(
            f"[PIPELINE] ✅ Embeddings completed:\n"
            f"[PIPELINE]   {_make_progress_bar(1.0, width=30)}\n"
            f"[PIPELINE]   🧠 Vectors created: {len(chunks)}\n"
            f"[PIPELINE]   📡 API calls: {num_batches}\n"
            f"[PIPELINE]   ⏱️ Duration: {_format_duration(self.stats.embedding_time)}"
        )

        # =====================================================================
        # ÉTAPE 7: INSERTION FAISS (LOCAL)
        # =====================================================================
        self.log.info(
            f"[PIPELINE] ───────────────────────────────────────────────────────\n"
            f"[PIPELINE] 💾 Phase 2.2: FAISS INSERT\n"
            f"[PIPELINE] ───────────────────────────────────────────────────────\n"
            f"[PIPELINE]   Database: {self.db_path}\n"
            f"[PIPELINE]   Collection: {self.collection_name}\n"
            f"[PIPELINE]   Rebuild: {rebuild}"
        )
        faiss_start = time.time()

        if progress_cb:
            progress_cb(0.85, f"💾 FAISS {_make_progress_bar(0, width=20)} init...")

        # Créer/ouvrir le store FAISS
        store = build_faiss_store(self.db_path, use_local_cache=False, lazy_load=True)

        # Option rebuild
        if rebuild:
            try:
                existing = store.list_collections()
                if self.collection_name in existing:
                    store.delete_collection(self.collection_name)
                    self.log.info(f"[PIPELINE] Collection '{self.collection_name}' deleted (rebuild=True)")
            except Exception as e:
                self.log.warning(f"[PIPELINE] Could not delete collection: {e}")

        # Créer la collection
        collection = store.get_or_create_collection(name=self.collection_name, dimension=1024)

        # Validation globale: vérifier que le nombre d'embeddings correspond aux chunks
        n = len(chunks)
        if len(embeddings) != n:
            self.log.error(
                f"[PIPELINE] ⚠️ Global embedding count mismatch: "
                f"{len(embeddings)} embeddings vs {n} chunks"
            )
            # Ajuster n au minimum pour éviter IndexError
            n = min(len(embeddings), n)

        # Insérer par batch
        max_batch = 4000
        num_faiss_batches = (n + max_batch - 1) // max_batch

        for batch_idx, start in enumerate(range(0, n, max_batch)):
            end = min(start + max_batch, n)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            # Validation: vérifier que le nombre d'embeddings correspond aux chunks
            if len(batch_embeddings) != len(batch_chunks):
                self.log.error(
                    f"[PIPELINE] ⚠️ Embedding count mismatch in batch {batch_idx}: "
                    f"{len(batch_embeddings)} embeddings vs {len(batch_chunks)} chunks. Skipping batch."
                )
                continue

            progress = 0.85 + ((batch_idx + 1) / num_faiss_batches) * 0.1
            bar = _make_progress_bar((batch_idx + 1) / num_faiss_batches, width=20)
            if progress_cb:
                progress_cb(progress, f"💾 FAISS {bar} batch {batch_idx+1}/{num_faiss_batches}")

            collection.add(
                ids=[c.faiss_id for c in batch_chunks],
                embeddings=batch_embeddings.tolist(),
                documents=[c.text for c in batch_chunks],
                metadatas=[c.metadata for c in batch_chunks],
            )

        self.stats.faiss_time = time.time() - faiss_start

        # Cleanup
        del collection
        del store
        gc.collect()

        # Synchronisation pour stockage réseau
        time.sleep(1)

        if progress_cb:
            progress_cb(1.0, f"✅ Done! {n} chunks indexed")

        self.log.info(
            f"[PIPELINE] ✅ FAISS insert completed:\n"
            f"[PIPELINE]   {_make_progress_bar(1.0, width=30)}\n"
            f"[PIPELINE]   💾 Chunks indexed: {n}\n"
            f"[PIPELINE]   📦 Batches: {num_faiss_batches}\n"
            f"[PIPELINE]   ⏱️ Duration: {_format_duration(self.stats.faiss_time)}"
        )

        return {
            "total_chunks": n,
            "embedding_batches": self.stats.embedding_batches,
            "status": "success",
        }

    # =========================================================================
    #  MAIN ENTRY POINTS
    # =========================================================================

    def ingest_files(
        self,
        file_paths: List[str],
        rebuild: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Ingère une liste de fichiers dans la collection FAISS.

        Pipeline complet:
        1. Download → 2. Extract PJ → 3. Extract text → 4. OCR fallback → 5. Chunk → 6. Embed → 7. FAISS

        En mode streaming (low_memory), chaque fichier est traité entièrement avant
        de passer au suivant, libérant la mémoire après chaque fichier.

        Args:
            file_paths: Liste des chemins de fichiers
            rebuild: Si True, recrée la collection
            progress_cb: Callback de progression

        Returns:
            Rapport d'ingestion avec statistiques
        """
        total_start = time.time()
        self.stats = PipelineStats(total_files=len(file_paths))

        mode_str = "STREAMING (low memory)" if self.streaming_mode else "BATCH (normal)"

        self.log.info(
            f"\n"
            f"[PIPELINE] ╔═══════════════════════════════════════════════════════════╗\n"
            f"[PIPELINE] ║         📚 INGESTION PIPELINE STARTED                     ║\n"
            f"[PIPELINE] ╠═══════════════════════════════════════════════════════════╣\n"
            f"[PIPELINE] ║  Files: {len(file_paths):<10}                                   ║\n"
            f"[PIPELINE] ║  Database: {os.path.basename(self.db_path):<46} ║\n"
            f"[PIPELINE] ║  Collection: {self.collection_name:<44} ║\n"
            f"[PIPELINE] ║  Rebuild: {str(rebuild):<47} ║\n"
            f"[PIPELINE] ║  Mode: {mode_str:<50} ║\n"
            f"[PIPELINE] ╚═══════════════════════════════════════════════════════════╝"
        )

        try:
            # Choisir le mode de traitement
            if self.streaming_mode:
                return self._ingest_files_streaming(file_paths, rebuild, progress_cb, total_start)
            else:
                return self._ingest_files_batch(file_paths, rebuild, progress_cb, total_start)

        except Exception as e:
            self.log.error(f"[PIPELINE] ❌ Pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_files": self.stats.total_files,
                "processed_files": self.stats.processed_files,
            }

    def _ingest_files_batch(
        self,
        file_paths: List[str],
        rebuild: bool,
        progress_cb: Optional[ProgressCallback],
        total_start: float,
    ) -> Dict[str, Any]:
        """
        Mode batch: traite tous les fichiers en parallèle (mode normal).
        Plus rapide mais utilise plus de mémoire.
        """
        # Phase 1: Local processing

        # 1. Download
        file_infos = self.download_to_temp(file_paths, progress_cb)

        # 2. Extract attachments
        all_files = self.extract_attachments(file_infos, progress_cb)

        # 3-4. Extract text with OCR fallback
        extraction_results = self.extract_text_local(all_files, progress_cb)

        # 5. Chunking
        chunks = self.chunk_documents(extraction_results, progress_cb)

        # Phase 2: Network operations

        # 6-7. Embed and insert
        result = self.batch_embed_and_insert(chunks, rebuild, progress_cb)

        # Finalize stats
        self.stats.total_time = time.time() - total_start
        self.stats.processed_files = len([r for r in extraction_results if r.text])
        self.stats.failed_files = len([r for r in extraction_results if r.error])

        return self._finalize_result()

    def _ingest_files_streaming(
        self,
        file_paths: List[str],
        rebuild: bool,
        progress_cb: Optional[ProgressCallback],
        total_start: float,
    ) -> Dict[str, Any]:
        """
        Mode streaming: traite chaque fichier entièrement avant le suivant.
        Plus lent mais utilise beaucoup moins de mémoire (adapté aux PC 8 Go RAM).

        Pour chaque fichier:
        1. Download → 2. Extract PJ → 3. Extract text → 4. OCR → 5. Chunk → 6. Embed → 7. FAISS
        8. Libérer la mémoire avant le fichier suivant
        """
        print(f"\n{'='*60}")
        print(f"🔄 STREAMING MODE ENABLED (Low Memory)")
        print(f"{'='*60}")
        print(f"  Processing one file at a time to minimize RAM usage")
        print(f"{'─'*60}")

        total = len(file_paths)
        all_chunks_count = 0
        processed = 0
        failed = 0

        # Préparer le store FAISS une seule fois
        store = build_faiss_store(self.db_path, use_local_cache=False, lazy_load=True)

        # Option rebuild
        if rebuild:
            try:
                existing = store.list_collections()
                if self.collection_name in existing:
                    store.delete_collection(self.collection_name)
                    self.log.info(f"[PIPELINE] Collection '{self.collection_name}' deleted (rebuild=True)")
            except Exception as e:
                self.log.warning(f"[PIPELINE] Could not delete collection: {e}")

        collection = store.get_or_create_collection(name=self.collection_name, dimension=1024)

        # Client d'embeddings (réutilisé pour tous les fichiers)
        http_client = create_http_client()
        emb_client = DirectOpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=SNOWFLAKE_API_KEY,
            base_url=SNOWFLAKE_API_BASE,
            http_client=http_client,
            role_prefix=True,
            logger=self.log,
        )

        for idx, file_path in enumerate(file_paths):
            file_start = time.time()
            progress_base = idx / total
            filename = os.path.basename(file_path)

            print(f"\n{'─'*60}")
            print(f"📄 [{idx+1}/{total}] Processing: {filename}")
            print(f"{'─'*60}")

            if progress_cb:
                progress_cb(progress_base, f"📄 File {idx+1}/{total}: {filename[:30]}...")

            try:
                # 1. Download ce fichier uniquement
                download_start = time.time()
                file_infos = self.download_to_temp([file_path], None)
                self.stats.download_time += time.time() - download_start

                if not file_infos or not file_infos[0].local_path:
                    print(f"  ❌ Failed to download")
                    failed += 1
                    continue

                # 2. Extract attachments
                all_files = self.extract_attachments(file_infos, None)
                self.stats.total_attachments += len(all_files) - len(file_infos)

                # 3-4. Extract text
                extract_start = time.time()
                extraction_results = self.extract_text_local(all_files, None)
                self.stats.extraction_time += time.time() - extract_start

                valid_extractions = [r for r in extraction_results if r.text]
                if not valid_extractions:
                    print(f"  ⚠️ No text extracted")
                    failed += 1
                    continue

                # Compter OCR
                for r in extraction_results:
                    if r.llm_ocr_used:
                        self.stats.ocr_files += 1

                # 5. Chunking
                chunk_start = time.time()
                chunks = self.chunk_documents(extraction_results, None)
                self.stats.chunking_time += time.time() - chunk_start

                if not chunks:
                    print(f"  ⚠️ No chunks created")
                    failed += 1
                    continue

                # 6. Embeddings
                embed_start = time.time()
                texts = [c.text for c in chunks]
                embeddings = embed_in_batches(
                    texts=texts,
                    role="doc",
                    batch_size=self.embedding_batch_size,
                    emb_client=emb_client,
                    log=self.log,
                    dry_run=False,
                )
                self.stats.embedding_time += time.time() - embed_start

                # 7. Insert into FAISS
                faiss_start = time.time()
                collection.add(
                    ids=[c.faiss_id for c in chunks],
                    embeddings=embeddings.tolist(),
                    documents=[c.text for c in chunks],
                    metadatas=[c.metadata for c in chunks],
                )
                self.stats.faiss_time += time.time() - faiss_start

                # Stats
                all_chunks_count += len(chunks)
                processed += 1
                file_duration = time.time() - file_start

                print(f"  ✅ {len(chunks)} chunks indexed in {_format_duration(file_duration)}")

                # 8. Libérer la mémoire
                del file_infos
                del all_files
                del extraction_results
                del chunks
                del texts
                del embeddings
                gc.collect()

                print(f"  🧹 Memory released")

            except Exception as e:
                import traceback
                print(f"  ❌ Error: {e}")
                self.log.error(f"[PIPELINE] File {filename} failed: {traceback.format_exc()}")
                failed += 1

                # Libérer la mémoire même en cas d'erreur
                gc.collect()

        # Cleanup du client HTTP et du store
        try:
            http_client.close()
        except Exception:
            pass

        del collection
        del store
        gc.collect()

        # Finalize stats
        self.stats.total_time = time.time() - total_start
        self.stats.processed_files = processed
        self.stats.failed_files = failed
        self.stats.total_chunks = all_chunks_count

        return self._finalize_result()

    def _finalize_result(self) -> Dict[str, Any]:
        """Génère le rapport final de l'ingestion."""
        # Calculer les pourcentages de temps
        total_time = self.stats.total_time or 1
        dl_pct = (self.stats.download_time / total_time) * 100
        ext_pct = (self.stats.extraction_time / total_time) * 100
        chunk_pct = (self.stats.chunking_time / total_time) * 100
        embed_pct = (self.stats.embedding_time / total_time) * 100
        faiss_pct = (self.stats.faiss_time / total_time) * 100

        self.log.info(
            f"\n"
            f"[PIPELINE] ╔═══════════════════════════════════════════════════════════╗\n"
            f"[PIPELINE] ║         ✅ INGESTION COMPLETED SUCCESSFULLY               ║\n"
            f"[PIPELINE] ╠═══════════════════════════════════════════════════════════╣\n"
            f"[PIPELINE] ║  📊 SUMMARY                                               ║\n"
            f"[PIPELINE] ║  ─────────────────────────────────────────────────────── ║\n"
            f"[PIPELINE] ║  Files processed: {self.stats.processed_files}/{self.stats.total_files} (+{self.stats.total_attachments} attachments)        ║\n"
            f"[PIPELINE] ║  Files failed: {self.stats.failed_files:<42} ║\n"
            f"[PIPELINE] ║  OCR used: {self.stats.ocr_files:<46} ║\n"
            f"[PIPELINE] ║  Chunks created: {self.stats.total_chunks:<40} ║\n"
            f"[PIPELINE] ╠═══════════════════════════════════════════════════════════╣\n"
            f"[PIPELINE] ║  ⏱️ TIMING BREAKDOWN                                       ║\n"
            f"[PIPELINE] ║  ─────────────────────────────────────────────────────── ║\n"
            f"[PIPELINE] ║  Download:   {_format_duration(self.stats.download_time):>8} ({dl_pct:5.1f}%)                     ║\n"
            f"[PIPELINE] ║  Extraction: {_format_duration(self.stats.extraction_time):>8} ({ext_pct:5.1f}%)                     ║\n"
            f"[PIPELINE] ║  Chunking:   {_format_duration(self.stats.chunking_time):>8} ({chunk_pct:5.1f}%)                     ║\n"
            f"[PIPELINE] ║  Embedding:  {_format_duration(self.stats.embedding_time):>8} ({embed_pct:5.1f}%)                     ║\n"
            f"[PIPELINE] ║  FAISS:      {_format_duration(self.stats.faiss_time):>8} ({faiss_pct:5.1f}%)                     ║\n"
            f"[PIPELINE] ║  ─────────────────────────────────────────────────────── ║\n"
            f"[PIPELINE] ║  TOTAL:      {_format_duration(self.stats.total_time):>8}                                ║\n"
            f"[PIPELINE] ╚═══════════════════════════════════════════════════════════╝"
        )

        return {
            "status": "success",
            "total_files": self.stats.total_files,
            "processed_files": self.stats.processed_files,
            "failed_files": self.stats.failed_files,
            "attachments": self.stats.total_attachments,
            "ocr_files": self.stats.ocr_files,
            "total_chunks": self.stats.total_chunks,
            "total_time": self.stats.total_time,
            "timings": {
                "download": self.stats.download_time,
                "extraction": self.stats.extraction_time,
                "chunking": self.stats.chunking_time,
                "embedding": self.stats.embedding_time,
                "faiss": self.stats.faiss_time,
            },
        }

    def ingest_from_csv(
        self,
        csv_path: str,
        file_column: str = "path",
        delimiter: str = ";",
        rebuild: bool = False,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Ingère les fichiers référencés dans un CSV.

        Args:
            csv_path: Chemin du fichier CSV
            file_column: Nom de la colonne contenant les chemins
            delimiter: Délimiteur CSV
            rebuild: Si True, recrée la collection
            progress_cb: Callback de progression

        Returns:
            Rapport d'ingestion
        """
        self.log.info(f"[PIPELINE] Reading files from CSV: {csv_path}")

        file_paths: List[str] = []

        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)

                for row in reader:
                    path = row.get(file_column, "").strip()
                    if not path:
                        continue

                    # Accepter les URLs directement (vérification lors du téléchargement)
                    if self._is_url(path):
                        file_paths.append(path)
                    elif os.path.exists(path):
                        file_paths.append(path)
                    else:
                        self.log.warning(f"[PIPELINE] File not found: {path}")

            self.log.info(f"[PIPELINE] Found {len(file_paths)} files in CSV")

            if not file_paths:
                return {
                    "status": "error",
                    "error": f"No valid files found in CSV column '{file_column}'",
                }

            return self.ingest_files(file_paths, rebuild=rebuild, progress_cb=progress_cb)

        except Exception as e:
            self.log.error(f"[PIPELINE] Failed to read CSV: {e}")
            return {
                "status": "error",
                "error": str(e),
            }


# =============================================================================
#  CONVENIENCE FUNCTIONS
# =============================================================================

def quick_ingest(
    file_paths: List[str],
    db_path: str,
    collection_name: str,
    quality_threshold: float = 0.5,
    force_ocr: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Fonction simplifiée pour ingérer des fichiers rapidement.

    Args:
        file_paths: Liste des fichiers à ingérer
        db_path: Chemin de la base FAISS
        collection_name: Nom de la collection
        quality_threshold: Seuil de qualité OCR (0-1)
        force_ocr: Forcer l'OCR LLM pour tous les PDF
        progress_cb: Callback de progression

    Returns:
        Rapport d'ingestion

    Example:
        >>> result = quick_ingest(
        ...     file_paths=["doc1.pdf", "doc2.pdf"],
        ...     db_path="/path/to/db",
        ...     collection_name="my_docs",
        ...     quality_threshold=0.5,
        ... )
        >>> print(f"Indexé {result['total_chunks']} chunks")
    """
    with OptimizedIngestionPipeline(
        db_path=db_path,
        collection_name=collection_name,
        quality_threshold=quality_threshold,
        force_ocr=force_ocr,
    ) as pipeline:
        return pipeline.ingest_files(file_paths, progress_cb=progress_cb)


def ingest_csv(
    csv_path: str,
    db_path: str,
    collection_name: str,
    file_column: str = "path",
    delimiter: str = ";",
    quality_threshold: float = 0.5,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Ingère les fichiers depuis un CSV.

    Example:
        >>> result = ingest_csv(
        ...     csv_path="files.csv",
        ...     db_path="/path/to/db",
        ...     collection_name="documents",
        ...     file_column="file_path",
        ... )
    """
    with OptimizedIngestionPipeline(
        db_path=db_path,
        collection_name=collection_name,
        quality_threshold=quality_threshold,
    ) as pipeline:
        return pipeline.ingest_from_csv(
            csv_path=csv_path,
            file_column=file_column,
            delimiter=delimiter,
            progress_cb=progress_cb,
        )


# =============================================================================
#  COMPATIBILITY FUNCTION (drop-in replacement for rag_ingestion.ingest_documents)
# =============================================================================

def ingest_documents(
    file_paths: List[str],
    db_path: str,
    collection_name: str,
    chunk_size: int = 1500,
    use_easa_sections: bool = True,
    rebuild: bool = False,
    log=None,
    logical_paths: Optional[Dict[str, str]] = None,
    parent_paths: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable] = None,
    xml_configs: Optional[Dict] = None,
    quality_threshold: float = 0.5,
    force_ocr: bool = False,
    # Options low memory (None = détection automatique basée sur RAM)
    low_memory: Optional[bool] = None,
    max_workers: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    streaming_mode: Optional[bool] = None,
    aggressive_gc: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Fonction de compatibilité pour remplacer rag_ingestion.ingest_documents.

    Utilise le nouveau pipeline optimisé en interne tout en conservant
    la même interface que l'ancienne fonction.

    Args:
        file_paths: Liste des fichiers à ingérer
        db_path: Chemin de la base FAISS
        collection_name: Nom de la collection
        chunk_size: Taille des chunks
        use_easa_sections: Utiliser les sections EASA
        rebuild: Reconstruire la collection
        log: Logger (optionnel)
        logical_paths: Mapping des chemins logiques
        parent_paths: Mapping des fichiers parents (pièces jointes)
        progress_callback: Callback de progression
        xml_configs: Configs XML (non utilisé dans le nouveau pipeline)
        quality_threshold: Seuil de qualité OCR (0-1)
        force_ocr: Forcer l'OCR pour tous les PDF
        low_memory: Mode faible mémoire (None = auto-détection basée sur RAM,
                   True = forcé, False = désactivé). Auto-activé si RAM <= 10 Go.
        max_workers: Nombre de workers parallèles (défaut: 1 en low_memory, 8 sinon)
        embedding_batch_size: Taille des batches d'embeddings (défaut: 8 en low_memory)
        streaming_mode: Traiter fichier par fichier (défaut: True en low_memory)
        aggressive_gc: Libération mémoire agressive (défaut: True en low_memory)

    Returns:
        Dict avec total_chunks et files (format compatible)
    """
    print(f"\n{'='*60}")
    print(f"🚀 INGESTION PIPELINE - CONFIGURATION AUTO")
    print(f"{'='*60}")
    print(f"  📁 Fichiers: {len(file_paths)}")
    print(f"  📂 Base: {db_path}")
    print(f"  📚 Collection: {collection_name}")
    print(f"  🔄 Rebuild: {rebuild}")
    print(f"  💾 Config: {OPTIMAL_CONFIG['description']}")
    print(f"{'='*60}\n")

    with OptimizedIngestionPipeline(
        db_path=db_path,
        collection_name=collection_name,
        chunk_size=chunk_size,
        use_easa_sections=use_easa_sections,
        quality_threshold=quality_threshold,
        force_ocr=force_ocr,
        log=log,
        # Options low memory
        low_memory=low_memory,
        max_workers=max_workers,
        embedding_batch_size=embedding_batch_size,
        streaming_mode=streaming_mode,
        aggressive_gc=aggressive_gc,
        # Chemins logiques (URLs Confluence, etc.)
        logical_paths=logical_paths,
    ) as pipeline:
        # Appeler le pipeline
        result = pipeline.ingest_files(
            file_paths=file_paths,
            rebuild=rebuild,
            progress_cb=progress_callback,
        )

        # Convertir le résultat au format attendu par l'ancien système
        files_report = []
        if result.get("status") == "success":
            # Créer un rapport par fichier (format simplifié)
            for i, path in enumerate(file_paths):
                files_report.append({
                    "file": os.path.basename(path),
                    "num_chunks": result.get("total_chunks", 0) // max(len(file_paths), 1),
                    "language": "auto",
                    "sections_detected": use_easa_sections,
                })

        return {
            "total_chunks": result.get("total_chunks", 0),
            "files": files_report,
            "status": result.get("status", "error"),
            "processed_files": result.get("processed_files", 0),
            "failed_files": result.get("failed_files", 0),
            "attachments": result.get("attachments", 0),
        }
