import os
import uuid
import gc
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable

from langdetect import detect

from core.faiss_store import FaissStore

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

from processing.pdf_processing import extract_text_from_pdf, extract_attachments_from_pdf
from processing.docx_processing import extract_text_from_docx
from processing.csv_processing import extract_text_from_csv
from processing.xml_processing import extract_text_from_xml, XMLParseConfig
from chunking.chunking import (
    simple_chunk,
    chunk_easa_sections,
    smart_chunk_generic,
    adaptive_chunk_document,
    augment_chunks,
    _calculate_content_density,
    add_cross_references_to_chunks,
    extract_cross_references,
)
from chunking.easa_sections import split_easa_sections

logger = make_logger(debug=False)

# =====================================================================
#  FAISS HELPERS
# =====================================================================

def build_faiss_store(path: str, use_local_cache: bool = False, lazy_load: bool = True) -> FaissStore:
    """Create (and if needed, initialize) a FAISS store.

    Args:
        path: Chemin du répertoire de la base
        use_local_cache: Si True, utilise le cache local pour les lectures
        lazy_load: Si True, charge les index seulement au premier accès
    """
    os.makedirs(path, exist_ok=True)
    return FaissStore(path=path, use_local_cache=use_local_cache, lazy_load=lazy_load)


def get_or_create_collection(store: FaissStore, name: str):
    """Return an existing collection or create it if missing."""
    return store.get_or_create_collection(name=name, dimension=1024)  # Snowflake Arctic = 1024d

# =====================================================================
#  FILE LOADING
# =====================================================================

def load_file_content(path: str, xml_configs: Optional[Dict[str, XMLParseConfig]] = None) -> str:
    """Load text from a supported file type (PDF, DOCX, CSV, TXT, MD, XML).

    Args:
        path: Chemin vers le fichier
        xml_configs: Dict optionnel {chemin_fichier: XMLParseConfig} pour les fichiers XML
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".doc", ".docx"):
        # Note: legacy .doc may fail depending on content; .docx is fully supported.
        return extract_text_from_docx(path)
    if ext == ".csv":
        return extract_text_from_csv(path)
    if ext == ".xml":
        # Utiliser la config spécifique si fournie, sinon config par défaut
        config = xml_configs.get(path) if xml_configs else None
        return extract_text_from_xml(path, config)
    if ext in (".txt", ".md"):
        # Plain text / Markdown files: read as UTF-8 with tolerant error handling
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Unsupported file format for ingestion: {ext} ({path})")


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unk"


# Variable globale pour stocker les configs XML (utilisée par le worker)
_xml_configs_global: Optional[Dict[str, XMLParseConfig]] = None


def _load_single_file_worker(path: str) -> Dict[str, Any]:
    """
    Worker function for parallel file loading.
    Returns a dict with path, text, language, and error (if any).
    Must be at module level for pickling by multiprocessing.
    """
    global _xml_configs_global

    result = {
        "path": path,
        "text": "",
        "language": "",
        "error": None
    }

    try:
        if not os.path.isfile(path):
            result["error"] = f"File not found: {path}"
            return result

        text = load_file_content(path, _xml_configs_global)

        if not text.strip():
            result["error"] = f"No text extracted from {path}"
            return result

        result["text"] = text
        result["language"] = detect_language(text) or ""

    except Exception as e:
        result["error"] = f"Error loading {path}: {str(e)}"

    return result


# =====================================================================
#  INGESTION
# =====================================================================

def ingest_documents(
    file_paths: List[str],
    db_path: str,
    collection_name: str,
    chunk_size: int = 1000,
    use_easa_sections: bool = True,
    rebuild: bool = False,
    log=None,
    logical_paths: Optional[Dict[str, str]] = None,
    parent_paths: Optional[Dict[str, str]] = None,
    progress_callback: Optional[Callable] = None,
    xml_configs: Optional[Dict[str, XMLParseConfig]] = None,
) -> Dict[str, Any]:
    """Ingest a list of documents into a FAISS collection.

    Steps:
      - load raw text (PDF / DOCX / CSV / XML)
      - optionally split into EASA sections
      - chunk text
      - compute embeddings with Snowflake
      - add to FAISS

    Args:
        logical_paths: Dict optionnel {chemin_réel: chemin_logique} pour le CSV de tracking
        parent_paths: Dict optionnel {chemin_pièce_jointe: chemin_fichier_parent} pour les PJ
        xml_configs: Dict optionnel {chemin_fichier: XMLParseConfig} pour les fichiers XML

    Returns a small report with total_chunks and per-file info.
    """
    global _xml_configs_global
    _xml_configs_global = xml_configs  # Rendre accessible au worker

    _log = log or logger

    _log.info(f"[INGEST] DB={db_path} | collection={collection_name}")
    client = build_faiss_store(db_path)

    # Option rebuild: drop collection if it already exists
    if rebuild:
        try:
            existing = client.list_collections()  # FAISS retourne directement une liste de noms
            if collection_name in existing:
                client.delete_collection(collection_name)
                _log.info(
                    f"[INGEST] Collection '{collection_name}' deleted (rebuild=True)"
                )
        except Exception as e:
            _log.warning(
                f"[INGEST] Could not delete collection '{collection_name}': {e}"
            )

    col = get_or_create_collection(client, collection_name)

    # Embeddings client for Snowflake Arctic
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )

    total_chunks = 0
    file_reports: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Chargement parallèle des fichiers (extraction texte multi-threads)
    # ------------------------------------------------------------------
    # Note: ThreadPoolExecutor au lieu de ProcessPoolExecutor pour éviter
    # les problèmes avec PyMuPDF sur Windows (MemoryError, DLL load failures)
    _log.info(f"[INGEST] Starting parallel file loading with {multiprocessing.cpu_count()} threads")

    # Utiliser ThreadPoolExecutor pour le traitement parallèle
    # Threads au lieu de processus = meilleure compatibilité Windows + PyMuPDF
    # max(1, ...) pour éviter l'erreur si file_paths est vide
    max_workers = max(1, min(multiprocessing.cpu_count(), len(file_paths)))

    # Dictionnaire pour stocker les résultats chargés: path -> {text, language, error}
    loaded_files: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les fichiers pour traitement parallèle
        future_to_path = {executor.submit(_load_single_file_worker, path): path for path in file_paths}

        # Compteur pour la barre de progression
        completed_count = 0
        total_count = len(file_paths)

        # Récupérer les résultats au fur et à mesure
        for future in as_completed(future_to_path):
            try:
                result = future.result()
                path = result["path"]
                filename = os.path.basename(path)

                if result["error"]:
                    _log.error(f"[INGEST] {result['error']}")
                else:
                    loaded_files[path] = result
                    _log.info(f"[INGEST] Loaded file: {path} ({len(result['text'])} chars)")

                # Mise à jour de la progression
                completed_count += 1
                if progress_callback:
                    progress_fraction = 0.3 * (completed_count / total_count)  # 0-30% pour le chargement
                    progress_callback(
                        progress_fraction,
                        f"📄 Chargement: {completed_count}/{total_count} - {filename}"
                    )

            except Exception as e:
                path = future_to_path[future]
                _log.error(f"[INGEST] Unexpected error loading {path}: {e}")
                completed_count += 1
                if progress_callback:
                    progress_fraction = 0.3 * (completed_count / total_count)
                    progress_callback(progress_fraction, f"❌ Erreur: {os.path.basename(path)}")

    _log.info(f"[INGEST] Parallel loading complete: {len(loaded_files)}/{len(file_paths)} files loaded")

    # ------------------------------------------------------------------
    # Fonction worker pour chunker un fichier (parallélisable)
    # ------------------------------------------------------------------
    def _chunk_single_file(args):
        """Worker pour chunker un seul fichier en parallèle."""
        path, file_data, use_easa, chunk_sz, log_paths, par_paths = args

        text = file_data["text"]
        language = file_data["language"]
        base_name = os.path.basename(path)

        # Déterminer si c'est une pièce jointe et quel est le fichier parent
        parent_file = par_paths.get(path) if par_paths else None
        is_attachment = parent_file is not None

        chunks_list = []
        metas_list = []
        faiss_ids_list = []

        # Essayer de découper en sections EASA
        sections = split_easa_sections(text) if use_easa else []

        # Analyser la densité du document UNE SEULE FOIS
        density_info = _calculate_content_density(text)
        density_type = density_info["density_type"]
        density_score = density_info["density_score"]

        # Calculer la taille adaptative
        base_sizes = {"very_dense": 800, "dense": 1200, "normal": 1500, "sparse": 2000}
        recommended = base_sizes.get(density_type, 1500)
        ratio = recommended / 1500
        adapted_chunk_size = max(600, min(int(chunk_sz * ratio), 2000))

        if sections:
            # CAS 1 — Sections EASA
            smart_chunks = chunk_easa_sections(
                sections,
                max_chunk_size=adapted_chunk_size + 500,
                min_chunk_size=200,
                merge_small_sections=True,
                add_context_prefix=True,
            )

            smart_chunks = augment_chunks(
                smart_chunks,
                add_keywords=True,
                add_key_phrases=True,
                add_density_info=False,
            )

            for chunk in smart_chunks:
                chunk["density_type"] = density_type
                chunk["density_score"] = density_score

            smart_chunks = add_cross_references_to_chunks(smart_chunks)

            for smart_chunk in smart_chunks:
                ch = smart_chunk.get("text", "")
                if not ch:
                    continue

                sec_id = smart_chunk.get("section_id", "")
                sec_kind = smart_chunk.get("section_kind", "")
                sec_title = smart_chunk.get("section_title", "")
                chunk_idx = smart_chunk.get("chunk_index", 0)
                keywords = smart_chunk.get("keywords", [])
                references_to = smart_chunk.get("references_to", [])

                safe_sec_id = sec_id.replace(" ", "_").replace("|", "_") if sec_id else "no_section"
                chunk_id = f"{base_name}_{safe_sec_id}_{chunk_idx}"
                faiss_id = f"{chunk_id}__{uuid.uuid4().hex[:8]}"

                chunks_list.append(ch)
                metas_list.append({
                    "source_file": base_name,
                    "path": log_paths.get(path, path) if log_paths else path,
                    "parent_file": parent_file,
                    "is_attachment": is_attachment,
                    "chunk_id": chunk_id,
                    "section_id": sec_id,
                    "section_kind": sec_kind,
                    "section_title": sec_title,
                    "language": language,
                    "is_complete_section": smart_chunk.get("is_complete_section", False),
                    "keywords": keywords[:10] if keywords else [],
                    "density_type": density_type,
                    "density_score": density_score,
                    "references_to": references_to[:5] if references_to else [],
                })
                faiss_ids_list.append(faiss_id)
        else:
            # CAS 2 — Chunking générique
            smart_chunks = smart_chunk_generic(
                text,
                source_file=base_name,
                chunk_size=adapted_chunk_size + 300,
                min_chunk_size=200,
                overlap=100,
                add_source_prefix=True,
                preserve_lists=True,
                preserve_headers=True,
            )

            smart_chunks = augment_chunks(
                smart_chunks,
                add_keywords=True,
                add_key_phrases=True,
                add_density_info=False,
            )

            for chunk in smart_chunks:
                chunk["density_type"] = density_type
                chunk["density_score"] = density_score

            smart_chunks = add_cross_references_to_chunks(smart_chunks)

            for smart_chunk in smart_chunks:
                ch = smart_chunk.get("text", "")
                if not ch:
                    continue

                chunk_idx = smart_chunk.get("chunk_index", 0)
                header = smart_chunk.get("header", "")
                keywords = smart_chunk.get("keywords", [])
                references_to = smart_chunk.get("references_to", [])

                chunk_id = f"{base_name}_chunk_{chunk_idx}"
                faiss_id = f"{chunk_id}__{uuid.uuid4().hex[:8]}"

                chunks_list.append(ch)
                metas_list.append({
                    "source_file": base_name,
                    "path": log_paths.get(path, path) if log_paths else path,
                    "parent_file": parent_file,
                    "is_attachment": is_attachment,
                    "chunk_id": chunk_id,
                    "section_id": header[:50] if header else "",
                    "section_kind": smart_chunk.get("type", ""),
                    "section_title": header if header else "",
                    "language": language,
                    "keywords": keywords[:10] if keywords else [],
                    "density_type": density_type,
                    "density_score": density_score,
                    "references_to": references_to[:5] if references_to else [],
                })
                faiss_ids_list.append(faiss_id)

        return {
            "path": path,
            "base_name": base_name,
            "chunks": chunks_list,
            "metas": metas_list,
            "faiss_ids": faiss_ids_list,
            "language": language,
            "sections_detected": bool(sections),
            "num_chunks": len(chunks_list),
        }

    # ------------------------------------------------------------------
    # Traitement PARALLÈLE des fichiers (chunking + augmentation)
    # ------------------------------------------------------------------
    total_loaded = len(loaded_files)

    # S'assurer qu'on a au moins 1 worker (évite l'erreur si total_loaded=0)
    chunking_workers = max(1, min(multiprocessing.cpu_count(), total_loaded, 8))
    _log.info(f"[INGEST] Starting parallel chunking with {chunking_workers} workers for {total_loaded} files")

    # Préparer les arguments pour chaque fichier
    chunking_args = [
        (path, loaded_files[path], use_easa_sections, chunk_size, logical_paths, parent_paths)
        for path in file_paths if path in loaded_files
    ]

    # Exécuter le chunking en parallèle
    all_file_results = []
    with ThreadPoolExecutor(max_workers=chunking_workers) as executor:
        futures = {executor.submit(_chunk_single_file, args): args[0] for args in chunking_args}

        completed = 0
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                if result["num_chunks"] > 0:
                    all_file_results.append(result)
                    _log.info(f"[INGEST] Chunked {result['base_name']}: {result['num_chunks']} chunks")
                else:
                    _log.warning(f"[INGEST] No chunks generated for {path}")
            except Exception as e:
                _log.error(f"[INGEST] Error chunking {path}: {e}")

            completed += 1
            if progress_callback:
                progress_fraction = 0.3 + (0.3 * completed / total_loaded)  # 30-60% pour chunking
                progress_callback(progress_fraction, f"🔄 Chunking: {completed}/{total_loaded}")

    _log.info(f"[INGEST] Parallel chunking complete: {len(all_file_results)} files with chunks")

    # ------------------------------------------------------------------
    # Agrégation et Embedding de tous les chunks
    # ------------------------------------------------------------------
    all_chunks = []
    all_metas = []
    all_faiss_ids = []

    for result in all_file_results:
        all_chunks.extend(result["chunks"])
        all_metas.extend(result["metas"])
        all_faiss_ids.extend(result["faiss_ids"])
        file_reports.append({
            "file": result["base_name"],
            "num_chunks": result["num_chunks"],
            "language": result["language"],
            "sections_detected": result["sections_detected"],
        })

    if not all_chunks:
        _log.warning("[INGEST] No chunks generated from any file")
    else:
        # Embedding de tous les chunks en une seule passe (déjà parallélisé)
        _log.info(f"[INGEST] Embedding {len(all_chunks)} total chunks")
        if progress_callback:
            progress_callback(0.6, f"🧠 Embedding {len(all_chunks)} chunks...")

        embeddings = embed_in_batches(
            texts=all_chunks,
            role="doc",
            batch_size=BATCH_SIZE,
            emb_client=emb_client,
            log=_log,
            dry_run=False,
        )

        # Push to FAISS in batches
        max_batch = 4000
        n = len(all_chunks)
        _log.info(f"[INGEST] Adding {n} embeddings to FAISS in batches of {max_batch}")

        if progress_callback:
            progress_callback(0.9, f"💾 Indexation FAISS...")

        for start in range(0, n, max_batch):
            end = start + max_batch
            _log.debug(f"[INGEST] FAISS add batch {start}:{end}")
            col.add(
                documents=all_chunks[start:end],
                metadatas=all_metas[start:end],
                embeddings=embeddings[start:end].tolist(),
                ids=all_faiss_ids[start:end],
            )

        total_chunks = len(all_chunks)

        if progress_callback:
            progress_callback(0.95, f"✅ {total_chunks} chunks indexés")

    _log.info("[INGEST] Completed")

    # =====================================================================
    # Nettoyage explicite pour assurer la synchronisation sur stockage réseau
    # =====================================================================
    # FAISS sauvegarde automatiquement après chaque add() dans notre implémentation,
    # mais on garde un délai pour que Windows synchronise les fichiers vers le réseau.

    try:
        _log.info("[INGEST] FAISS cleanup for network storage synchronization...")

        # Libérer les ressources Python
        del col
        del client
        gc.collect()

        _log.info("[INGEST] FAISS store closed and resources freed")

        # Délai pour que Windows synchronise les fichiers FAISS vers le réseau
        _log.info("[INGEST] Waiting 2 seconds for OS to flush data to network storage...")
        time.sleep(2)

        _log.info("[INGEST] ✅ Database fully synchronized - safe to shut down PC")

    except Exception as e:
        _log.warning(f"[INGEST] Error during cleanup (database may still be OK): {e}")

    return {"total_chunks": total_chunks, "files": file_reports}
