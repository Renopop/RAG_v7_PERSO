# streamlit_RAG.py
# Interface Streamlit pour :
#  - ingestion de documents dans FAISS (via CSV ou upload direct)
#  - question / réponse RAG via Snowflake + DALLEM
#
# Cette version :
#  - sépare clairement l’onglet d’ingestion et l’onglet RAG
#  - le bouton "Lancer l'ingestion" n’apparaît QUE dans l’onglet Ingestion

import os
import tempfile
import shutil
import hashlib
import shutil
import csv
import io
import getpass
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
from contextlib import nullcontext

import streamlit as st

from core.models_utils import EMBED_MODEL, LLM_MODEL, make_logger
# Utilisation du nouveau pipeline optimisé avec affichage détaillé
from ingestion.ingestion_pipeline import ingest_documents
from core.faiss_store import build_faiss_store, list_all_collections
from query.rag_query import run_rag_query, run_multi_collection_rag_query, MULTI_COLLECTION_AVAILABLE
from core.faiss_store import get_cache_manager, LocalCacheManager
from feedback.feedback_store import FeedbackStore, create_feedback, QueryFeedback
from core.config_manager import (
    load_config,
    save_config,
    is_config_valid,
    render_config_page_streamlit,
    StorageConfig,
    validate_all_directories,
    create_directory,
)
from processing.xml_processing import (
    XMLParseConfig,
    SectionPattern,
    PATTERN_DESCRIPTIONS,
    analyze_xml,
    preview_sections,
)

# Optionnel : fonction d'extraction des pièces jointes PDF si disponible
try:
    from processing.pdf_processing import extract_attachments_from_pdf
except ImportError:
    extract_attachments_from_pdf = None

# Optionnel : support Confluence
try:
    from processing.confluence_processing import (
        test_confluence_connection,
        list_spaces,
        get_space_info,
        extract_text_from_confluence_space,
        group_pages_by_section,
    )
    CONFLUENCE_AVAILABLE = True
except ImportError:
    CONFLUENCE_AVAILABLE = False

logger = make_logger(debug=False)


def sanitize_collection_name(name: str) -> str:
    """
    Nettoie un nom pour en faire un nom de collection valide.
    - Supprime les accents (é→e, à→a, etc.)
    - Ne garde que les caractères alphanumériques, espaces, tirets, underscores
    - Convertit en minuscules
    - Limite à 50 caractères
    """
    # Normaliser et supprimer les accents
    normalized = unicodedata.normalize('NFD', name)
    without_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    # Ne garder que les caractères sûrs
    safe = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in without_accents)
    return safe.lower().strip()[:50]


# =====================================================================
#  CACHE POUR LES RESSOURCES ET REQUÊTES RAG (amélioration performance)
# =====================================================================

@st.cache_resource(ttl=600, show_spinner=False)  # Cache 10 minutes (singleton)
def get_cached_faiss_store(db_path: str, use_local_cache: bool = False):
    """Retourne un store FAISS caché pour éviter les reconstructions."""
    return build_faiss_store(db_path, use_local_cache=use_local_cache, lazy_load=True)


@st.cache_data(ttl=1800, show_spinner=False)  # Cache 30 minutes
def cached_rag_query(
    db_path: str,
    collection_name: str,
    question: str,
    top_k: int = 30,
    synthesize_all: bool = False,
    use_feedback_reranking: bool = False,
    use_query_expansion: bool = True,
    use_bge_reranker: bool = True,
) -> dict:
    """
    Version cachée de run_rag_query pour éviter les recalculs sur requêtes identiques.
    Le cache est invalidé après 30 minutes (ttl=1800).
    Utilise automatiquement le cache local si disponible.
    """
    return run_rag_query(
        db_path=db_path,
        collection_name=collection_name,
        question=question,
        top_k=top_k,
        synthesize_all=synthesize_all,
        log=logger,
        feedback_store=None,  # Pas de feedback dans le cache (sinon problèmes de sérialisation)
        use_feedback_reranking=use_feedback_reranking,
        use_query_expansion=use_query_expansion,
        use_bge_reranker=use_bge_reranker,
        use_local_cache=True,  # Toujours actif - utilise cache local si disponible
    )


@st.cache_data(ttl=1800, show_spinner=False)  # Cache 30 minutes
def cached_multi_collection_rag_query(
    db_path: str,
    collection_names: tuple,  # tuple pour hashabilité (cache)
    question: str,
    top_k: int = 30,
    use_hybrid_search: bool = True,
    use_bge_reranker: bool = True,
) -> dict:
    """
    Version cachée de run_multi_collection_rag_query pour recherche inter-bases.
    Interroge plusieurs collections simultanément et fusionne les résultats.
    """
    if not MULTI_COLLECTION_AVAILABLE:
        raise ImportError("Multi-collection search not available")

    return run_multi_collection_rag_query(
        db_path=db_path,
        collection_names=list(collection_names),  # Reconvertir en list
        question=question,
        top_k=top_k,
        call_llm=True,
        log=logger,
        use_hybrid_search=use_hybrid_search,
        use_bge_reranker=use_bge_reranker,
        use_lost_in_middle=True,
        use_query_understanding=True,
        use_answer_grounding=True,
    )


# =====================================================================
#  CONFIG LOCALE - CHARGÉE DEPUIS config.json
# =====================================================================
# La configuration est maintenant gérée par config_manager.py
# Si les répertoires ne sont pas accessibles, une page de configuration
# sera affichée pour permettre à l'utilisateur de les configurer.

# Charger la configuration
_config = load_config()

# Variables globales pour compatibilité avec le code existant
BASE_ROOT_DIR = _config.base_root_dir
CSV_IMPORT_DIR = _config.csv_import_dir
CSV_EXPORT_DIR = _config.csv_export_dir
FEEDBACK_DIR = _config.feedback_dir


def _init_feedback_store():
    """Initialise le store de feedbacks de manière sécurisée."""
    try:
        if os.path.exists(FEEDBACK_DIR) or os.access(os.path.dirname(FEEDBACK_DIR), os.W_OK):
            os.makedirs(FEEDBACK_DIR, exist_ok=True)
            return FeedbackStore(FEEDBACK_DIR)
    except Exception as e:
        logger.warning(f"[CONFIG] Impossible d'initialiser le feedback store: {e}")
    return None


# Initialisation du store de feedbacks (peut être None si répertoire inaccessible)
feedback_store = _init_feedback_store()



def build_attachment_csv_path(source_path: str, attachment_filename: str) -> str:
    """
    Construit le chemin logique pour une pièce jointe à enregistrer dans le CSV.
    Format : dir(source) / (stem(source) + "-" + attachment_filename)
    Exemple :
        source_path = D:\test\toto.pdf
        attachment_filename = tata.pdf
        -> D:\test\toto-tata.pdf
    """
    directory = os.path.dirname(source_path)
    stem, _ = os.path.splitext(os.path.basename(source_path))
    return os.path.join(directory, f"{stem}-{attachment_filename}")


def get_tracking_csv_path(base_name: str) -> str:
    """
    Retourne le chemin du fichier CSV de tracking pour une base donnée.
    Format : documents_ingeres_[nom_base].csv
    """
    if not CSV_EXPORT_DIR or CSV_EXPORT_DIR == "A_COMPLETER_CHEMIN_EXPORT_CSV":
        return None
    csv_filename = f"documents_ingeres_{base_name}.csv"
    return os.path.join(CSV_EXPORT_DIR, csv_filename)


def append_to_tracking_csv(base_name: str, entries: List[Tuple[str, str]], log_func=None) -> bool:
    """
    Ajoute des entrées au CSV de tracking pour une base donnée.

    Cette fonction écrit immédiatement dans le fichier CSV, permettant
    de sauvegarder la progression même si l'ingestion est interrompue.

    Args:
        base_name: Nom de la base FAISS
        entries: Liste de tuples (group, path) à ajouter
        log_func: Fonction de logging optionnelle

    Returns:
        True si succès, False sinon
    """
    if not entries:
        return True

    csv_path = get_tracking_csv_path(base_name)
    if not csv_path:
        return False

    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Vérifier si le fichier existe et a un header
        file_exists = os.path.exists(csv_path)
        needs_header = not file_exists

        if file_exists:
            # Vérifier si le fichier est vide ou n'a pas de header
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                first_line = f.readline().strip()
                if not first_line or first_line.lower() != "group;path":
                    needs_header = True

        # Ouvrir en mode append
        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";")

            if needs_header:
                writer.writerow(["group", "path"])

            for group, path in entries:
                writer.writerow([group, path])

        if log_func:
            log_func(f"[TRACKING] {len(entries)} entrée(s) ajoutée(s) au CSV de tracking pour '{base_name}'")

        return True

    except Exception as e:
        if log_func:
            log_func(f"[TRACKING-ERROR] Erreur écriture tracking pour '{base_name}': {e}")
        return False

# ========================
#   Gestion des verrous d'ingestion (coordination multi-utilisateurs)
# ========================

def create_ingestion_lock(base_root: str, base_name: str) -> bool:
    """
    Crée un fichier de verrou pour indiquer qu'une ingestion est en cours.
    Retourne True si le verrou a été créé, False si la base est déjà verrouillée.
    """
    import datetime
    import socket

    lock_dir = os.path.join(base_root, base_name)
    os.makedirs(lock_dir, exist_ok=True)

    lock_file = os.path.join(lock_dir, ".ingestion_lock")

    # Vérifier si le verrou existe déjà
    if os.path.exists(lock_file):
        return False

    # Créer le fichier de verrou avec des infos
    try:
        with open(lock_file, "w", encoding="utf-8") as f:
            f.write(f"timestamp={datetime.datetime.now().isoformat()}\n")
            f.write(f"hostname={socket.gethostname()}\n")
            f.write(f"user={os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))}\n")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la création du verrou : {e}")
        return False


def remove_ingestion_lock(base_root: str, base_name: str) -> None:
    """Supprime le fichier de verrou d'ingestion."""
    lock_file = os.path.join(base_root, base_name, ".ingestion_lock")
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
            logger.info(f"Verrou d'ingestion supprimé pour {base_name}")
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du verrou : {e}")


def is_base_locked(base_root: str, base_name: str, timeout_hours: int = 2) -> bool:
    """
    Vérifie si une base est verrouillée pour ingestion.
    Si le verrou est plus vieux que timeout_hours, il est considéré comme orphelin et supprimé.
    """
    import datetime

    lock_file = os.path.join(base_root, base_name, ".ingestion_lock")

    if not os.path.exists(lock_file):
        return False

    # Vérifier l'âge du verrou
    try:
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(lock_file))
        age = datetime.datetime.now() - mod_time

        if age.total_seconds() > timeout_hours * 3600:
            logger.warning(f"Verrou orphelin détecté pour {base_name} (âge: {age}). Suppression automatique.")
            remove_ingestion_lock(base_root, base_name)
            return False

        return True
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du verrou : {e}")
        return False


def get_lock_info(base_root: str, base_name: str) -> Optional[Dict[str, str]]:
    """Retourne les informations du verrou (timestamp, user, hostname) ou None."""
    import datetime

    lock_file = os.path.join(base_root, base_name, ".ingestion_lock")

    if not os.path.exists(lock_file):
        return None

    try:
        info = {}
        with open(lock_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    info[key] = value

        # Calculer la durée
        if "timestamp" in info:
            lock_time = datetime.datetime.fromisoformat(info["timestamp"])
            duration = datetime.datetime.now() - lock_time
            info["duration"] = str(duration).split('.')[0]  # Format HH:MM:SS

        return info
    except Exception as e:
        logger.error(f"Erreur lors de la lecture des infos du verrou : {e}")
        return None


# ========================
#   Utils CSV / bases / collections
# ========================

@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 minutes
def list_bases(base_root: str) -> List[str]:
    """Liste les bases FAISS (sous-dossiers) dans base_root."""
    p = Path(base_root)
    if not p.exists() or not p.is_dir():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 minutes
def list_collections_for_base(base_root: str, base_name: str) -> List[str]:
    """Liste les collections d'une base donnée (en interrogeant FAISS)."""
    db_path = os.path.join(base_root, base_name)
    try:
        store = get_cached_faiss_store(db_path)
        colls = store.list_collections()  # FAISS retourne directement une liste de noms
        logger.debug(f"[list_collections] Base {base_name}: {len(colls)} collections trouvées")
        return sorted(colls)
    except Exception as e:
        logger.error(f"[list_collections] Erreur pour {base_name}: {e}")
        return []


@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 minutes
def get_collection_doc_counts(base_root: str, base_name: str) -> Dict[str, int]:
    """
    Retourne un dict {nom_collection: nombre_d'éléments} pour une base donnée.
    Le nombre correspond au nombre de chunks indexés dans la collection.
    """
    db_path = os.path.join(base_root, base_name)
    counts: Dict[str, int] = {}
    try:
        store = get_cached_faiss_store(db_path)
        for col_name in store.list_collections():  # FAISS retourne directement les noms
            try:
                collection = store.get_collection(col_name)
                n = collection.count()
            except Exception:
                n = 0
            counts[col_name] = n
    except Exception as e:
        logger.warning(f"[UI] Impossible de compter les documents pour base={base_name} : {e}")
    return counts


def parse_csv_groups_and_paths(file_bytes: bytes) -> Dict[str, List[str]]:
    """Parse un CSV bytes -> dict {group: [paths...]}.

    - séparateur ';' ou ',' (détection automatique)
    - en-tête possible : group;path
    - sinon, col0 = group, col1 = path
    - nettoyage du BOM éventuel sur 'group'
    """
    # Décoder les bytes en texte, gérer BOM UTF-8
    try:
        text = file_bytes.decode("utf-8-sig", errors="ignore")
    except Exception:
        text = file_bytes.decode("utf-8", errors="ignore")

    # Normaliser les sauts de ligne (Windows \r\n -> \n)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Créer le buffer StringIO
    buf = io.StringIO(text)

    # Détecter le délimiteur
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"

    reader = csv.reader(buf, delimiter=delimiter)
    rows = list(reader)

    if not rows:
        return {}

    header = rows[0]
    header = [h.lstrip("\ufeff") for h in header]
    header_lower = [h.strip().lower() for h in header]

    if "group" in header_lower and "path" in header_lower:
        idx_group = header_lower.index("group")
        idx_path = header_lower.index("path")
        data_rows = rows[1:]
    else:
        idx_group = 0
        idx_path = 1 if len(header) > 1 else 0
        data_rows = rows

    groups: Dict[str, List[str]] = {}
    for r in data_rows:
        if len(r) <= max(idx_group, idx_path):
            continue
        g = (r[idx_group] or "").strip().lstrip("\ufeff")
        p = (r[idx_path] or "").strip()
        if not p:
            continue
        if not g:
            g = "ALL"
        groups.setdefault(g, []).append(p)

    return groups


# ========================
#   PAGE CONFIG
# ========================

st.set_page_config(
    page_title="RaGME_UP - PROP",
    layout="wide",
)

# ========================
#   ACTUALISATION DU CACHE AU CHARGEMENT
# ========================
# Vider le cache des collections au premier chargement de la session
# pour garantir des données fraîches depuis le réseau
if "cache_cleared_on_load" not in st.session_state:
    st.session_state["cache_cleared_on_load"] = True
    # Vider les caches des listes uniquement (PAS les stores FAISS)
    list_bases.clear()
    list_collections_for_base.clear()
    get_collection_doc_counts.clear()

# ========================
#   VÉRIFICATION DE LA CONFIGURATION
# ========================
# Si les répertoires ne sont pas accessibles, afficher la page de configuration

if not is_config_valid(_config):
    st.title("Configuration des répertoires")

    st.warning("Les répertoires de stockage ne sont pas accessibles. Veuillez les configurer.")

    # Afficher le statut de chaque répertoire
    results = validate_all_directories(_config)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Statut des répertoires")

        for key, (valid, message, label, path) in results.items():
            if valid:
                st.success(f"✅ **{label}**\n\n`{path}`")
            else:
                st.error(f"❌ **{label}**\n\n`{path}`\n\n_{message}_")

    with col2:
        st.subheader("Actions")

        # Bouton pour créer les répertoires manquants
        if st.button("Créer les répertoires manquants", type="primary"):
            created = []
            failed = []

            for key, (valid, message, label, path) in results.items():
                if not valid:
                    success, msg = create_directory(path)
                    if success:
                        created.append(label)
                    else:
                        failed.append(f"{label}: {msg}")

            if created:
                st.success(f"Créés: {', '.join(created)}")
            if failed:
                st.error(f"Échecs: {', '.join(failed)}")

            st.rerun()

        st.markdown("---")

        # Formulaire pour modifier les chemins
        st.subheader("Modifier les chemins")

        new_base_root = st.text_input("Bases FAISS", value=_config.base_root_dir)
        new_csv_import = st.text_input("CSV ingestion", value=_config.csv_import_dir)
        new_csv_export = st.text_input("CSV tracking", value=_config.csv_export_dir)
        new_feedback = st.text_input("Feedbacks", value=_config.feedback_dir)

        if st.button("Sauvegarder la configuration"):
            new_config = StorageConfig(
                base_root_dir=new_base_root,
                csv_import_dir=new_csv_import,
                csv_export_dir=new_csv_export,
                feedback_dir=new_feedback,
            )
            if save_config(new_config):
                st.success("Configuration sauvegardée!")
                st.rerun()
            else:
                st.error("Erreur lors de la sauvegarde")

    st.info("""
    **Aide:**
    - Les chemins doivent être des chemins absolus
    - Les répertoires doivent être accessibles en lecture et écriture
    - Cliquez sur "Créer les répertoires manquants" pour créer les dossiers automatiquement
    - Ou modifiez les chemins et cliquez sur "Sauvegarder la configuration"
    """)

    st.stop()  # Arrêter l'exécution du reste de l'app

# ========================
#   APPLICATION PRINCIPALE
# ========================

# Utilisateur admin (seul à voir tous les onglets)
ADMIN_USER = "agdgtrl"
current_user = getpass.getuser().lower()

st.title("RaGME_UP - PROP")

# ========================
#   DOCUMENTATION ET AIDE (sous le titre)
# ========================
# Chemin vers la racine du projet (parent de ui/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DOC_FILES = {
    "📋 README": os.path.join(PROJECT_ROOT, "README.md"),
    "👤 Guide Utilisateur": os.path.join(PROJECT_ROOT, "docs", "GUIDE_UTILISATEUR.md"),
    "🔧 Architecture Technique": os.path.join(PROJECT_ROOT, "docs", "ARCHITECTURE_TECHNIQUE.md"),
    "🌐 Installation Réseau": os.path.join(PROJECT_ROOT, "docs", "INSTALLATION_RESEAU.md")
}

col_doc, col_help = st.columns([3, 1])
with col_doc:
    selected_doc = st.selectbox(
        "📚 Documentation",
        options=[""] + list(DOC_FILES.keys()),
        format_func=lambda x: "Choisir un document..." if x == "" else x,
        key="doc_select_main"
    )
with col_help:
    with st.expander("❓ Aide"):
        st.markdown("""
        **Onglets :**
        - **Gestion CSV** : Créer/modifier CSV
        - **Ingestion** : Charger documents
        - **Questions RAG** : Interroger
        """)

if selected_doc and selected_doc in DOC_FILES:
    doc_path = DOC_FILES[selected_doc]  # Chemins absolus dans DOC_FILES
    with st.expander(f"📄 {selected_doc}", expanded=True):
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# ========================
#   SIDEBAR (admin uniquement)
# ========================

# Définir les variables par défaut (utilisées même si sidebar cachée)
base_root = BASE_ROOT_DIR
bases = list_bases(base_root)
use_easa_sections = st.session_state.get("use_easa_sections", False)

# Sidebar visible uniquement pour admin
if current_user == ADMIN_USER:
    with st.sidebar:
        st.header("⚙️ Configuration globale")

        base_root = BASE_ROOT_DIR
        st.warning(
            f"📁 Bases FAISS : `{BASE_ROOT_DIR}`"
        )
        st.warning(
            f"📝 CSV d'ingestion : `{CSV_IMPORT_DIR}`"
        )
        st.warning(
            f"📊 CSV de tracking : `{CSV_EXPORT_DIR}`"
        )
        bases = list_bases(base_root)
        if bases:
            st.success(f"✅ {len(bases)} base(s) trouvée(s) sous {base_root}")
        else:
            st.info("ℹ️ Aucune base trouvée sous ce dossier pour l'instant.")

        st.markdown("---")

        st.markdown("### 🤖 Modèles utilisés")
        st.caption(f"🔹 Embeddings : **Snowflake** – `{EMBED_MODEL}`")
        st.caption(f"🔹 LLM : **DALLEM** – `{LLM_MODEL}`")


# ========================
#   TABS
# ========================
# Seul admin voit tous les onglets
if current_user == ADMIN_USER:
    tab_csv, tab_ingest, tab_confluence, tab_purge, tab_rag, tab_analytics = st.tabs(
        ["📝 Gestion CSV", "📥 Ingestion documents", "🌐 Confluence", "🗑️ Purge des bases", "❓ Questions RAG", "📊 Tableau de bord"]
    )
else:
    tab_csv, tab_ingest, tab_rag = st.tabs(
        ["📝 Gestion CSV", "📥 Ingestion documents", "❓ Questions RAG"]
    )
    tab_confluence = None
    tab_purge = None
    tab_analytics = None


# ========================
#   TAB GESTION CSV
# ========================
with tab_csv:
    st.subheader("📝 Gestion des fichiers CSV d'ingestion")

    # Initialiser le mode dans session_state
    if "csv_mode" not in st.session_state:
        st.session_state.csv_mode = None

    # Deux gros boutons pour choisir le mode
    st.markdown("### Choisissez une action")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button(
            "📝 Création d'un CSV",
            type="primary",
            use_container_width=True,
            help="Lance l'interface GUI moderne pour créer un CSV avec vrais chemins"
        ):
            # Lancer directement l'application GUI
            import subprocess
            import sys

            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["python", "csv_generator_gui.py"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    subprocess.Popen(["python", "csv_generator_gui.py"])

                st.success("✅ Application GUI lancée ! Vérifiez vos fenêtres ouvertes.")
                st.info("💡 Utilisez l'application pour créer votre CSV, puis revenez ici pour l'ingérer.")
            except Exception as e:
                st.error(f"❌ Erreur lors du lancement : {e}")
                st.info("💡 Lancez manuellement dans un terminal :")
                st.code("python csv_generator_gui.py", language="bash")

    with col_btn2:
        if st.button(
            "📝 Éditer un CSV",
            type="primary",
            use_container_width=True,
            help="Lance l'interface GUI pour éditer un CSV existant"
        ):
            # Lancer la GUI sans argument - l'utilisateur choisira le CSV dans l'interface
            import subprocess
            import sys

            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["python", "csv_generator_gui.py"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    subprocess.Popen(["python", "csv_generator_gui.py"])

                st.success("✅ Application GUI lancée !")
                st.info("💡 Utilisez le bouton '📂 Ouvrir un CSV' dans l'interface pour charger un CSV existant.")
            except Exception as e:
                st.error(f"❌ Erreur lors du lancement : {e}")
                st.info("💡 Lancez manuellement dans un terminal :")
                st.code("python csv_generator_gui.py", language="bash")

    st.markdown("---")


# ========================
#   TAB INGESTION
# ========================
with tab_ingest:
    st.subheader("📥 Ingestion de documents via CSV")

    chunk_size = 1000

    # Initialiser le flag d'arrêt d'ingestion
    if "stop_ingestion" not in st.session_state:
        st.session_state["stop_ingestion"] = False
    if "ingestion_running" not in st.session_state:
        st.session_state["ingestion_running"] = False
    if "created_locks" not in st.session_state:
        st.session_state["created_locks"] = []

    # Nettoyage des verrous si l'ingestion a été interrompue
    if st.session_state["stop_ingestion"] and st.session_state["created_locks"]:
        for base_name in st.session_state["created_locks"]:
            remove_ingestion_lock(base_root, base_name)
            logger.info(f"Verrou d'ingestion nettoyé après arrêt pour {base_name}")
        st.session_state["created_locks"] = []
        st.session_state["stop_ingestion"] = False
        st.session_state["ingestion_running"] = False
        st.session_state["bulk_update_all"] = False  # Réinitialiser la sélection globale
        st.rerun()

    # Sélection de CSV depuis CSV_IMPORT_DIR
    st.markdown(f"### 📂 Sélectionner des CSV depuis `{CSV_IMPORT_DIR}`")

    # Lister les fichiers CSV disponibles dans CSV_IMPORT_DIR
    available_csvs = []
    if CSV_IMPORT_DIR and os.path.isdir(CSV_IMPORT_DIR):
        available_csvs = sorted([
            f for f in os.listdir(CSV_IMPORT_DIR)
            if f.lower().endswith('.csv')
        ])

    if available_csvs:
        # Bouton pour mise à jour globale de toutes les bases (admin uniquement)
        if current_user == ADMIN_USER:
            col_update_all, col_count = st.columns([2, 1])
            with col_update_all:
                if st.button(
                    "🔄 Mise à jour de toutes les bases",
                    type="primary",
                    use_container_width=True,
                    help="Traite automatiquement TOUS les fichiers CSV du répertoire"
                ):
                    st.session_state["bulk_update_all"] = True
                    st.rerun()
            with col_count:
                st.info(f"📊 **{len(available_csvs)}** CSV disponibles")

        # Si mise à jour globale demandée (admin uniquement), pré-sélectionner tous les CSV
        if current_user == ADMIN_USER and st.session_state.get("bulk_update_all", False):
            st.warning(f"⚠️ **Mise à jour globale** : {len(available_csvs)} fichiers CSV seront traités. Cela peut prendre plusieurs minutes.")
            default_selection = available_csvs
        else:
            default_selection = []

        selected_csv_files = st.multiselect(
            "Fichiers CSV disponibles",
            options=available_csvs,
            default=default_selection,
            help=f"Sélectionnez un ou plusieurs CSV depuis {CSV_IMPORT_DIR}"
        )

        # Bouton pour annuler la sélection globale (admin uniquement)
        if current_user == ADMIN_USER and st.session_state.get("bulk_update_all", False):
            if st.button("❌ Annuler la sélection globale"):
                st.session_state["bulk_update_all"] = False
                st.rerun()
    else:
        selected_csv_files = []
        if not CSV_IMPORT_DIR:
            st.warning("⚠️ CSV_IMPORT_DIR non configuré")
        elif not os.path.isdir(CSV_IMPORT_DIR):
            st.warning(f"⚠️ Répertoire inexistant : {CSV_IMPORT_DIR}")
        else:
            st.info("Aucun fichier CSV trouvé dans le répertoire")

    st.markdown("---")
    st.markdown("### 📤 Ou uploader des CSV")

    uploaded_csvs = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=True,
        key="csv_ingest",
        label_visibility="collapsed"
    )

    # Combiner les CSV sélectionnés et uploadés
    csv_files_to_process = []

    # Ajouter les CSV sélectionnés depuis le répertoire
    for csv_name in selected_csv_files:
        csv_path = os.path.join(CSV_IMPORT_DIR, csv_name)
        if os.path.exists(csv_path):
            csv_files_to_process.append(("local", csv_name, csv_path))

    # Ajouter les CSV uploadés
    if uploaded_csvs:
        for uploaded in uploaded_csvs:
            csv_files_to_process.append(("uploaded", uploaded.name, uploaded))

    # ========================================================================
    # OPTION EASA - Auto-activation pour base CERTIFICATION
    # ========================================================================
    # Détecter les bases sélectionnées
    selected_bases = set()
    for source_type, csv_name, _ in csv_files_to_process:
        base_name = Path(csv_name).stem.upper()
        selected_bases.add(base_name)

    # Vérifier si CERTIFICATION est dans les bases sélectionnées
    has_certification = "CERTIFICATION" in selected_bases

    if has_certification:
        # Auto-activation pour CERTIFICATION
        use_easa_sections = True
        st.info("✈️ **Mode EASA activé automatiquement** pour la base CERTIFICATION (découpage CS/AMC/GM)")
    else:
        # Option manuelle pour les autres bases
        if "use_easa_sections" not in st.session_state:
            st.session_state["use_easa_sections"] = False

        use_easa_sections = st.checkbox(
            "✈️ Utiliser les sections EASA (CS / AMC / GM)",
            value=st.session_state["use_easa_sections"],
            help=(
                "Quand activé, le texte est d'abord découpé par sections CS/AMC/GM "
                "avant le chunking. Activé automatiquement pour la base CERTIFICATION."
            ),
        )
        st.session_state["use_easa_sections"] = use_easa_sections

    # Afficher l'état des bases (indicateur de disponibilité pour coordination)
    st.markdown("---")
    st.markdown("### 📊 État des bases (coordination multi-utilisateurs)")

    if bases:
        status_cols = st.columns(min(len(bases), 4))  # Max 4 colonnes
        for idx, base in enumerate(bases):
            col_idx = idx % 4
            with status_cols[col_idx]:
                is_locked = is_base_locked(base_root, base)
                if is_locked:
                    lock_info = get_lock_info(base_root, base)
                    user = lock_info.get("user", "unknown") if lock_info else "unknown"
                    duration = lock_info.get("duration", "?") if lock_info else "?"
                    st.error(f"🔴 **{base}**")
                    st.caption(f"Ingestion en cours\nUtilisateur: {user}\nDurée: {duration}")
                else:
                    st.success(f"🟢 **{base}**")
                    st.caption("Disponible")
    else:
        st.info("Aucune base trouvée")

    st.markdown("---")

    # Pour reconstruire un CSV récapitulatif : base;group;path
    ingested_entries: List[Tuple[str, str, str]] = []
    logical_paths: Dict[str, str] = {}

    # Statistiques d'ingestion
    ingestion_stats = {
        "csv_new_files": 0,
        "csv_missing_files": 0,
        "csv_skipped_existing": 0,
        "csv_attachments": 0,
    }

    # Récapitulatif des fichiers ingérés pendant cette exécution
    session_ingested_files = []

    # ========================================================================
    # DÉTECTION ET CONFIGURATION DES FICHIERS XML
    # ========================================================================

    # Initialiser les états de session pour XML
    if "xml_configs" not in st.session_state:
        st.session_state["xml_configs"] = {}  # {path: XMLParseConfig}
    if "xml_preview_validated" not in st.session_state:
        st.session_state["xml_preview_validated"] = False
    if "detected_xml_files" not in st.session_state:
        st.session_state["detected_xml_files"] = []

    def detect_xml_files_in_csvs(csv_list) -> List[str]:
        """Analyse les CSV (locaux ou uploadés) pour trouver les fichiers XML."""
        xml_files = []
        for source_type, csv_name, csv_source in csv_list:
            # Lire le contenu selon la source
            if source_type == "local":
                with open(csv_source, "rb") as f:
                    content = f.read()
            else:
                csv_source.seek(0)
                content = csv_source.read()
                csv_source.seek(0)
            groups = parse_csv_groups_and_paths(content)
            for group_name, paths in groups.items():
                for path in paths:
                    if path.lower().endswith(".xml") and os.path.isfile(path):
                        if path not in xml_files:
                            xml_files.append(path)
        return xml_files

    # Détecter les fichiers XML quand des CSV sont sélectionnés/uploadés
    xml_files_detected = []
    if csv_files_to_process:
        xml_files_detected = detect_xml_files_in_csvs(csv_files_to_process)
        st.session_state["detected_xml_files"] = xml_files_detected

    # Afficher l'interface de prévisualisation XML si des fichiers XML sont détectés
    if xml_files_detected:
        st.markdown("---")
        st.markdown("### 📄 Fichiers XML détectés - Normes EASA")
        st.info(f"**{len(xml_files_detected)} fichier(s) XML** détecté(s). "
                "Choisissez le pattern de découpage pour chaque fichier.")

        for xml_path in xml_files_detected:
            with st.expander(f"📄 {os.path.basename(xml_path)}", expanded=True):
                try:
                    # Sélection du pattern de découpage
                    pattern_options = list(SectionPattern)
                    pattern_labels = [PATTERN_DESCRIPTIONS[p] for p in pattern_options]

                    selected_idx = st.selectbox(
                        "Pattern de découpage",
                        range(len(pattern_options)),
                        index=pattern_options.index(SectionPattern.ALL_EASA),
                        format_func=lambda i: pattern_labels[i],
                        key=f"pattern_{xml_path}",
                        help="Choisissez comment découper le document en sections"
                    )
                    selected_pattern = pattern_options[selected_idx]

                    # Champ pour pattern custom
                    custom_regex = None
                    if selected_pattern == SectionPattern.CUSTOM:
                        custom_regex = st.text_input(
                            "Pattern regex personnalisé",
                            value=r"(SECTION[-\s]?\d+)",
                            key=f"custom_pattern_{xml_path}",
                            help="Entrez une expression régulière. Le groupe (1) sera utilisé comme code de section."
                        )
                        st.caption("Exemples: `(CHAPTER\\s+\\d+)`, `(ART\\.?\\s*\\d+)`, `(§\\s*\\d+\\.\\d+)`")

                    # Créer la config
                    config = XMLParseConfig(
                        pattern_type=selected_pattern,
                        custom_pattern=custom_regex
                    )

                    # Analyser avec le pattern sélectionné
                    analysis = analyze_xml(xml_path, config)

                    if analysis.get("error"):
                        st.error(f"Erreur: {analysis['error']}")
                        continue

                    # Stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Caractères", f"{analysis['total_chars']:,}")
                    with col2:
                        st.metric("📑 Sections détectées", analysis['sections_count'])

                    # Liste des sections trouvées
                    if analysis['sections']:
                        st.markdown("**Sections trouvées:**")
                        sections_display = []
                        for sec in analysis['sections'][:15]:
                            title_part = f" - {sec['title']}" if sec['title'] else ""
                            sections_display.append(f"• **{sec['code']}**{title_part} ({sec['length']:,} car.)")

                        st.markdown("\n".join(sections_display))

                        if analysis['sections_count'] > 15:
                            st.caption(f"... et {analysis['sections_count'] - 15} autres sections")

                    # Prévisualisation du texte
                    if st.button(f"👁️ Voir aperçu complet", key=f"preview_{xml_path}"):
                        preview_text, _ = preview_sections(xml_path, config, max_sections=20)
                        st.text_area("Aperçu", preview_text, height=400, key=f"preview_area_{xml_path}")

                    # Sauvegarder la config
                    st.session_state["xml_configs"][xml_path] = config

                except Exception as e:
                    st.error(f"Erreur: {e}")

        # Validation
        if st.button("✅ Confirmer et continuer", type="primary"):
            st.session_state["xml_preview_validated"] = True
            st.success("Configuration validée !")

        if not st.session_state["xml_preview_validated"]:
            st.warning("⚠️ Cliquez sur 'Confirmer et continuer' pour valider.")

        st.markdown("---")

    # Bouton d'ingestion et bouton stop
    can_ingest = True
    if xml_files_detected and not st.session_state["xml_preview_validated"]:
        can_ingest = False

    # Callback pour le bouton stop
    def stop_ingestion_callback():
        st.session_state["stop_ingestion"] = True

    # Afficher les boutons côte à côte
    col_start, col_stop = st.columns([3, 1])
    with col_start:
        start_button = st.button(
            "🚀 Lancer l'ingestion",
            help="Lance l'ingestion des documents listés dans le CSV. Les fichiers sont découpés en chunks, vectorisés et indexés dans FAISS.",
            disabled=not can_ingest
        )
    with col_stop:
        # Placeholder pour le bouton stop (visible uniquement pendant l'ingestion)
        stop_button_placeholder = st.empty()

    if start_button:
        # Réinitialiser le flag d'arrêt
        st.session_state["stop_ingestion"] = False
        st.session_state["ingestion_running"] = True
        # ------------------------------------------------------------------
        # Charger les CSV de tracking par base existants pour éviter de ré-ingérer
        # des fichiers déjà traités.
        # Clé utilisée : (base, group, path)
        # Structure : {base_name: {(group, path): True}}
        # ------------------------------------------------------------------
        existing_entries_by_base: Dict[str, Dict[Tuple[str, str], bool]] = {}

        def load_tracking_csv_for_base(base_name: str) -> Dict[Tuple[str, str], bool]:
            """Charge le CSV de tracking pour une base donnée."""
            entries = {}
            csv_path = get_tracking_csv_path(base_name)
            if csv_path and os.path.exists(csv_path):
                try:
                    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f_csv:
                        reader = csv.reader(f_csv, delimiter=";")
                        rows = list(reader)
                        # Skip header if present
                        start_idx_rows = 1 if rows and rows[0] and rows[0][0].lower() == "group" else 0
                        for r in rows[start_idx_rows:]:
                            if len(r) < 2:
                                continue
                            g, p = r[0].strip(), r[1].strip()
                            # Stocker à la fois le chemin original et normalisé pour une comparaison robuste
                            entries[(g, p)] = True
                            p_normalized = os.path.normpath(p)
                            if p_normalized != p:
                                entries[(g, p_normalized)] = True
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du CSV de tracking pour {base_name} : {e}")
            return entries

        # Rien à ingérer ?
        if not csv_files_to_process:
            st.warning("Aucun CSV d'ingestion sélectionné ou uploadé.")
        else:
            # Collecter les bases qui vont être modifiées (depuis les noms de CSV)
            bases_to_ingest = set()
            for source_type, csv_name, _ in csv_files_to_process:
                base_name = Path(csv_name).stem
                bases_to_ingest.add(base_name)

            # Vérifier qu'aucune base n'est déjà verrouillée
            locked_bases = []
            for base_name in bases_to_ingest:
                if is_base_locked(base_root, base_name):
                    locked_bases.append(base_name)

            if locked_bases:
                st.error(
                    f"❌ Impossible de lancer l'ingestion : les bases suivantes sont déjà en cours d'ingestion :\n\n"
                    + "\n".join([f"- **{b}**" for b in locked_bases])
                    + "\n\nVeuillez attendre la fin de l'ingestion en cours ou vérifier les indicateurs ci-dessus."
                )
                st.stop()

            # Créer les verrous pour toutes les bases
            created_locks = []
            st.session_state["created_locks"] = []  # Réinitialiser
            for base_name in bases_to_ingest:
                if create_ingestion_lock(base_root, base_name):
                    created_locks.append(base_name)
                    st.session_state["created_locks"].append(base_name)  # Sauvegarder pour cleanup
                    logger.info(f"Verrou d'ingestion créé pour {base_name}")
                else:
                    st.error(f"❌ Impossible de créer le verrou pour la base {base_name}")
                    # Nettoyer les verrous déjà créés
                    for b in created_locks:
                        remove_ingestion_lock(base_root, b)
                    st.session_state["created_locks"] = []
                    st.stop()

            try:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                log_box = st.empty()
                log_lines: List[str] = []

                # Afficher le bouton stop
                stop_button_placeholder.button(
                    "🛑 Stop",
                    type="secondary",
                    on_click=stop_ingestion_callback,
                    key="stop_ingestion_btn"
                )

                def check_stop() -> bool:
                    """Vérifie si l'utilisateur a demandé l'arrêt."""
                    return st.session_state.get("stop_ingestion", False)

                def log(msg: str) -> None:
                    logger.info(msg)
                    log_lines.append(msg)
                    log_box.text_area("🪵 Logs ingestion", value="\n".join(log_lines), height=280)

                def progress(frac: float, msg: str) -> None:
                    progress_bar.progress(min(max(frac, 0.0), 1.0))
                    status_text.info(msg)

                # ------------------------------------------------------------------
                # 1) Ingestion à partir des CSV
                # ------------------------------------------------------------------
                ingestion_stopped = False
                if csv_files_to_process:
                    csv_temp_dir = tempfile.mkdtemp(prefix="rag_ingest_csv_")
                    try:
                        for source_type, csv_name, csv_source in csv_files_to_process:
                            # Vérifier si l'utilisateur a demandé l'arrêt
                            if check_stop():
                                log("⚠️ Ingestion interrompue par l'utilisateur")
                                ingestion_stopped = True
                                break
                            # Lire le contenu selon la source (locale ou uploadée)
                            if source_type == "local":
                                with open(csv_source, "rb") as f:
                                    data = f.read()
                            else:
                                csv_source.seek(0)
                                data = csv_source.read()

                            groups = parse_csv_groups_and_paths(data)
                            if not groups:
                                st.warning(f"Aucune donnée exploitable dans le CSV {csv_name}")
                                continue

                            total_files_in_csv = sum(len(paths) for paths in groups.values())
                            base_name = Path(csv_name).stem
                            db_path = os.path.join(base_root, base_name)
                            os.makedirs(db_path, exist_ok=True)
                            st.write(f"📂 CSV `{csv_name}` → base `{base_name}` ({total_files_in_csv} fichiers)")
    
                            # Charger le CSV de tracking pour cette base
                            if base_name not in existing_entries_by_base:
                                existing_entries_by_base[base_name] = load_tracking_csv_for_base(base_name)
    
                            for group_name, paths in groups.items():
                                # Vérifier si l'utilisateur a demandé l'arrêt
                                if check_stop():
                                    log("⚠️ Ingestion interrompue par l'utilisateur")
                                    ingestion_stopped = True
                                    break

                                progress(0.05, f"[{base_name}/{group_name}] Validation des chemins…")
                                new_paths: List[str] = []
                                missing_paths: List[str] = []
    
                                # Validation automatique des chemins / détection fichiers manquants
                                for p in paths:
                                    if not os.path.exists(p):
                                        missing_paths.append(p)
                                        ingestion_stats["csv_missing_files"] += 1
                                        continue
                                    # Normaliser le chemin pour la comparaison (gère / vs \ et la casse sur Windows)
                                    p_normalized = os.path.normpath(p)
                                    key = (group_name, p_normalized)
                                    # Vérifier aussi avec le chemin original au cas où
                                    key_original = (group_name, p)
                                    if key in existing_entries_by_base[base_name] or key_original in existing_entries_by_base[base_name]:
                                        ingestion_stats["csv_skipped_existing"] += 1
                                        continue
                                    new_paths.append(p)

                                if missing_paths:
                                    st.warning(
                                        f"⚠️ {len(missing_paths)} fichier(s) introuvable(s) pour "
                                        f"base={base_name}, group={group_name}. Voir les logs pour le détail."
                                    )
                                    for mp in missing_paths:
                                        log(f"[MISSING] Fichier introuvable : {mp}")
    
                                # --- Extraction des pièces jointes des PDF listés dans le CSV ---
                                temp_dirs_to_cleanup = []  # Liste des répertoires temporaires à nettoyer
                                parent_paths = {}  # Dict {chemin_pj: chemin_fichier_parent} pour le bouton Ouvrir
                                if extract_attachments_from_pdf is not None:
                                    progress(0.10, f"[{base_name}/{group_name}] Analyse des pièces jointes PDF…")
                                    original_pdf_paths = [p for p in new_paths if p.lower().endswith(".pdf")]
                                    for pdf_path in original_pdf_paths:
                                        try:
                                            # Utiliser un répertoire temporaire pour les pièces jointes
                                            info = extract_attachments_from_pdf(
                                                pdf_path,
                                                output_dir=None,  # Utilise tempfile automatiquement
                                            )
                                        except Exception as e:
                                            log(
                                                f"[ATTACH] Erreur lors de l'extraction des pièces jointes de {pdf_path}: {e}"
                                            )
                                            info = {"attachments_paths": [], "pdf_without_attachments_path": None, "temp_dir": None}
    
                                        # Compatibilité : ancien retour (liste) ou nouveau (dict)
                                        if isinstance(info, dict):
                                            att_paths = info.get("attachments_paths", []) or []
                                            pdf_without = info.get("pdf_without_attachments_path")
                                            temp_dir = info.get("temp_dir")
                                            
                                            # Garder trace du répertoire temporaire pour le nettoyer après l'ingestion
                                            if temp_dir:
                                                temp_dirs_to_cleanup.append(temp_dir)
                                        else:
                                            att_paths = info or []
                                            pdf_without = None
                                            temp_dir = None
    
                                        # Le PDF sans PJ est créé mais PAS ingéré ET PAS ajouté au CSV
                                        if pdf_without:
                                            log(
                                                f"[ATTACH-INFO] PDF sans PJ créé (non ingéré, non tracé) : {os.path.basename(pdf_without)}"
                                            )
    
                                        if att_paths:
                                            log(
                                                f"[ATTACH] {len(att_paths)} pièce(s) jointe(s) détectée(s) dans {pdf_path}"
                                            )
    
                                        for att_path in att_paths:
                                            att_name = os.path.basename(att_path)
                                            if not att_path:
                                                continue
    
                                            # Chemin logique pour le CSV, basé sur le fichier source
                                            csv_path_for_attachment = build_attachment_csv_path(pdf_path, att_name)
                                            key_attach = (group_name, csv_path_for_attachment)
                                            if key_attach in existing_entries_by_base[base_name]:
                                                log(
                                                    f"[ATTACH-SKIP] Pièce jointe déjà dans le CSV de tracking : "
                                                    f"base={base_name}, group={group_name}, path={csv_path_for_attachment}"
                                                )
                                                continue
    
                                            log(
                                                f"[ATTACH-ADD] Ajout pièce jointe extraite : {att_name} "
                                                f"(base={base_name}, group={group_name})"
                                            )
                                            # Pour ingestion RAG, on utilise le chemin réel,
                                            # mais on associe un chemin logique pour le CSV / ref_id
                                            logical_paths[att_path] = csv_path_for_attachment
                                            # Stocker le fichier parent pour le bouton "Ouvrir"
                                            parent_paths[att_path] = pdf_path
                                            new_paths.append(att_path)
                                            # Pour la traçabilité CSV, on stocke le chemin logique demandé
                                            ingested_entries.append((base_name, group_name, csv_path_for_attachment))
                                            existing_entries_by_base[base_name][key_attach] = True
                                            ingestion_stats["csv_attachments"] += 1
                                # --- Fin extraction pièces jointes pour CSV ---
    
                                if not new_paths:
                                    log(
                                        f"[INGEST] Aucun nouveau fichier pour base={base_name}, "
                                        f"collection={group_name} (tout déjà ingéré ou introuvable)."
                                    )
                                    continue
    
                                ingestion_stats["csv_new_files"] += len(new_paths)
                                log(
                                    f"[INGEST] CSV {csv_name} → base={base_name} collection={group_name} "
                                    f"({len(new_paths)} nouveau(x) fichier(s))"
                                )
                                # Ajout dans le récapitulatif uniquement des nouveaux fichiers
                                for p in new_paths:
                                    key2 = (group_name, p)
                                    if key2 not in existing_entries_by_base[base_name]:
                                        ingested_entries.append((base_name, group_name, p))
                                        existing_entries_by_base[base_name][key2] = True
    
                                # Appel ingestion avec callback de progression
                                progress(0.05, f"[{base_name}/{group_name}] Démarrage de l'ingestion…")
                                try:
                                    # Récupérer les configs XML depuis session_state
                                    xml_configs_for_ingestion = st.session_state.get("xml_configs", {})

                                    report = ingest_documents(
                                        file_paths=new_paths,
                                        db_path=db_path,
                                        collection_name=group_name,
                                        chunk_size=chunk_size,
                                        use_easa_sections=use_easa_sections,
                                        rebuild=False,
                                        log=logger,
                                        logical_paths=logical_paths,
                                        parent_paths=parent_paths,
                                        progress_callback=progress,
                                        xml_configs=xml_configs_for_ingestion,
                                    )
                                    log(
                                        f"[INGEST] OK base={base_name}, collection={group_name}, "
                                        f"chunks={report.get('total_chunks', '?')}"
                                    )
    
                                    # Alimente le récapitulatif de session (ingestion via CSV)
                                    if isinstance(report, dict):
                                        for fi in report.get("files", []):
                                            session_ingested_files.append(
                                                {
                                                    "base": base_name,
                                                    "collection": group_name,
                                                    "file": fi.get("file", ""),
                                                    "chunks": fi.get("num_chunks", 0),
                                                    "language": fi.get("language", ""),
                                                    "sections": "oui" if fi.get("sections_detected") else "non",
                                                }
                                            )

                                    # SAUVEGARDE IMMÉDIATE dans le CSV de tracking
                                    # Cela permet de reprendre l'ingestion en cas d'interruption
                                    entries_to_save = []
                                    for p in new_paths:
                                        # Utiliser le chemin logique si disponible, sinon le chemin original
                                        csv_path_to_save = logical_paths.get(p, p)
                                        entries_to_save.append((group_name, csv_path_to_save))

                                    append_to_tracking_csv(base_name, entries_to_save, log_func=log)

                                except TypeError:
                                    # compat: ancienne signature sans mot-clés
                                    report = ingest_documents(new_paths, db_path, group_name)
                                    log(
                                        f"[INGEST] (compat) OK base={base_name}, collection={group_name}"
                                    )

                                    # SAUVEGARDE IMMÉDIATE (mode compat)
                                    entries_to_save = []
                                    for p in new_paths:
                                        csv_path_to_save = logical_paths.get(p, p)
                                        entries_to_save.append((group_name, csv_path_to_save))
                                    append_to_tracking_csv(base_name, entries_to_save, log_func=log)

                                # Nettoyage des répertoires temporaires des pièces jointes
                                for temp_dir in temp_dirs_to_cleanup:
                                    try:
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                        log(f"[CLEANUP] Répertoire temporaire pièces jointes supprimé : {temp_dir}")
                                    except Exception as e:
                                        log(f"[CLEANUP] Échec suppression répertoire temporaire {temp_dir} : {e}")

                                # Si arrêt demandé, sortir de la boucle des groupes
                                if ingestion_stopped:
                                    break

                            # Si arrêt demandé, sortir de la boucle des CSV
                            if ingestion_stopped:
                                break
                    finally:
                        try:
                            shutil.rmtree(csv_temp_dir, ignore_errors=True)
                            logger.info(f"[CLEANUP] Répertoire temporaire CSV supprimé : {csv_temp_dir}")
                        except Exception as e:
                            logger.error(f"[CLEANUP] Échec de suppression du répertoire temporaire CSV {csv_temp_dir} : {e}")

                # ------------------------------------------------------------------
                # Fin : génération / mise à jour CSV global + dashboard de synthèse
                # ------------------------------------------------------------------
                if not csv_files_to_process:
                    st.warning("Aucun CSV d'ingestion sélectionné ou uploadé.")
                elif ingestion_stopped:
                    progress(1.0, "⚠️ Ingestion interrompue")
                    st.warning("⚠️ Ingestion interrompue par l'utilisateur. Les fichiers déjà traités ont été sauvegardés.")
                else:
                    progress(1.0, "✅ Ingestion terminée.")
                    st.success("Ingestion terminée.")

                    # Réinitialiser les états XML après ingestion réussie
                    st.session_state["xml_preview_validated"] = False
                    st.session_state["xml_configs"] = {}
                    st.session_state["detected_xml_files"] = []

                    # Récapitulatif des fichiers ingérés pour cette exécution
                    if session_ingested_files:
                        st.markdown("### 📊 Récapitulatif des fichiers ingérés (cette exécution)")
                        session_ingested_files_sorted = sorted(
                            session_ingested_files,
                            key=lambda x: (x.get("base", ""), x.get("collection", ""), x.get("file", "")),
                        )
                        st.table(session_ingested_files_sorted)
                    else:
                        st.info("Aucun nouveau fichier n'a été ingéré pendant cette exécution.")

                    # ------------------------------------------------------------------
                    # Génération / mise à jour des CSV de tracking par base
                    # ------------------------------------------------------------------
                    if existing_entries_by_base:
                        os.makedirs(CSV_EXPORT_DIR, exist_ok=True)

                        for base_name, entries_dict in existing_entries_by_base.items():
                            # Convertir le dict en liste de tuples (group, path)
                            entries_list = [(g, p) for (g, p) in entries_dict.keys()]

                            if entries_list:
                                output_buf = io.StringIO()
                                writer = csv.writer(output_buf, delimiter=";")
                                writer.writerow(["group", "path"])
                                for g, p in sorted(entries_list):
                                    writer.writerow([g, p])

                                csv_data = output_buf.getvalue().encode("utf-8-sig")
                                csv_path = get_tracking_csv_path(base_name)

                                if csv_path:
                                    try:
                                        with open(csv_path, "wb") as f_out:
                                            f_out.write(csv_data)
                                        log(f"CSV de tracking mis à jour pour la base '{base_name}' : {csv_path}")
                                    except Exception as e:
                                        st.error(
                                            f"Impossible d'écrire le CSV de tracking pour '{base_name}' : {e}"
                                        )

                        st.markdown("#### 📄 CSV de tracking mis à jour par base")
                        st.info(f"✅ {len(existing_entries_by_base)} CSV de tracking mis à jour (un par base)")
                    else:
                        st.info("Aucun document ingéré, CSV de tracking non généré.")

                    # ------------------------------------------------------------------
                    # Résumé compact de l'ingestion (dashboard)
                    # ------------------------------------------------------------------
                    st.markdown("#### 📊 Résumé de l'ingestion")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Nouveaux fichiers", ingestion_stats["csv_new_files"])
                    with c2:
                        st.metric("Fichiers manquants", ingestion_stats["csv_missing_files"])
                    with c3:
                        st.metric("Déjà présents (skipped)", ingestion_stats["csv_skipped_existing"])
                    with c4:
                        st.metric("Pièces jointes", ingestion_stats["csv_attachments"])

            finally:
                # Supprimer tous les verrous créés, même en cas d'erreur
                for base_name in created_locks:
                    remove_ingestion_lock(base_root, base_name)
                    logger.info(f"Verrou d'ingestion supprimé pour {base_name}")

                # Réinitialiser les flags d'ingestion et les verrous sauvegardés
                st.session_state["ingestion_running"] = False
                st.session_state["stop_ingestion"] = False
                st.session_state["created_locks"] = []
                st.session_state["bulk_update_all"] = False  # Réinitialiser la sélection globale

                # Masquer le bouton stop
                stop_button_placeholder.empty()


# ========================
#   TAB CONFLUENCE (admin uniquement)
# ========================
with (tab_confluence if tab_confluence is not None else nullcontext()):
 if tab_confluence is None:
    pass  # Tab non visible pour cet utilisateur
 else:
    st.subheader("🌐 Ingestion depuis Confluence")

    if not CONFLUENCE_AVAILABLE:
        st.error(
            "❌ Le module Confluence n'est pas disponible. "
            "Installez les dépendances : `pip install beautifulsoup4 requests`"
        )
    else:
        st.markdown("""
        Cette page permet d'ingérer le contenu d'un **espace Confluence entier** dans le RAG.

        **Prérequis :**
        - URL de votre instance Confluence
        - Identifiants (nom d'utilisateur + mot de passe ou token API)
        - Clé de l'espace à ingérer (ex: `PROJ`, `DOC`, `KB`)
        """)

        # Initialisation session_state pour Confluence
        if "confluence_connected" not in st.session_state:
            st.session_state.confluence_connected = False
        if "confluence_spaces" not in st.session_state:
            st.session_state.confluence_spaces = []

        st.markdown("---")
        st.markdown("### 1️⃣ Connexion à Confluence")

        confluence_url_input = st.text_input(
            "URL Confluence",
            value="https://confluence.c3.dassault-aviation.pro:8445/spaces/EQUIP00000179/",
            help="Collez n'importe quelle URL Confluence depuis votre navigateur (page, espace, accueil...)"
        )

        # Auto-extraction de l'URL de base et du context path
        def parse_confluence_url(url: str) -> tuple:
            """Extrait base_url et context_path depuis une URL Confluence."""
            if not url:
                return "", ""
            url = url.strip()
            from urllib.parse import urlparse
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            path = parsed.path

            # Chercher les patterns Confluence connus dans le chemin
            patterns = ["/display/", "/pages/", "/spaces/", "/rest/api/", "/wiki/", "/confluence/"]
            context_path = ""
            for pattern in patterns:
                if pattern in path:
                    idx = path.find(pattern)
                    context_path = path[:idx] if idx > 0 else ""
                    break
            return base, context_path.rstrip("/")

        confluence_url, confluence_context_path_auto = parse_confluence_url(confluence_url_input)

        if confluence_url_input and confluence_url:
            st.caption(f"🔗 Base détectée: `{confluence_url}` | Contexte auto-détecté: `{confluence_context_path_auto or '(aucun)'}`")

        # Option pour forcer le contexte si l'auto-détection échoue
        col_context, col_context_help = st.columns([2, 2])
        with col_context:
            confluence_context_manual = st.text_input(
                "Chemin de contexte (optionnel)",
                value="",
                placeholder="/confluence ou /wiki",
                help="Si 'API non trouvée', essayez /confluence ou /wiki"
            )
        with col_context_help:
            st.caption("💡 Laissez vide pour utiliser l'auto-détection. Sinon essayez `/confluence` ou `/wiki`")

        # Utiliser le contexte manuel s'il est renseigné, sinon l'auto-détecté
        confluence_context_path = confluence_context_manual.strip() if confluence_context_manual.strip() else confluence_context_path_auto

        col_user, col_pwd = st.columns(2)
        with col_user:
            confluence_user = st.text_input(
                "Nom d'utilisateur / Email",
                placeholder="votre.email@entreprise.com"
            )
        with col_pwd:
            confluence_password = st.text_input(
                "Mot de passe / Token API",
                type="password",
                help="Pour Confluence Cloud, utilisez un token API (créé depuis les paramètres Atlassian)"
            )

        col_ssl, col_empty = st.columns([1, 3])
        with col_ssl:
            confluence_skip_ssl = st.checkbox(
                "Ignorer SSL",
                value=False,
                help="Cochez si votre serveur utilise un certificat auto-signé"
            )

        col_test, col_status = st.columns([1, 3])
        with col_test:
            if st.button("🔗 Tester la connexion", disabled=not (confluence_url and confluence_user and confluence_password)):
                with st.spinner("Test de connexion..."):
                    verify_ssl = not confluence_skip_ssl
                    result = test_confluence_connection(confluence_url, confluence_user, confluence_password, verify_ssl=verify_ssl, context_path=confluence_context_path)
                    if result["success"]:
                        st.session_state.confluence_connected = True
                        st.session_state.confluence_url = confluence_url
                        st.session_state.confluence_user = confluence_user
                        st.session_state.confluence_password = confluence_password
                        st.session_state.confluence_verify_ssl = verify_ssl
                        st.session_state.confluence_context_path = confluence_context_path
                        # Charger la liste des espaces
                        st.session_state.confluence_spaces = list_spaces(confluence_url, confluence_user, confluence_password, verify_ssl=verify_ssl, context_path=confluence_context_path)
                        st.success(result["message"])
                    else:
                        st.session_state.confluence_connected = False
                        st.error(result["message"])

        with col_status:
            if st.session_state.confluence_connected:
                st.success("✅ Connecté à Confluence")

        # Section de sélection de l'espace (visible seulement si connecté)
        if st.session_state.confluence_connected:
            st.markdown("---")
            st.markdown("### 2️⃣ Sélection de l'espace")

            # Liste déroulante des espaces ou saisie manuelle
            space_input_mode = st.radio(
                "Mode de sélection",
                ["Liste des espaces", "Saisie manuelle"],
                horizontal=True
            )

            if space_input_mode == "Liste des espaces":
                if st.session_state.confluence_spaces:
                    space_options = {f"{s['name']} ({s['key']})": s['key'] for s in st.session_state.confluence_spaces}
                    selected_space_label = st.selectbox(
                        "Espace Confluence",
                        options=list(space_options.keys())
                    )
                    confluence_space_key = space_options.get(selected_space_label, "")
                else:
                    st.warning("Aucun espace trouvé. Utilisez la saisie manuelle.")
                    confluence_space_key = ""
            else:
                confluence_space_key = st.text_input(
                    "Clé de l'espace",
                    placeholder="PROJ",
                    help="La clé de l'espace (visible dans l'URL des pages)"
                )

            # Afficher les infos de l'espace
            if confluence_space_key:
                space_info = get_space_info(
                    st.session_state.confluence_url,
                    confluence_space_key,
                    st.session_state.confluence_user,
                    st.session_state.confluence_password,
                    verify_ssl=st.session_state.get("confluence_verify_ssl", True),
                    context_path=st.session_state.get("confluence_context_path", "")
                )
                if space_info:
                    st.info(f"📁 **{space_info['name']}** - {space_info.get('description', 'Pas de description')[:100]}")
                else:
                    st.warning(f"⚠️ Espace '{confluence_space_key}' non trouvé ou inaccessible")

            st.markdown("---")
            st.markdown("### 3️⃣ Scanner l'espace")

            # Initialiser session_state pour le scan
            if "confluence_scanned_pages" not in st.session_state:
                st.session_state.confluence_scanned_pages = None
            if "confluence_sections" not in st.session_state:
                st.session_state.confluence_sections = {}
            if "confluence_selected_sections" not in st.session_state:
                st.session_state.confluence_selected_sections = {}

            col_scan, col_scan_status = st.columns([1, 3])
            with col_scan:
                if st.button("🔍 Scanner l'espace", disabled=not confluence_space_key):
                    with st.spinner("Scan en cours..."):
                        pages = extract_text_from_confluence_space(
                            st.session_state.confluence_url,
                            confluence_space_key,
                            st.session_state.confluence_user,
                            st.session_state.confluence_password,
                            verify_ssl=st.session_state.get("confluence_verify_ssl", True),
                            context_path=st.session_state.get("confluence_context_path", "")
                        )
                        if pages:
                            st.session_state.confluence_scanned_pages = pages
                            st.session_state.confluence_sections = group_pages_by_section(pages)
                            # Initialiser toutes les sections comme sélectionnées
                            st.session_state.confluence_selected_sections = {
                                section: True for section in st.session_state.confluence_sections.keys()
                            }
                            st.success(f"✅ {len(pages)} pages trouvées dans {len(st.session_state.confluence_sections)} sections")
                        else:
                            st.error("Aucune page trouvée")
                            st.session_state.confluence_scanned_pages = None

            # Afficher les résultats du scan
            if st.session_state.confluence_scanned_pages and st.session_state.confluence_sections:
                st.markdown("---")

                # Détecter si structure plate (chaque page = sa propre section)
                is_flat_structure = len(st.session_state.confluence_sections) == len(st.session_state.confluence_scanned_pages)

                if is_flat_structure:
                    # MODE PLAT: Sélection individuelle des pages
                    st.markdown("### 4️⃣ Sélection des pages")
                    st.markdown(f"**{len(st.session_state.confluence_scanned_pages)} pages** trouvées (structure plate, pas de hiérarchie)")

                    # Initialiser la sélection des pages si nécessaire
                    if "confluence_selected_pages" not in st.session_state:
                        st.session_state.confluence_selected_pages = {}

                    # S'assurer que toutes les pages ont une entrée
                    for page in st.session_state.confluence_scanned_pages:
                        page_id = page.get("id", page.get("title"))
                        if page_id not in st.session_state.confluence_selected_pages:
                            st.session_state.confluence_selected_pages[page_id] = True

                    # Boutons tout sélectionner / tout désélectionner
                    col_all, col_none, col_spacer = st.columns([1, 1, 4])
                    with col_all:
                        if st.button("✅ Tout sélectionner", key="select_all_pages"):
                            for page in st.session_state.confluence_scanned_pages:
                                page_id = page.get("id", page.get("title"))
                                st.session_state.confluence_selected_pages[page_id] = True
                            st.rerun()
                    with col_none:
                        if st.button("❌ Tout désélectionner", key="deselect_all_pages"):
                            for page in st.session_state.confluence_scanned_pages:
                                page_id = page.get("id", page.get("title"))
                                st.session_state.confluence_selected_pages[page_id] = False
                            st.rerun()

                    # Afficher les pages avec checkboxes
                    st.markdown("---")
                    for page in st.session_state.confluence_scanned_pages:
                        page_id = page.get("id", page.get("title"))
                        page_title = page.get("title", "Sans titre")
                        page_url = page.get("url", "")
                        text_preview = page.get("text", "")[:150].replace("\n", " ")

                        col_check, col_info = st.columns([3, 1])
                        with col_check:
                            # Checkbox avec titre
                            label = f"📄 **{page_title}**"
                            if page_url:
                                label = f"📄 [{page_title}]({page_url})"
                            page_selected = st.checkbox(
                                label,
                                value=st.session_state.confluence_selected_pages.get(page_id, True),
                                key=f"page_{page_id}"
                            )
                            st.session_state.confluence_selected_pages[page_id] = page_selected
                            if text_preview:
                                st.caption(f"  {text_preview}...")

                    # Compter les pages sélectionnées
                    selected_pages_count = sum(1 for v in st.session_state.confluence_selected_pages.values() if v)
                    st.info(f"📊 **{selected_pages_count} pages** sélectionnées sur {len(st.session_state.confluence_scanned_pages)}")

                else:
                    # MODE HIÉRARCHIQUE: Sélection par sections (code existant)
                    st.markdown("### 4️⃣ Sélection des sections")
                    st.markdown(f"**{len(st.session_state.confluence_scanned_pages)} pages** réparties en **{len(st.session_state.confluence_sections)} sections**")

                    # Boutons tout sélectionner / tout désélectionner
                    col_all, col_none, col_spacer = st.columns([1, 1, 4])
                    with col_all:
                        if st.button("✅ Tout sélectionner"):
                            for section in st.session_state.confluence_sections.keys():
                                st.session_state.confluence_selected_sections[section] = True
                            st.rerun()
                    with col_none:
                        if st.button("❌ Tout désélectionner"):
                            for section in st.session_state.confluence_sections.keys():
                                st.session_state.confluence_selected_sections[section] = False
                            st.rerun()

                    # Afficher les sections avec leurs pages
                    for section_name, section_pages in st.session_state.confluence_sections.items():
                        section_selected = st.checkbox(
                            f"📁 **{section_name}** ({len(section_pages)} pages)",
                            value=st.session_state.confluence_selected_sections.get(section_name, True),
                            key=f"section_{section_name}"
                        )
                        st.session_state.confluence_selected_sections[section_name] = section_selected

                        with st.expander(f"Voir les {len(section_pages)} pages de '{section_name}'", expanded=False):
                            for page in section_pages:
                                page_url = page.get("url", "")
                                page_title = page.get("title", "Sans titre")
                                text_preview = page.get("text", "")[:200].replace("\n", " ")
                                if page_url:
                                    st.markdown(f"- [{page_title}]({page_url})")
                                else:
                                    st.markdown(f"- {page_title}")
                                if text_preview:
                                    st.caption(f"  {text_preview}...")

                    # Compter les pages sélectionnées
                    selected_pages_count = sum(
                        len(pages) for section, pages in st.session_state.confluence_sections.items()
                        if st.session_state.confluence_selected_sections.get(section, False)
                    )
                    selected_sections_count = sum(1 for v in st.session_state.confluence_selected_sections.values() if v)
                    st.info(f"📊 **{selected_sections_count} sections** et **{selected_pages_count} pages** sélectionnées")

                st.markdown("---")
                st.markdown("### 5️⃣ Configuration de l'ingestion")

                # Base FAISS dédiée pour Confluence
                CONFLUENCE_BASE_NAME = "CONFLUENCE"
                confluence_base_path = os.path.join(base_root, CONFLUENCE_BASE_NAME)
                if not os.path.exists(confluence_base_path):
                    os.makedirs(confluence_base_path, exist_ok=True)

                st.info(f"📦 **Base FAISS dédiée** : `{CONFLUENCE_BASE_NAME}`")

                # Mode de collection (différent selon structure)
                if is_flat_structure:
                    # Structure plate: une seule collection
                    collection_mode = "Une seule collection pour tout"
                    confluence_collection = st.text_input(
                        "Nom de la collection",
                        value=confluence_space_key.lower() if confluence_space_key else "",
                        help="Toutes les pages sélectionnées iront dans cette collection"
                    )
                else:
                    # Structure hiérarchique: choix possible
                    collection_mode = st.radio(
                        "Mode de création des collections",
                        ["Une collection par section", "Une seule collection pour tout"],
                        help="Par section: chaque section devient une collection séparée. Unique: tout dans une seule collection."
                    )

                    if collection_mode == "Une seule collection pour tout":
                        confluence_collection = st.text_input(
                            "Nom de la collection unique",
                            value=confluence_space_key.lower() if confluence_space_key else "",
                            help="Toutes les pages iront dans cette collection"
                        )
                    else:
                        st.caption("Les collections seront nommées d'après les sections (nettoyées en minuscules)")
                        preview_names = []
                        for section in st.session_state.confluence_sections.keys():
                            if st.session_state.confluence_selected_sections.get(section, False):
                                safe_name = sanitize_collection_name(section)
                                preview_names.append(safe_name)
                        if preview_names:
                            st.caption(f"Collections qui seront créées: `{'`, `'.join(preview_names[:5])}`{'...' if len(preview_names) > 5 else ''}")

                # Option de reconstruction
                confluence_rebuild = st.checkbox(
                    "🔄 Reconstruire les collections (supprimer l'existant)",
                    value=True,
                    help="Recommandé pour une mise à jour complète"
                )

                st.markdown("---")
                st.markdown("### 6️⃣ Lancer l'ingestion")

                can_ingest = selected_pages_count > 0 and (
                    collection_mode == "Une collection par section" or confluence_collection
                )

                button_label = "🚀 Ingérer les pages sélectionnées" if is_flat_structure else "🚀 Ingérer les sections sélectionnées"
                if st.button(button_label, disabled=not can_ingest, type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    log_container = st.empty()
                    logs = []

                    def log(msg):
                        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                        log_container.code("\n".join(logs[-20:]), language="")

                    def update_progress(value, text):
                        progress_bar.progress(min(value, 1.0))
                        status_text.text(text)

                    try:
                        db_path = os.path.join(base_root, CONFLUENCE_BASE_NAME)
                        total_chunks = 0
                        total_pages = 0

                        if is_flat_structure:
                            # MODE PLAT: Collecter les pages sélectionnées individuellement
                            selected_pages_list = [
                                page for page in st.session_state.confluence_scanned_pages
                                if st.session_state.confluence_selected_pages.get(
                                    page.get("id", page.get("title")), False
                                )
                            ]

                            log(f"📁 Collection: '{confluence_collection}'")
                            log(f"📄 {len(selected_pages_list)} pages sélectionnées")
                            temp_dir = tempfile.mkdtemp(prefix="confluence_")
                            file_paths = []
                            logical_paths = {}

                            for page in selected_pages_list:
                                if not page.get("text", "").strip():
                                    continue
                                safe_title = sanitize_collection_name(page["title"])
                                temp_file = os.path.join(temp_dir, f"{page['id']}_{safe_title}.txt")
                                with open(temp_file, "w", encoding="utf-8") as f:
                                    f.write(page["text"])
                                file_paths.append(temp_file)
                                logical_paths[temp_file] = page.get("url", page["path"])

                            update_progress(0.3, "Ingestion dans FAISS...")

                            if file_paths:
                                report = ingest_documents(
                                    file_paths=file_paths,
                                    db_path=db_path,
                                    collection_name=confluence_collection,
                                    chunk_size=1000,
                                    use_easa_sections=False,
                                    rebuild=confluence_rebuild,
                                    log=logger,
                                    logical_paths=logical_paths,
                                    progress_callback=lambda p, t: update_progress(0.3 + p * 0.65, t),
                                )
                                total_chunks = report.get("total_chunks", 0)
                                total_pages = len(file_paths)

                            shutil.rmtree(temp_dir, ignore_errors=True)

                        else:
                            # MODE HIÉRARCHIQUE: Collecter les sections sélectionnées
                            selected_sections = [
                                (section, pages) for section, pages in st.session_state.confluence_sections.items()
                                if st.session_state.confluence_selected_sections.get(section, False)
                            ]

                            if collection_mode == "Une collection par section":
                                # Ingestion section par section
                                for idx, (section_name, section_pages) in enumerate(selected_sections):
                                    section_progress_base = idx / len(selected_sections)
                                    section_progress_step = 1 / len(selected_sections)

                                    safe_collection = sanitize_collection_name(section_name)
                                    log(f"📁 Section '{section_name}' -> collection '{safe_collection}'")
                                    update_progress(section_progress_base, f"Ingestion de '{section_name}'...")

                                    temp_dir = tempfile.mkdtemp(prefix="confluence_")
                                    file_paths = []
                                    logical_paths = {}

                                    for page in section_pages:
                                        if not page.get("text", "").strip():
                                            continue
                                        safe_title = sanitize_collection_name(page["title"])
                                        temp_file = os.path.join(temp_dir, f"{page['id']}_{safe_title}.txt")
                                        with open(temp_file, "w", encoding="utf-8") as f:
                                            f.write(page["text"])
                                        file_paths.append(temp_file)
                                        logical_paths[temp_file] = page.get("url", page["path"])

                                    if file_paths:
                                        report = ingest_documents(
                                            file_paths=file_paths,
                                            db_path=db_path,
                                            collection_name=safe_collection,
                                            chunk_size=1000,
                                            use_easa_sections=False,
                                            rebuild=confluence_rebuild,
                                            log=logger,
                                            logical_paths=logical_paths,
                                            progress_callback=lambda p, t: update_progress(
                                                section_progress_base + p * section_progress_step * 0.9, t
                                            ),
                                        )
                                        total_chunks += report.get("total_chunks", 0)
                                        total_pages += len(file_paths)
                                        log(f"  ✅ {report.get('total_chunks', 0)} chunks créés")

                                    shutil.rmtree(temp_dir, ignore_errors=True)

                            else:
                                # Ingestion dans une seule collection
                                log(f"📁 Collection unique: '{confluence_collection}'")
                                temp_dir = tempfile.mkdtemp(prefix="confluence_")
                                file_paths = []
                                logical_paths = {}

                                for section_name, section_pages in selected_sections:
                                    for page in section_pages:
                                        if not page.get("text", "").strip():
                                            continue
                                        safe_title = sanitize_collection_name(page["title"])
                                        temp_file = os.path.join(temp_dir, f"{page['id']}_{safe_title}.txt")
                                        with open(temp_file, "w", encoding="utf-8") as f:
                                            f.write(page["text"])
                                        file_paths.append(temp_file)
                                        logical_paths[temp_file] = page.get("url", page["path"])

                                log(f"📄 {len(file_paths)} fichiers préparés")
                                update_progress(0.3, "Ingestion dans FAISS...")

                                if file_paths:
                                    report = ingest_documents(
                                        file_paths=file_paths,
                                        db_path=db_path,
                                        collection_name=confluence_collection,
                                        chunk_size=1000,
                                        use_easa_sections=False,
                                        rebuild=confluence_rebuild,
                                        log=logger,
                                        logical_paths=logical_paths,
                                        progress_callback=lambda p, t: update_progress(0.3 + p * 0.65, t),
                                    )
                                    total_chunks = report.get("total_chunks", 0)
                                    total_pages = len(file_paths)

                                shutil.rmtree(temp_dir, ignore_errors=True)

                        update_progress(1.0, "Terminé!")
                        log(f"✅ Ingestion terminée: {total_chunks} chunks créés")

                        if is_flat_structure:
                            st.success(
                                f"✅ **Ingestion réussie !**\n\n"
                                f"- Pages traitées: {total_pages}\n"
                                f"- Chunks créés: {total_chunks}\n"
                                f"- Collection: {confluence_collection}\n"
                                f"- Base: {CONFLUENCE_BASE_NAME}"
                            )
                        else:
                            collections_created = len(selected_sections) if collection_mode == "Une collection par section" else 1
                            st.success(
                                f"✅ **Ingestion réussie !**\n\n"
                                f"- Sections traitées: {len(selected_sections)}\n"
                                f"- Pages traitées: {total_pages}\n"
                                f"- Chunks créés: {total_chunks}\n"
                                f"- Collections créées: {collections_created}\n"
                                f"- Base: {CONFLUENCE_BASE_NAME}"
                            )

                        # Réinitialiser le scan
                        st.session_state.confluence_scanned_pages = None
                        st.session_state.confluence_sections = {}
                        if "confluence_selected_pages" in st.session_state:
                            st.session_state.confluence_selected_pages = {}

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'ingestion: {str(e)}")
                        log(f"ERREUR: {str(e)}")
                        import traceback
                        log(traceback.format_exc())


# ========================
#   TAB PURGE DES BASES (admin uniquement)
# ========================
with (tab_purge if tab_purge is not None else nullcontext()):
 if tab_purge is None:
    pass
 else:
    st.subheader("🗑️ Purge des bases FAISS")

    st.warning(
        "⚠️ **ATTENTION** : La purge d'une base supprimera **TOUT** le contenu de toutes ses collections "
        "et mettra à jour le CSV de tracking correspondant. Cette action est irréversible !"
    )

    if not bases:
        st.info("ℹ️ Aucune base FAISS trouvée.")
    else:
        st.markdown("### Sélection de la base à purger")

        base_to_purge = st.selectbox(
            "Choisissez la base à purger",
            options=bases,
            key="base_to_purge"
        )

        if base_to_purge:
            # Afficher les statistiques de la base
            st.markdown(f"### 📊 Statistiques de la base `{base_to_purge}`")

            collections = list_collections_for_base(base_root, base_to_purge)
            col_counts = get_collection_doc_counts(base_root, base_to_purge)

            if collections:
                total_docs = sum(col_counts.values())

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre de collections", len(collections))
                with col2:
                    st.metric("Total de chunks indexés", total_docs)

                st.markdown("#### Détail par collection")
                collections_details = []
                for coll_name in collections:
                    collections_details.append({
                        "Collection": coll_name,
                        "Nombre de chunks": col_counts.get(coll_name, 0)
                    })
                st.table(collections_details)

                # Vérifier si un CSV de tracking existe
                tracking_csv_path = get_tracking_csv_path(base_to_purge)
                csv_exists = tracking_csv_path and os.path.exists(tracking_csv_path)

                if csv_exists:
                    st.info(f"📄 Un CSV de tracking existe pour cette base : `{os.path.basename(tracking_csv_path)}`")
                else:
                    st.info("ℹ️ Aucun CSV de tracking trouvé pour cette base.")

                # Section de confirmation
                st.markdown("---")
                st.markdown("### ⚠️ Confirmation de purge")

                confirm_text = st.text_input(
                    f"Pour confirmer la purge, tapez le nom de la base : **{base_to_purge}**",
                    key="confirm_purge_text"
                )

                if st.button("🗑️ PURGER LA BASE", type="primary", disabled=(confirm_text != base_to_purge), help="Supprime définitivement tout le contenu de la base sélectionnée. Cette action est irréversible !"):
                    if confirm_text == base_to_purge:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        try:
                            db_path = os.path.join(base_root, base_to_purge)
                            store = build_faiss_store(db_path)

                            # Purger chaque collection (FAISS: suppression complète de la collection)
                            status_text.info(f"Purge de {len(collections)} collection(s) en cours...")
                            purged_collections = []

                            for idx, coll_name in enumerate(collections):
                                try:
                                    # Compter les éléments avant suppression
                                    try:
                                        collection = store.get_collection(coll_name)
                                        doc_count = collection.count()
                                    except Exception:
                                        doc_count = 0

                                    # FAISS: supprimer toute la collection
                                    store.delete_collection(coll_name)

                                    logger.info(f"[PURGE] Collection '{coll_name}' purgée : {doc_count} chunks supprimés")
                                    purged_collections.append((coll_name, doc_count))

                                    progress_bar.progress((idx + 1) / len(collections))

                                except Exception as e:
                                    st.error(f"❌ Erreur lors de la purge de la collection '{coll_name}' : {e}")
                                    logger.error(f"[PURGE] Erreur collection {coll_name}: {e}")

                            # Supprimer le CSV de tracking
                            if csv_exists:
                                try:
                                    os.remove(tracking_csv_path)
                                    st.success(f"✅ CSV de tracking supprimé : {os.path.basename(tracking_csv_path)}")
                                    logger.info(f"[PURGE] CSV de tracking supprimé : {tracking_csv_path}")
                                except Exception as e:
                                    st.warning(f"⚠️ Impossible de supprimer le CSV de tracking : {e}")
                                    logger.error(f"[PURGE] Erreur suppression CSV : {e}")

                            progress_bar.progress(1.0)
                            status_text.empty()

                            # Afficher le résumé
                            st.success(f"✅ Base `{base_to_purge}` purgée avec succès !")

                            total_purged = sum(count for _, count in purged_collections)
                            st.markdown(f"**Total de chunks supprimés** : {total_purged}")

                            st.markdown("#### Détail de la purge")
                            purge_details = []
                            for coll_name, count in purged_collections:
                                purge_details.append({
                                    "Collection": coll_name,
                                    "Chunks supprimés": count
                                })
                            st.table(purge_details)

                            st.info("🔄 Rechargez la page pour voir les changements dans la sidebar.")

                        except Exception as e:
                            st.error(f"❌ Erreur lors de la purge de la base : {e}")
                            logger.error(f"[PURGE] Erreur globale : {e}")
                    else:
                        st.warning("⚠️ Le nom de la base ne correspond pas. Purge annulée.")
            else:
                st.info("ℹ️ Cette base ne contient aucune collection.")


# ========================
#   Helpers affichage RAG
# ========================

def _validate_file_path(file_path: str) -> tuple:
    """
    Valide un chemin de fichier pour prévenir les attaques par traversée de répertoire.

    Returns:
        Tuple (is_safe, error_message)
    """
    if not file_path:
        return False, "Chemin vide"

    # Normaliser le chemin pour éviter les attaques de traversée
    normalized_path = os.path.normpath(os.path.abspath(file_path))

    # Vérifier les patterns de traversée de répertoire
    if ".." in file_path:
        return False, "Chemin interdit: traversée de répertoire détectée"

    # Vérifier que le fichier existe
    if not os.path.exists(normalized_path):
        return False, f"Fichier non trouvé: {normalized_path}"

    # Vérifier que c'est un fichier (pas un répertoire)
    if not os.path.isfile(normalized_path):
        return False, f"N'est pas un fichier: {normalized_path}"

    return True, normalized_path


def open_file_callback(file_path: str):
    """Callback pour ouvrir un fichier ou une URL sans recharger la page Streamlit"""
    try:
        import subprocess
        import platform
        import webbrowser

        # Vérifier si c'est une URL (Confluence, etc.)
        if file_path.startswith("http://") or file_path.startswith("https://"):
            # Ouvrir l'URL dans le navigateur par défaut
            webbrowser.open(file_path)
            return

        # Validation du chemin de fichier local
        is_safe, result = _validate_file_path(file_path)
        if not is_safe:
            st.session_state['file_open_error'] = f"Chemin non autorisé: {result}"
            return

        validated_path = result

        # Ouvrir le fichier avec le programme par défaut du système
        if platform.system() == "Windows":
            os.startfile(validated_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", validated_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", validated_path])
    except Exception as e:
        # Stocker l'erreur dans session_state pour l'afficher
        st.session_state['file_open_error'] = f"Impossible d'ouvrir le fichier : {e}"


def is_llm_error_response(answer: str) -> bool:
    """Vérifie si la réponse est un message d'erreur du LLM."""
    if not answer:
        return False
    error_markers = [
        "ERREUR DE COMMUNICATION AVEC LE LLM",
        "Le serveur n'a pas pu répondre",
        "Veuillez reposer votre question",
    ]
    return any(marker in answer for marker in error_markers)


def display_answer_with_error_check(answer: str) -> None:
    """Affiche la réponse avec détection d'erreur et alerte."""
    if not answer or not answer.strip():
        st.warning("⚠️ Aucune réponse générée par le LLM. Vérifiez la configuration des modèles ou les logs.")
    elif is_llm_error_response(answer):
        st.error(answer)
        st.info("💡 **Conseil**: Attendez quelques secondes puis cliquez sur 'Rechercher' à nouveau.")
    else:
        st.write(answer)


def render_sources_list(
    sources: List[Dict],
    global_title: str = "Sources utilisées (triées par pertinence)",
    show_collection: bool = False,
) -> None:
    """Affichage compact et coloré des sources RAG avec contexte par chunk."""
    if not sources:
        return

    # Limiter à 10 meilleurs candidats
    sources = sources[:10]

    st.markdown(f"### {global_title}")
    for i, src in enumerate(sources, start=1):
        score = float(src.get("score", 0.0) or 0.0)
        distance = float(src.get("distance", 0.0) or 0.0)

        # Code couleur avec seuils ajustés (plus permissifs)
        # Vert: score >= 0.5, Orange: score >= 0.3, Rouge: score < 0.3
        if score >= 0.5:
            badge = "🟢"
        elif score >= 0.3:
            badge = "🟠"
        else:
            badge = "🔴"

        collection_label = ""
        if show_collection:
            collection_label = f"[{src.get('collection', '?')}] "

        # Nom du document (source_file)
        doc_name = src.get('source_file', 'unknown')

        header = (
            f"{badge} {i}. {collection_label}"
            f"{doc_name} "
            f"(score {score:.3f})"
        )

        with st.expander(header):
            # Afficher le chemin complet du fichier
            file_path = src.get("path", "")
            is_attachment = src.get("is_attachment", False)
            parent_file = src.get("parent_file", "")

            if file_path:
                # Détecter si c'est une URL (Confluence, etc.)
                is_url = file_path.startswith("http://") or file_path.startswith("https://")

                # Afficher la source avec indication si c'est une pièce jointe ou URL
                if is_url:
                    st.markdown(f"**Source** : [{file_path}]({file_path}) *(Confluence)*")
                elif is_attachment and parent_file:
                    st.markdown(f"**Source** : `{file_path}` *(pièce jointe)*")
                    st.markdown(f"**Fichier parent** : `{parent_file}`")
                else:
                    st.markdown(f"**Source** : `{file_path}`")

                # Bouton pour ouvrir le fichier ou l'URL (utilise un callback pour éviter le rerun)
                # Pour les pièces jointes, ouvrir le fichier parent
                file_to_open = parent_file if (is_attachment and parent_file) else file_path
                file_hash = hashlib.md5(file_to_open.encode()).hexdigest()[:8]

                # Label différent pour URL vs fichier local
                button_label = "🔗 Ouvrir dans le navigateur" if is_url else "📂 Ouvrir"
                st.button(
                    button_label,
                    key=f"open_{i}_{file_hash}",
                    on_click=open_file_callback,
                    args=(file_to_open,)
                )

            section_id = src.get("section_id") or ""
            section_title = src.get("section_title") or ""
            if section_id or section_title:
                st.markdown("**Section EASA détectée :**")
                st.write(f"{section_id} {section_title}".strip())

            st.markdown("**Passage utilisé (chunk) :**")
            st.write(src.get("text", ""))


# ========================
#   TAB RAG
# ========================
with tab_rag:
    st.subheader("❓ Poser une question au RAG (DALLEM)")

    # Sélection de la base et collection
    sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 1])

    with sel_col1:
        st.markdown("**Base sélectionnée :**")
        if bases:
            base_for_query = st.selectbox(
                "Base pour la recherche",
                options=bases,
                label_visibility="collapsed"
            )
        else:
            base_for_query = st.selectbox(
                "Base pour la recherche",
                options=["(aucune)"],
                label_visibility="collapsed"
            )

    with sel_col2:
        st.markdown("**Collection(s) :**")
        # Obtenir les collections pour la base sélectionnée
        if bases and base_for_query != "(aucune)":
            collections_for_query = list_collections_for_base(base_root, base_for_query)
            col_counts = get_collection_doc_counts(base_root, base_for_query)
        else:
            collections_for_query = []
            col_counts = {}

        # Mode de sélection: simple ou multi
        multi_mode = st.checkbox(
            "🔀 Multi-collection",
            value=False,
            help="Interroger plusieurs collections simultanément (recherche inter-bases)"
        )

        if multi_mode and MULTI_COLLECTION_AVAILABLE:
            # Mode multi-sélection
            collections_selected = st.multiselect(
                "Collections pour la recherche",
                options=collections_for_query if collections_for_query else [],
                default=collections_for_query[:2] if len(collections_for_query) >= 2 else collections_for_query,
                label_visibility="collapsed",
                help="Sélectionnez 2+ collections pour la recherche inter-bases"
            )
            collection_for_query = collections_selected if collections_selected else []
        elif multi_mode and not MULTI_COLLECTION_AVAILABLE:
            st.warning("⚠️ Module multi-collection non disponible")
            collection_for_query = st.selectbox(
                "Collection pour la recherche",
                options=(["ALL"] + collections_for_query) if collections_for_query else ["ALL"],
                label_visibility="collapsed"
            )
        else:
            # Mode simple (selectbox classique)
            collection_for_query = st.selectbox(
                "Collection pour la recherche",
                options=(["ALL"] + collections_for_query) if collections_for_query else ["ALL"],
                label_visibility="collapsed"
            )

        # Debug: afficher si aucune collection trouvée
        if bases and base_for_query != "(aucune)" and not collections_for_query:
            db_path_debug = os.path.join(base_root, base_for_query)
            if os.path.exists(db_path_debug):
                subdirs = [d for d in os.listdir(db_path_debug) if os.path.isdir(os.path.join(db_path_debug, d))]
                if subdirs:
                    # Montrer le contenu du premier dossier pour debug
                    first_subdir = subdirs[0]
                    first_path = os.path.join(db_path_debug, first_subdir)
                    files_in_first = os.listdir(first_path) if os.path.exists(first_path) else []
                    with st.expander(f"⚠️ {len(subdirs)} dossier(s) trouvé(s) mais aucune collection valide", expanded=True):
                        st.write(f"**Dossiers:** {', '.join(subdirs[:5])}{'...' if len(subdirs) > 5 else ''}")
                        st.write(f"**Fichiers dans '{first_subdir}':** {', '.join(files_in_first[:10]) if files_in_first else '(vide)'}")
                        st.caption("Une collection valide doit contenir `index.faiss` ou `metadata.json`")
                else:
                    st.info(f"ℹ️ La base '{base_for_query}' est vide (aucun sous-dossier)")
            else:
                st.error(f"❌ Chemin inaccessible: {db_path_debug}")

    with sel_col3:
        st.markdown("&nbsp;")  # Espacement pour aligner
        if st.button("🔄 Actualiser", use_container_width=True, help="Actualiser la liste des bases et collections depuis le réseau"):
            # Vider uniquement les caches des listes (PAS les stores FAISS)
            list_bases.clear()
            list_collections_for_base.clear()
            get_collection_doc_counts.clear()
            st.rerun()

    # ========================
    #   SECTION CACHE LOCAL
    # ========================
    cache_mgr = get_cache_manager()
    cache_status = cache_mgr.get_cache_status()

    with st.expander("💾 Cache Local (performances réseau)", expanded=False):
        # Afficher le statut global du cache
        if cache_status['collections_count'] > 0:
            st.success(f"📊 {cache_status['collections_count']} collection(s) en cache ({cache_status['total_size_mb']:.1f} MB)")

        if bases and base_for_query != "(aucune)":
            base_path_cache = os.path.join(base_root, base_for_query)

            # Option pour copier toutes les bases ou juste la collection en cours
            cache_scope = st.radio(
                "Portée du cache",
                options=["Collection sélectionnée", "Toutes les bases"],
                key="cache_scope_rag",
                horizontal=True
            )

            if cache_scope == "Toutes les bases":
                st.warning("⚠️ Copier toutes les bases peut prendre plusieurs minutes.")

            # Vérifier le statut du cache pour la collection sélectionnée
            # Note: collection_for_query peut être une string ou une liste (mode multi-collection)
            is_single_collection = isinstance(collection_for_query, str) and collection_for_query != "ALL"

            if is_single_collection and collections_for_query:
                collection_path_cache = os.path.join(base_path_cache, collection_for_query)
                is_cached = cache_mgr.is_cached(collection_path_cache)

                if is_cached:
                    is_valid = cache_mgr.is_cache_valid(collection_path_cache)
                    if is_valid:
                        st.success(f"✅ **{collection_for_query}** : en cache local (performances optimales)")
                    else:
                        st.warning(f"⚠️ **{collection_for_query}** : cache obsolète - cliquez pour actualiser")
                else:
                    st.info(f"ℹ️ **{collection_for_query}** : pas en cache local")
            elif isinstance(collection_for_query, list) and collection_for_query:
                # Mode multi-collection: afficher le statut pour chaque collection
                cached_count = 0
                for coll in collection_for_query:
                    cp = os.path.join(base_path_cache, coll)
                    if cache_mgr.is_cached(cp) and cache_mgr.is_cache_valid(cp):
                        cached_count += 1
                if cached_count == len(collection_for_query):
                    st.success(f"✅ **{len(collection_for_query)} collections** : toutes en cache local")
                elif cached_count > 0:
                    st.warning(f"⚠️ **{cached_count}/{len(collection_for_query)} collections** en cache local")
                else:
                    st.info(f"ℹ️ **{len(collection_for_query)} collections** : aucune en cache local")

            # Boutons d'action
            col_cache1, col_cache2, col_cache3 = st.columns(3)

            with col_cache1:
                if st.button("📥 Copier en local", type="primary", use_container_width=True, key="btn_cache_copy"):
                    if cache_scope == "Toutes les bases":
                        with st.spinner("Copie de toutes les bases en cours..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            total_copied = 0
                            total_errors = 0

                            all_collections = []
                            for b in bases:
                                bp = os.path.join(base_root, b)
                                try:
                                    s = build_faiss_store(bp)
                                    for c in s.list_collections():
                                        all_collections.append((bp, c))
                                except Exception:
                                    pass

                            total = len(all_collections)
                            for idx, (bp, c) in enumerate(all_collections):
                                cp = os.path.join(bp, c)
                                status_text.text(f"Copie {c}... ({idx+1}/{total})")
                                progress_bar.progress(int((idx / total) * 100))
                                try:
                                    cache_mgr.copy_to_cache(cp)
                                    total_copied += 1
                                except Exception as e:
                                    total_errors += 1

                            progress_bar.progress(100)
                            if total_errors > 0:
                                st.warning(f"✅ {total_copied} collections copiées, {total_errors} erreurs")
                            else:
                                st.success(f"✅ {total_copied} collections copiées !")
                            st.rerun()
                    else:
                        # Copier la collection sélectionnée ou toutes les collections de la base
                        if isinstance(collection_for_query, str) and collection_for_query != "ALL":
                            # Mode collection unique
                            collection_path_cache = os.path.join(base_path_cache, collection_for_query)
                            with st.spinner(f"Copie de {collection_for_query}..."):
                                try:
                                    cache_mgr.copy_to_cache(collection_path_cache)
                                    st.success("✅ Copié !")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Erreur: {e}")
                        elif isinstance(collection_for_query, list) and collection_for_query:
                            # Mode multi-collection: copier toutes les collections sélectionnées
                            with st.spinner(f"Copie de {len(collection_for_query)} collections..."):
                                copied = 0
                                for coll in collection_for_query:
                                    cp = os.path.join(base_path_cache, coll)
                                    try:
                                        cache_mgr.copy_to_cache(cp)
                                        copied += 1
                                    except Exception:
                                        pass
                                st.success(f"✅ {copied} collections copiées !")
                                st.rerun()
                        else:
                            # Copier toutes les collections de la base sélectionnée (ALL)
                            with st.spinner(f"Copie de toutes les collections de {base_for_query}..."):
                                for coll in collections_for_query:
                                    cp = os.path.join(base_path_cache, coll)
                                    try:
                                        cache_mgr.copy_to_cache(cp)
                                    except Exception:
                                        pass
                                st.success("✅ Copié !")
                                st.rerun()

            with col_cache2:
                if cache_status['collections_count'] > 0:
                    if st.button("🧹 Vider cache", use_container_width=True, key="btn_cache_clear"):
                        cache_mgr.clear_all_cache()
                        st.rerun()

            with col_cache3:
                # Bouton supprimer cache - seulement en mode collection unique
                if isinstance(collection_for_query, str) and collection_for_query != "ALL" and collections_for_query:
                    collection_path_cache = os.path.join(base_path_cache, collection_for_query)
                    if cache_mgr.is_cached(collection_path_cache):
                        if st.button("🗑️ Supprimer", use_container_width=True, key="btn_cache_del"):
                            cache_mgr.invalidate_cache(collection_path_cache)
                            st.rerun()
                elif isinstance(collection_for_query, list) and collection_for_query:
                    # Mode multi-collection: proposer de supprimer le cache des collections sélectionnées
                    has_cache = any(
                        cache_mgr.is_cached(os.path.join(base_path_cache, c))
                        for c in collection_for_query
                    )
                    if has_cache:
                        if st.button("🗑️ Supprimer caches", use_container_width=True, key="btn_cache_del"):
                            for coll in collection_for_query:
                                cp = os.path.join(base_path_cache, coll)
                                if cache_mgr.is_cached(cp):
                                    cache_mgr.invalidate_cache(cp)
                            st.rerun()

    st.markdown("---")

    question = st.text_area("Question", height=110, placeholder="Saisissez votre question métier…")

    top_k = 30  # Recherche 30 candidats, affiche les 10 meilleurs

    synthesize_all = st.checkbox(
        "Réponse synthétisée unique si 'ALL' est sélectionné",
        value=True,
        help=(
            "Si coché et que la collection choisie est 'ALL', le LLM produira une "
            "seule réponse globale en utilisant le contexte de toutes les collections. "
            "Si décoché, une réponse sera produite pour chaque collection."
        ),
    )

    # Option de re-ranking basé sur les feedbacks
    use_feedback_reranking = st.checkbox(
        "🔄 Utiliser les retours utilisateurs pour améliorer les résultats",
        value=True,
        help=(
            "Si activé, les sources sont re-classées en fonction des feedbacks "
            "précédents : les sources bien notées sont favorisées, celles mal "
            "notées sont pénalisées. Les questions similaires passées sont "
            "également prises en compte."
        ),
    )

    # Bouton pour poser la question
    if st.button("🤖 Poser la question", help="Recherche les documents pertinents et génère une réponse via DALLEM basée sur le contexte trouvé."):
        if not question.strip():
            st.warning("Merci de saisir une question.")
        else:
            if not bases or base_for_query == "(aucune)":
                st.error("Aucune base valide sélectionnée pour la recherche.")
            else:
                db_path_query = os.path.join(base_root, base_for_query)
                os.makedirs(db_path_query, exist_ok=True)

                try:
                    # Réinitialiser le flag de feedback pour permettre un nouveau feedback
                    st.session_state['feedback_submitted'] = False

                    # Stocker les paramètres de la recherche
                    st.session_state['last_search'] = {
                        'base': base_for_query,
                        'collection': collection_for_query,
                        'question': question,
                        'synthesize_all': synthesize_all,
                        'top_k': top_k
                    }

                    # Déterminer le mode de recherche
                    is_multi_collection = isinstance(collection_for_query, list) and len(collection_for_query) > 0

                    if is_multi_collection:
                        # Mode multi-collection (recherche inter-bases)
                        st.info(
                            f"🔀 Recherche inter-bases sur {len(collection_for_query)} collections : "
                            + ", ".join(collection_for_query)
                        )
                        spinner_msg = "🔍 Recherche multi-collection en cours…"
                        with st.spinner(spinner_msg):
                            result = cached_multi_collection_rag_query(
                                db_path=db_path_query,
                                collection_names=tuple(collection_for_query),  # tuple pour hashabilité
                                question=question,
                                top_k=top_k,
                                use_hybrid_search=True,
                                use_bge_reranker=True,
                            )

                        # Stocker le résultat dans session_state
                        st.session_state['last_result'] = result
                        st.session_state['result_type'] = 'multi_collection'

                        # Afficher les stats de provenance
                        if result.get('collection_stats'):
                            stats = result['collection_stats']
                            st.success(
                                f"✅ {sum(stats.values())} sources trouvées : " +
                                ", ".join([f"{k}: {v}" for k, v in stats.items()])
                            )

                    elif isinstance(collection_for_query, str) and collection_for_query != "ALL":
                        # RAG sur une collection unique
                        spinner_msg = "🔍 RAG en cours…"
                        with st.spinner(spinner_msg):
                            # Utiliser le cache pour les requêtes répétées (30 min TTL)
                            result = cached_rag_query(
                                db_path=db_path_query,
                                collection_name=collection_for_query,
                                question=question,
                                top_k=top_k,
                                use_feedback_reranking=use_feedback_reranking,
                                use_query_expansion=True,
                                use_bge_reranker=True,
                            )

                        # Stocker le résultat dans session_state
                        st.session_state['last_result'] = result
                        st.session_state['result_type'] = 'single'

                    elif isinstance(collection_for_query, list) and not collection_for_query:
                        # Liste vide sélectionnée en mode multi-collection
                        st.warning("⚠️ Aucune collection sélectionnée. Veuillez en choisir au moins une.")

                    else:
                        # RAG sur TOUTES les collections de la base
                        collections_all = list_collections_for_base(base_root, base_for_query)
                        if not collections_all:
                            st.warning("Aucune collection trouvée dans cette base.")
                        else:
                            if synthesize_all:
                                st.info(
                                    f"RAG synthétisé sur toutes les collections de la base `{base_for_query}`."
                                )
                                spinner_msg = "🔍 RAG en cours (ALL)…"
                                with st.spinner(spinner_msg):
                                    # Utiliser le cache pour les requêtes répétées (30 min TTL)
                                    result = cached_rag_query(
                                        db_path=db_path_query,
                                        collection_name="ALL",
                                        question=question,
                                        top_k=top_k,
                                        synthesize_all=True,
                                        use_feedback_reranking=use_feedback_reranking,
                                        use_query_expansion=True,
                                        use_bge_reranker=True,
                                    )

                                # Stocker le résultat dans session_state
                                st.session_state['last_result'] = result
                                st.session_state['result_type'] = 'synthesized'

                            else:
                                st.info(
                                    f"RAG sur toutes les collections de la base `{base_for_query}` : "
                                    + ", ".join(collections_all)
                                )
                                all_results = []
                                spinner_msg = "🔍 RAG en cours (toutes collections)…"
                                with st.spinner(spinner_msg):
                                    for coll in collections_all:
                                        try:
                                            # Utiliser le cache pour les requêtes répétées (30 min TTL)
                                            res = cached_rag_query(
                                                db_path=db_path_query,
                                                collection_name=coll,
                                                question=question,
                                                top_k=top_k,
                                                use_feedback_reranking=use_feedback_reranking,
                                                use_query_expansion=True,
                                                use_bge_reranker=True,
                                            )
                                            all_results.append((coll, res))
                                        except Exception as e:
                                            st.error(f"Erreur sur la collection {coll}: {e}")

                                # Stocker les résultats dans session_state
                                st.session_state['last_result'] = all_results
                                st.session_state['result_type'] = 'individual'

                except Exception as e:  # pragma: no cover
                    import traceback
                    import logging
                    # Log complet pour debug (côté serveur uniquement)
                    logging.error(f"Erreur RAG: {e}\n{traceback.format_exc()}")
                    # Message utilisateur sans détails techniques sensibles
                    st.error(f"❌ Erreur pendant l'appel RAG : {type(e).__name__}")
                    st.info("💡 Consultez les logs serveur pour plus de détails.")

    # Afficher les résultats depuis session_state (persiste même après un rerun)
    if 'last_result' in st.session_state and 'last_search' in st.session_state:
        search_params = st.session_state['last_search']
        result_type = st.session_state.get('result_type', 'single')

        # Afficher l'erreur d'ouverture de fichier si elle existe
        if 'file_open_error' in st.session_state:
            st.error(st.session_state['file_open_error'])
            del st.session_state['file_open_error']

        # Avertissement si le cache est obsolète
        cache_outdated = False
        if result_type in ('single', 'synthesized'):
            result = st.session_state['last_result']
            cache_outdated = result.get('cache_outdated', False)
        elif result_type == 'individual':
            all_results = st.session_state['last_result']
            cache_outdated = any(r.get('cache_outdated', False) for _, r in all_results)

        if cache_outdated:
            st.warning(
                "⚠️ **Cache local obsolète** - La base a été modifiée sur le réseau. "
                "Les données proviennent du réseau. Mettez à jour le cache local dans le menu latéral."
            )

        if result_type == 'single':
            # Affichage pour une collection unique
            result = st.session_state['last_result']
            st.markdown(
                f"### 🧠 Réponse (base=`{search_params['base']}`, collection=`{search_params['collection']}`)"
            )
            answer = result.get("answer", "")
            display_answer_with_error_check(answer)

            sources = result.get("sources", [])
            sources = sorted(
                sources, key=lambda s: s.get("score", 0.0), reverse=True
            )

            if sources:
                render_sources_list(
                    sources,
                    global_title="📚 Sources utilisées (triées par pertinence)",
                    show_collection=False,
                )

            with st.expander("🧩 Voir le contexte brut (concaténé – debug)"):
                st.text(result.get("context_str", ""))

        elif result_type == 'synthesized':
            # Affichage pour ALL synthétisé
            result = st.session_state['last_result']
            st.markdown(
                f"### 🧠 Réponse synthétisée (base=`{search_params['base']}`, collections=ALL)"
            )
            answer = result.get("answer", "")
            display_answer_with_error_check(answer)

            sources = result.get("sources", [])
            sources = sorted(
                sources, key=lambda s: s.get("score", 0.0), reverse=True
            )

            if sources:
                render_sources_list(
                    sources,
                    global_title=(
                        "📚 Sources utilisées (toutes collections, triées par pertinence)"
                    ),
                    show_collection=True,
                )

            with st.expander("🧩 Voir le contexte brut (concaténé – debug)"):
                st.text(result.get("context_str", ""))

        elif result_type == 'individual':
            # Affichage pour ALL individuel
            all_results = st.session_state['last_result']
            for coll, result in all_results:
                st.markdown(f"### 🧠 Réponse – Collection `{coll}`")
                answer = result.get("answer", "")
                display_answer_with_error_check(answer)

                sources = result.get("sources", [])
                sources = sorted(
                    sources, key=lambda s: s.get("score", 0.0), reverse=True
                )

                if sources:
                    render_sources_list(
                        sources,
                        global_title=(
                            f"📚 Sources utilisées (collection `{coll}`, triées par pertinence)"
                        ),
                        show_collection=False,
                    )

                with st.expander(
                    f"🧩 Contexte brut pour la collection `{coll}` (debug)"
                ):
                    st.text(result.get("context_str", ""))

        elif result_type == 'multi_collection':
            # Affichage pour recherche multi-collection (inter-bases)
            result = st.session_state['last_result']

            # Déterminer les collections interrogées
            collections_searched = result.get('collections_searched', [])
            collection_stats = result.get('collection_stats', {})

            st.markdown(
                f"### 🧠 Réponse Multi-Collection "
                f"(base=`{search_params['base']}`, collections={len(collections_searched)})"
            )

            # Afficher les stats de provenance
            if collection_stats:
                cols_stats = st.columns(len(collection_stats))
                for i, (coll_name, count) in enumerate(collection_stats.items()):
                    with cols_stats[i]:
                        st.metric(
                            label=coll_name,
                            value=f"{count} sources",
                            delta=None
                        )

            answer = result.get("answer", "")
            display_answer_with_error_check(answer)

            # Afficher le grounding si disponible
            grounding = result.get('grounding')
            if grounding:
                grounding_risk = grounding.get('risk', 'unknown')
                grounding_score = grounding.get('score', 0)

                if grounding_risk == 'low':
                    st.success(f"✅ Fiabilité: {grounding_score:.0%} (faible risque d'hallucination)")
                elif grounding_risk == 'medium':
                    st.warning(f"⚠️ Fiabilité: {grounding_score:.0%} (risque moyen d'hallucination)")
                else:
                    st.error(f"❌ Fiabilité: {grounding_score:.0%} (risque élevé d'hallucination)")

            sources = result.get("sources", [])
            sources = sorted(
                sources, key=lambda s: s.get("score", 0.0), reverse=True
            )

            if sources:
                render_sources_list(
                    sources,
                    global_title=(
                        f"📚 Sources multi-collection (triées par pertinence)"
                    ),
                    show_collection=True,  # Important: montrer la collection d'origine
                )

            # Expander pour les détails
            with st.expander("📊 Détails de la recherche multi-collection"):
                st.write("**Collections interrogées:**", ", ".join(collections_searched))
                st.write("**Répartition des sources:**", collection_stats)

                # Query analysis si disponible
                query_analysis = result.get('query_analysis')
                if query_analysis:
                    st.write("**Analyse de la question:**")
                    st.json(query_analysis)

            with st.expander("🧩 Voir le contexte brut (concaténé – debug)"):
                st.text(result.get("context_str", ""))

        # ========================
        #   SECTION FEEDBACK (SIMPLIFIÉ)
        # ========================
        st.markdown("---")
        st.markdown("### 📝 Cette réponse vous a-t-elle aidé ?")

        # Initialiser l'état du feedback si nécessaire
        if 'feedback_submitted' not in st.session_state:
            st.session_state['feedback_submitted'] = False

        if not st.session_state['feedback_submitted']:
            # Boutons pouce haut / pouce bas
            col_thumbs1, col_thumbs2, col_space = st.columns([1, 1, 3])

            with col_thumbs1:
                thumbs_up = st.button("👍 Oui", type="primary", use_container_width=True, help="La réponse est utile et pertinente")

            with col_thumbs2:
                thumbs_down = st.button("👎 Non", use_container_width=True, help="La réponse n'est pas satisfaisante")

            # Champ pour la réponse attendue (affiché seulement si pouce bas)
            if 'show_suggested_answer' not in st.session_state:
                st.session_state['show_suggested_answer'] = False

            if thumbs_up:
                try:
                    # Récupérer le texte de la réponse
                    if result_type == 'individual':
                        answer_text = "\n---\n".join([
                            f"[{coll}] {res.get('answer', '')}"
                            for coll, res in st.session_state.get('last_result', [])
                        ])
                    else:
                        answer_text = st.session_state.get('last_result', {}).get('answer', '')

                    # Créer le feedback positif
                    feedback = create_feedback(
                        base_name=search_params['base'],
                        collection_name=search_params['collection'],
                        question=search_params['question'],
                        is_positive=True,
                        answer_text=answer_text,
                        top_k_used=search_params.get('top_k', 10),
                        synthesize_all=search_params.get('synthesize_all', False)
                    )

                    # Sauvegarder
                    if feedback_store is None:
                        st.warning("Le système de feedback n'est pas disponible (répertoire inaccessible)")
                    else:
                        feedback_id = feedback_store.add_feedback(feedback)
                        st.session_state['feedback_submitted'] = True
                        st.session_state['last_feedback_id'] = feedback_id
                        st.session_state['feedback_positive'] = True
                        st.rerun()

                except Exception as e:
                    st.error(f"Erreur lors de l'enregistrement du feedback : {e}")

            if thumbs_down:
                st.session_state['show_suggested_answer'] = True
                st.rerun()

            # Formulaire pour la réponse attendue (si pouce bas)
            if st.session_state.get('show_suggested_answer', False):
                st.markdown("---")
                st.markdown("💡 **Aidez-nous à améliorer !** Quelle réponse auriez-vous attendue ?")

                suggested_answer = st.text_area(
                    "Réponse attendue",
                    placeholder="Décrivez la réponse que vous attendiez...",
                    height=100,
                    label_visibility="collapsed"
                )

                col_send, col_cancel = st.columns([1, 1])

                with col_send:
                    if st.button("📤 Envoyer", type="primary", use_container_width=True):
                        try:
                            # Récupérer le texte de la réponse
                            if result_type == 'individual':
                                answer_text = "\n---\n".join([
                                    f"[{coll}] {res.get('answer', '')}"
                                    for coll, res in st.session_state.get('last_result', [])
                                ])
                            else:
                                answer_text = st.session_state.get('last_result', {}).get('answer', '')

                            # Créer le feedback négatif avec suggestion
                            feedback = create_feedback(
                                base_name=search_params['base'],
                                collection_name=search_params['collection'],
                                question=search_params['question'],
                                is_positive=False,
                                suggested_answer=suggested_answer,
                                answer_text=answer_text,
                                top_k_used=search_params.get('top_k', 10),
                                synthesize_all=search_params.get('synthesize_all', False)
                            )

                            # Sauvegarder
                            if feedback_store is None:
                                st.warning("Le système de feedback n'est pas disponible (répertoire inaccessible)")
                            else:
                                feedback_id = feedback_store.add_feedback(feedback)
                                st.session_state['feedback_submitted'] = True
                                st.session_state['last_feedback_id'] = feedback_id
                                st.session_state['feedback_positive'] = False
                                st.session_state['show_suggested_answer'] = False
                                st.rerun()

                        except Exception as e:
                            st.error(f"Erreur lors de l'enregistrement du feedback : {e}")

                with col_cancel:
                    if st.button("❌ Annuler", use_container_width=True):
                        st.session_state['show_suggested_answer'] = False
                        st.rerun()

        else:
            if st.session_state.get('feedback_positive', True):
                st.success("✅ Merci pour votre retour positif !")
            else:
                st.success("✅ Merci ! Votre suggestion sera prise en compte.")

            if st.button("🔄 Nouvelle question", help="Poser une nouvelle question"):
                st.session_state['feedback_submitted'] = False
                st.session_state['show_suggested_answer'] = False
                if 'last_result' in st.session_state:
                    del st.session_state['last_result']
                if 'last_search' in st.session_state:
                    del st.session_state['last_search']
                st.rerun()


# ========================
#   TAB TABLEAU DE BORD ANALYTIQUE (SIMPLIFIÉ)
#   Accessible uniquement aux utilisateurs autorisés
# ========================
if tab_analytics is not None:
    with tab_analytics:
        st.subheader("📊 Tableau de bord - Retours utilisateurs")

        # Sélection de la base à analyser
        st.markdown("### Filtres")

        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            analytics_base = st.selectbox(
                "Base à analyser",
                options=["Toutes les bases"] + bases,
                key="analytics_base"
            )

        with col_filter2:
            analytics_period = st.selectbox(
                "Période d'analyse",
                options=[7, 14, 30, 60, 90],
                format_func=lambda x: f"Derniers {x} jours",
                index=2,
                key="analytics_period"
            )

        # Récupérer les statistiques
        base_filter = None if analytics_base == "Toutes les bases" else analytics_base

        if feedback_store is None:
            st.warning("Le système de feedback n'est pas disponible (répertoire inaccessible)")
            stats = {"total_feedbacks": 0}
            trends = {"total_feedbacks_in_period": 0, "trend_data": []}
        else:
            try:
                stats = feedback_store.get_statistics(base_filter)
                trends = feedback_store.get_feedback_trends(base_filter, days=analytics_period)
            except Exception as e:
                st.error(f"Erreur lors du chargement des statistiques : {e}")
                stats = {"total_feedbacks": 0}
                trends = {"total_feedbacks_in_period": 0, "trend_data": []}

        # Métriques principales
        st.markdown("### Métriques globales")

        if stats["total_feedbacks"] > 0:
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.metric(
                    "Total feedbacks",
                    stats["total_feedbacks"],
                    help="Nombre total de feedbacks enregistrés"
                )

            with col_m2:
                positive = stats.get("positive_feedbacks", 0)
                negative = stats.get("negative_feedbacks", 0)
                st.metric(
                    "👍 Positifs",
                    positive,
                    help="Nombre de réponses jugées utiles"
                )

            with col_m3:
                satisfaction_rate = stats.get("satisfaction_rate", 0)
                st.metric(
                    "Taux de satisfaction",
                    f"{satisfaction_rate}%",
                    help="Pourcentage de feedbacks positifs"
                )

            # Graphiques de tendance
            st.markdown("### Évolution des feedbacks")

            trend_data = trends.get("trend_data", [])
            if trend_data:
                import pandas as pd

                df_trends = pd.DataFrame(trend_data)
                df_trends["date"] = pd.to_datetime(df_trends["date"])

                # Graphique empilé positif/négatif
                st.bar_chart(
                    df_trends.set_index("date")[["positive", "negative"]],
                    use_container_width=True
                )
                st.caption("Feedbacks positifs (👍) et négatifs (👎) par jour")

            else:
                st.info(f"Pas de données de tendance pour les {analytics_period} derniers jours.")

            # Statistiques par collection
            st.markdown("### Satisfaction par collection")

            collection_stats = stats.get("collection_stats", {})
            if collection_stats:
                coll_data = []
                for coll_name, coll_stats in collection_stats.items():
                    coll_data.append({
                        "Collection": coll_name,
                        "Total": coll_stats.get("count", 0),
                        "👍": coll_stats.get("positive", 0),
                        "👎": coll_stats.get("negative", 0),
                        "Satisfaction": f"{coll_stats.get('satisfaction_rate', 0)}%"
                    })

                # Trier par satisfaction décroissante
                coll_data.sort(key=lambda x: float(x["Satisfaction"].replace("%", "")), reverse=True)
                st.table(coll_data)
            else:
                st.info("Aucune statistique par collection disponible.")

            # Questions avec feedback négatif
            st.markdown("### Questions avec feedback négatif")

            negative_questions = stats.get("negative_questions", [])
            if negative_questions:
                st.warning(f"{len(negative_questions)} question(s) avec feedback négatif")

                for idx, q in enumerate(negative_questions[:10], 1):
                    question_text = q.get('question', 'N/A')
                    question_preview = question_text[:80] + "..." if len(question_text) > 80 else question_text

                    with st.expander(f"👎 {question_preview}"):
                        st.write(f"**Question complète**: {question_text}")
                        st.write(f"**Base**: {q.get('base', 'N/A')}")
                        st.write(f"**Collection**: {q.get('collection', 'N/A')}")
                        st.write(f"**Date**: {q.get('timestamp', 'N/A')[:10] if q.get('timestamp') else 'N/A'}")
                        if q.get('suggested_answer'):
                            st.markdown("**💡 Réponse attendue par l'utilisateur:**")
                            st.info(q.get('suggested_answer'))
            else:
                st.success("Aucune question avec feedback négatif !")

            # Statistiques utilisateurs
            st.markdown("### Activité par utilisateur")

            user_stats = stats.get("user_stats", {})
            if user_stats:
                user_data = [
                    {"Utilisateur": user, "Feedbacks": count}
                    for user, count in sorted(user_stats.items(), key=lambda x: x[1], reverse=True)
                ]
                st.table(user_data[:10])
            else:
                st.info("Aucune statistique utilisateur disponible.")

            # Export des données
            st.markdown("---")
            st.markdown("### Export des données")

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                if st.button("📥 Exporter en CSV", use_container_width=True, help="Génère un fichier CSV téléchargeable contenant tous les feedbacks enregistrés."):
                    if feedback_store is None:
                        st.warning("Le système de feedback n'est pas disponible")
                    else:
                        try:
                            csv_content = feedback_store.export_feedbacks_csv(base_filter)
                            st.download_button(
                                label="💾 Télécharger le CSV",
                                data=csv_content.encode("utf-8-sig"),
                                file_name=f"feedbacks_{analytics_base.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'export : {e}")

            with col_export2:
                if st.button("📊 Rafraîchir les statistiques", use_container_width=True, help="Recharge les données et recalcule les statistiques du tableau de bord."):
                    st.rerun()

        else:
            st.info("Aucun feedback enregistré pour le moment.")
            st.markdown("""
            **Comment collecter des feedbacks ?**

            1. Posez une question dans l'onglet "❓ Questions RAG"
            2. Après avoir reçu une réponse, cliquez sur 👍 ou 👎
            3. Si vous cliquez 👎, vous pouvez indiquer la réponse attendue
            4. Les statistiques apparaîtront ici automatiquement
            """)
