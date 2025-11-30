# config_manager.py
"""
Gestionnaire de configuration pour les répertoires de stockage.

Ce module permet de:
- Charger/sauvegarder la configuration des chemins
- Valider l'existence des répertoires
- Proposer la création des répertoires manquants
- Afficher une interface de configuration si nécessaire
- Gerer le mode offline avec fallback automatique N:\ -> D:\
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict, field


# Fichier de configuration à la racine du projet
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

# Chemins réseau primaires (N:\)
PRIMARY_NETWORK_BASE = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE"

# Chemins locaux de fallback (D:\)
FALLBACK_LOCAL_BASE = r"D:\FAISS_DATABASE"

# Valeurs par défaut (chemins réseau PROP)
DEFAULT_CONFIG = {
    "base_root_dir": os.path.join(PRIMARY_NETWORK_BASE, "BaseDB"),
    "csv_import_dir": os.path.join(PRIMARY_NETWORK_BASE, "CSV_Ingestion"),
    "csv_export_dir": os.path.join(PRIMARY_NETWORK_BASE, "Fichiers_Tracking_CSV"),
    "feedback_dir": os.path.join(PRIMARY_NETWORK_BASE, "Feedbacks"),
    "offline_mode": False,
}

# Configuration fallback pour le mode offline
FALLBACK_CONFIG = {
    "base_root_dir": os.path.join(FALLBACK_LOCAL_BASE, "BaseDB"),
    "csv_import_dir": os.path.join(FALLBACK_LOCAL_BASE, "CSV_Ingestion"),
    "csv_export_dir": os.path.join(FALLBACK_LOCAL_BASE, "Fichiers_Tracking_CSV"),
    "feedback_dir": os.path.join(FALLBACK_LOCAL_BASE, "Feedbacks"),
    "offline_mode": True,
}


@dataclass
class StorageConfig:
    """Configuration des répertoires de stockage."""
    base_root_dir: str
    csv_import_dir: str
    csv_export_dir: str
    feedback_dir: str
    offline_mode: bool = False

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "StorageConfig":
        return cls(
            base_root_dir=data.get("base_root_dir", DEFAULT_CONFIG["base_root_dir"]),
            csv_import_dir=data.get("csv_import_dir", DEFAULT_CONFIG["csv_import_dir"]),
            csv_export_dir=data.get("csv_export_dir", DEFAULT_CONFIG["csv_export_dir"]),
            feedback_dir=data.get("feedback_dir", DEFAULT_CONFIG["feedback_dir"]),
            offline_mode=data.get("offline_mode", DEFAULT_CONFIG.get("offline_mode", False)),
        )


def load_config() -> StorageConfig:
    """
    Charge la configuration depuis le fichier config.json.
    Si le fichier n'existe pas, utilise les valeurs par défaut.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return StorageConfig.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[CONFIG] Erreur de lecture du fichier config: {e}")
            # Fallback aux valeurs par défaut
            pass

    return StorageConfig.from_dict(DEFAULT_CONFIG)


def save_config(config: StorageConfig) -> bool:
    """
    Sauvegarde la configuration dans le fichier config.json.

    Returns:
        True si la sauvegarde a réussi, False sinon.
    """
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"[CONFIG] Erreur de sauvegarde du fichier config: {e}")
        return False


# =====================================================================
#  FALLBACK AUTOMATIQUE N:\ -> D:\
# =====================================================================

def is_network_path_accessible(path: Optional[str] = None) -> bool:
    """
    Vérifie si le chemin réseau primaire est accessible.

    Args:
        path: Chemin à tester (défaut: PRIMARY_NETWORK_BASE)

    Returns:
        True si le chemin est accessible en lecture
    """
    test_path = path or PRIMARY_NETWORK_BASE

    try:
        # Pour les chemins réseau Windows (N:\, \\server\share)
        if test_path.startswith(("N:", "n:", r"\\", "//")):
            return os.path.exists(test_path) and os.access(test_path, os.R_OK)
        return os.path.exists(test_path) and os.access(test_path, os.R_OK)
    except (OSError, PermissionError, Exception) as e:
        print(f"[CONFIG] Chemin réseau inaccessible ({test_path}): {e}")
        return False


def get_effective_paths(force_offline: bool = False) -> Dict[str, str]:
    """
    Retourne les chemins effectifs avec fallback automatique.

    Si le chemin réseau N:\ n'est pas accessible, utilise D:\ automatiquement.

    Args:
        force_offline: Si True, utilise toujours les chemins locaux

    Returns:
        Dict avec les chemins effectifs
    """
    config = load_config()

    # Mode offline forcé
    if force_offline or config.offline_mode:
        print("[CONFIG] Mode offline actif - utilisation des chemins locaux (D:\\)")
        return {
            "base_root_dir": FALLBACK_CONFIG["base_root_dir"],
            "csv_import_dir": FALLBACK_CONFIG["csv_import_dir"],
            "csv_export_dir": FALLBACK_CONFIG["csv_export_dir"],
            "feedback_dir": FALLBACK_CONFIG["feedback_dir"],
            "using_fallback": True,
            "offline_mode": True,
        }

    # Tester l'accessibilité du chemin réseau
    if is_network_path_accessible(config.base_root_dir):
        print(f"[CONFIG] Chemin réseau accessible: {config.base_root_dir}")
        return {
            "base_root_dir": config.base_root_dir,
            "csv_import_dir": config.csv_import_dir,
            "csv_export_dir": config.csv_export_dir,
            "feedback_dir": config.feedback_dir,
            "using_fallback": False,
            "offline_mode": False,
        }

    # Fallback automatique vers D:\
    print(f"[CONFIG] Chemin réseau inaccessible, fallback vers D:\\")

    # Vérifier que le fallback existe
    if not os.path.exists(FALLBACK_LOCAL_BASE):
        try:
            os.makedirs(FALLBACK_LOCAL_BASE, exist_ok=True)
            print(f"[CONFIG] Création du répertoire fallback: {FALLBACK_LOCAL_BASE}")
        except Exception as e:
            print(f"[CONFIG] Impossible de créer le répertoire fallback: {e}")

    return {
        "base_root_dir": FALLBACK_CONFIG["base_root_dir"],
        "csv_import_dir": FALLBACK_CONFIG["csv_import_dir"],
        "csv_export_dir": FALLBACK_CONFIG["csv_export_dir"],
        "feedback_dir": FALLBACK_CONFIG["feedback_dir"],
        "using_fallback": True,
        "offline_mode": False,  # Fallback automatique, pas mode offline explicite
    }


def set_offline_mode(enabled: bool) -> bool:
    """
    Active ou désactive le mode offline.

    Args:
        enabled: True pour activer, False pour désactiver

    Returns:
        True si la configuration a été sauvegardée
    """
    config = load_config()
    config.offline_mode = enabled

    if enabled:
        # En mode offline, utiliser les chemins locaux
        config.base_root_dir = FALLBACK_CONFIG["base_root_dir"]
        config.csv_import_dir = FALLBACK_CONFIG["csv_import_dir"]
        config.csv_export_dir = FALLBACK_CONFIG["csv_export_dir"]
        config.feedback_dir = FALLBACK_CONFIG["feedback_dir"]
        print("[CONFIG] Mode offline activé - chemins locaux configurés")
    else:
        # En mode online, revenir aux chemins réseau
        config.base_root_dir = DEFAULT_CONFIG["base_root_dir"]
        config.csv_import_dir = DEFAULT_CONFIG["csv_import_dir"]
        config.csv_export_dir = DEFAULT_CONFIG["csv_export_dir"]
        config.feedback_dir = DEFAULT_CONFIG["feedback_dir"]
        print("[CONFIG] Mode online activé - chemins réseau configurés")

    return save_config(config)


def is_offline_mode() -> bool:
    """
    Vérifie si le mode offline est activé.

    Returns:
        True si mode offline, False sinon
    """
    config = load_config()
    return config.offline_mode


def ensure_fallback_directories() -> Tuple[bool, List[str]]:
    """
    S'assure que les répertoires fallback (D:\) existent.

    Returns:
        Tuple (succès, liste des erreurs)
    """
    errors = []
    directories = [
        FALLBACK_CONFIG["base_root_dir"],
        FALLBACK_CONFIG["csv_import_dir"],
        FALLBACK_CONFIG["csv_export_dir"],
        FALLBACK_CONFIG["feedback_dir"],
    ]

    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            errors.append(f"Impossible de créer {dir_path}: {e}")

    return len(errors) == 0, errors


def validate_directory(path: str) -> Tuple[bool, str]:
    """
    Valide qu'un répertoire existe et est accessible.

    Args:
        path: Chemin du répertoire à valider

    Returns:
        Tuple (valide, message)
    """
    if not path or path.strip() == "":
        return False, "Chemin vide"

    path = path.strip()

    # Vérifier si le chemin existe
    if not os.path.exists(path):
        return False, f"Le répertoire n'existe pas: {path}"

    # Vérifier si c'est un répertoire
    if not os.path.isdir(path):
        return False, f"Le chemin n'est pas un répertoire: {path}"

    # Vérifier les permissions d'écriture
    try:
        test_file = os.path.join(path, ".write_test_temp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (IOError, PermissionError) as e:
        return False, f"Pas de permission d'écriture: {path}"

    return True, "OK"


def validate_all_directories(config: StorageConfig) -> Dict[str, Tuple[bool, str]]:
    """
    Valide tous les répertoires de la configuration.

    Returns:
        Dict avec le nom du répertoire et le tuple (valide, message)
    """
    results = {}

    directories = {
        "base_root_dir": ("Bases FAISS", config.base_root_dir),
        "csv_import_dir": ("CSV d'ingestion", config.csv_import_dir),
        "csv_export_dir": ("CSV de tracking", config.csv_export_dir),
        "feedback_dir": ("Feedbacks", config.feedback_dir),
    }

    for key, (label, path) in directories.items():
        valid, message = validate_directory(path)
        results[key] = (valid, message, label, path)

    return results


def create_directory(path: str) -> Tuple[bool, str]:
    """
    Crée un répertoire (avec les parents si nécessaire).

    Returns:
        Tuple (succès, message)
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True, f"Répertoire créé: {path}"
    except PermissionError:
        return False, f"Permission refusée pour créer: {path}"
    except OSError as e:
        return False, f"Erreur de création: {e}"


def get_missing_directories(config: StorageConfig) -> List[Tuple[str, str, str]]:
    """
    Retourne la liste des répertoires manquants.

    Returns:
        Liste de tuples (key, label, path) pour les répertoires manquants
    """
    results = validate_all_directories(config)
    missing = []

    for key, (valid, message, label, path) in results.items():
        if not valid:
            missing.append((key, label, path, message))

    return missing


def is_config_valid(config: StorageConfig) -> bool:
    """
    Vérifie si la configuration est valide (tous les répertoires existent).
    """
    results = validate_all_directories(config)
    return all(valid for valid, _, _, _ in results.values())


def ensure_directories_exist(config: StorageConfig, create_if_missing: bool = False) -> Tuple[bool, List[str]]:
    """
    S'assure que tous les répertoires existent.

    Args:
        config: Configuration à valider
        create_if_missing: Si True, tente de créer les répertoires manquants

    Returns:
        Tuple (tous_valides, liste_des_erreurs)
    """
    errors = []
    results = validate_all_directories(config)

    for key, (valid, message, label, path) in results.items():
        if not valid:
            if create_if_missing:
                success, create_msg = create_directory(path)
                if not success:
                    errors.append(f"{label}: {create_msg}")
            else:
                errors.append(f"{label}: {message}")

    return len(errors) == 0, errors


# =====================================================================
#  FONCTIONS POUR STREAMLIT
# =====================================================================

def render_config_page_streamlit():
    """
    Affiche la page de configuration dans Streamlit.
    À utiliser quand les répertoires ne sont pas valides.
    """
    import streamlit as st

    st.title("Configuration des répertoires de stockage")
    st.warning("Les répertoires de stockage ne sont pas configurés ou inaccessibles.")
    st.info("Veuillez configurer les chemins ci-dessous pour continuer.")

    # Charger la configuration actuelle
    config = load_config()

    st.markdown("---")

    # Formulaire de configuration
    st.subheader("Répertoires de stockage")

    new_base_root = st.text_input(
        "Répertoire des bases FAISS",
        value=config.base_root_dir,
        help="Chemin absolu vers le dossier contenant les bases FAISS"
    )

    new_csv_import = st.text_input(
        "Répertoire des CSV d'ingestion",
        value=config.csv_import_dir,
        help="Chemin absolu vers le dossier contenant les CSV pour l'ingestion"
    )

    new_csv_export = st.text_input(
        "Répertoire des CSV de tracking",
        value=config.csv_export_dir,
        help="Chemin absolu vers le dossier pour exporter les CSV de suivi"
    )

    new_feedback = st.text_input(
        "Répertoire des feedbacks",
        value=config.feedback_dir,
        help="Chemin absolu vers le dossier pour stocker les feedbacks utilisateurs"
    )

    # Créer une nouvelle configuration
    new_config = StorageConfig(
        base_root_dir=new_base_root,
        csv_import_dir=new_csv_import,
        csv_export_dir=new_csv_export,
        feedback_dir=new_feedback,
    )

    # Afficher le statut de chaque répertoire
    st.markdown("---")
    st.subheader("Statut des répertoires")

    results = validate_all_directories(new_config)
    all_valid = True

    for key, (valid, message, label, path) in results.items():
        if valid:
            st.success(f"✅ {label}: {path}")
        else:
            st.error(f"❌ {label}: {message}")
            all_valid = False

    st.markdown("---")

    # Boutons d'action
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Créer les répertoires manquants", type="secondary"):
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

    with col2:
        if st.button("Sauvegarder la configuration", type="primary", disabled=not all_valid):
            if save_config(new_config):
                st.success("Configuration sauvegardée!")
                st.rerun()
            else:
                st.error("Erreur lors de la sauvegarde")

    with col3:
        if st.button("Utiliser les valeurs par défaut"):
            default_config = StorageConfig.from_dict(DEFAULT_CONFIG)
            save_config(default_config)
            st.rerun()

    # Message d'aide
    st.markdown("---")
    st.info("""
    **Aide:**
    - Les chemins doivent être des chemins absolus (ex: `C:\\Data\\FAISS` ou `N:\\Partage\\Data`)
    - Les répertoires doivent être accessibles en lecture et écriture
    - Si les répertoires n'existent pas, cliquez sur "Créer les répertoires manquants"
    - Une fois tous les répertoires valides (✅), cliquez sur "Sauvegarder la configuration"
    """)

    return all_valid


def check_and_show_config_if_needed() -> Optional[StorageConfig]:
    """
    Vérifie la configuration et affiche la page de configuration si nécessaire.

    Returns:
        StorageConfig si valide, None si la page de configuration est affichée
    """
    import streamlit as st

    config = load_config()

    if is_config_valid(config):
        return config

    # Configuration invalide - afficher la page de configuration
    render_config_page_streamlit()
    st.stop()  # Arrêter l'exécution du reste de l'app
    return None


# =====================================================================
#  COMPATIBILITÉ AVEC L'ANCIENNE INTERFACE
# =====================================================================

def get_base_root_dir(use_fallback: bool = True) -> str:
    """
    Retourne le répertoire des bases FAISS.

    Args:
        use_fallback: Si True, utilise le fallback automatique si réseau inaccessible

    Returns:
        Chemin du répertoire des bases FAISS
    """
    if use_fallback:
        paths = get_effective_paths()
        return paths["base_root_dir"]
    return load_config().base_root_dir


def get_csv_import_dir(use_fallback: bool = True) -> str:
    """
    Retourne le répertoire des CSV d'ingestion.

    Args:
        use_fallback: Si True, utilise le fallback automatique si réseau inaccessible
    """
    if use_fallback:
        paths = get_effective_paths()
        return paths["csv_import_dir"]
    return load_config().csv_import_dir


def get_csv_export_dir(use_fallback: bool = True) -> str:
    """
    Retourne le répertoire des CSV de tracking.

    Args:
        use_fallback: Si True, utilise le fallback automatique si réseau inaccessible
    """
    if use_fallback:
        paths = get_effective_paths()
        return paths["csv_export_dir"]
    return load_config().csv_export_dir


def get_feedback_dir(use_fallback: bool = True) -> str:
    """
    Retourne le répertoire des feedbacks.

    Args:
        use_fallback: Si True, utilise le fallback automatique si réseau inaccessible
    """
    if use_fallback:
        paths = get_effective_paths()
        return paths["feedback_dir"]
    return load_config().feedback_dir


def get_storage_status() -> Dict[str, any]:
    """
    Retourne le statut complet du stockage.

    Returns:
        Dict avec informations sur les chemins et leur accessibilité
    """
    config = load_config()
    effective = get_effective_paths()

    return {
        "config": {
            "base_root_dir": config.base_root_dir,
            "csv_import_dir": config.csv_import_dir,
            "csv_export_dir": config.csv_export_dir,
            "feedback_dir": config.feedback_dir,
            "offline_mode": config.offline_mode,
        },
        "effective": effective,
        "network_accessible": is_network_path_accessible(),
        "fallback_base": FALLBACK_LOCAL_BASE,
        "primary_base": PRIMARY_NETWORK_BASE,
    }
