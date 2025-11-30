# Architecture Technique - RaGME_UP PROP

Documentation technique complète du système RAG pour développeurs et mainteneurs.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Pipeline d'ingestion](#pipeline-dingestion)
4. [Système de Chunking](#système-de-chunking)
5. [Parsers de documents](#parsers-de-documents)
6. [Stockage vectoriel FAISS](#stockage-vectoriel-faiss)
7. [Cache local](#cache-local)
8. [Pipeline de requête](#pipeline-de-requête)
9. [Configuration](#configuration)
10. [API et modèles](#api-et-modèles)
11. [Dépendances](#dépendances)

---

## Vue d'ensemble

RaGME_UP PROP est un système RAG (Retrieval-Augmented Generation) conçu pour les documents techniques aéronautiques, avec support spécialisé pour les réglementations EASA (CS, AMC, GM).

### Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                              │
│  (streamlit_RAG.py, csv_generator_gui.py)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   INGESTION     │  │    REQUÊTE      │  │   FEEDBACK      │
│ rag_ingestion.py│  │  rag_query.py   │  │ feedback_store  │
└────────┬────────┘  └────────┬────────┘  └─────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAITEMENT DOCUMENTS                        │
│  pdf_processing.py | docx_processing.py | xml_processing.py     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CHUNKING                                  │
│  chunking.py | easa_sections.py                                 │
│  - Analyse densité adaptative                                   │
│  - Détection sections EASA                                      │
│  - Augmentation sémantique                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS & STOCKAGE                         │
│  models_utils.py | faiss_store.py                               │
│  - Snowflake Arctic API (1024 dims)                             │
│  - FAISS index par collection                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
RAG_v6/
├── streamlit_RAG.py          # Application Streamlit principale (98 KB)
├── csv_generator_gui.py      # GUI CustomTkinter pour CSV (33 KB)
│
├── # === INGESTION ===
├── rag_ingestion.py          # Orchestration ingestion classique (22 KB)
├── ingestion_pipeline.py     # Pipeline d'ingestion optimisé 2 phases (NEW)
├── pdf_processing.py         # Parser PDF (31 KB)
├── docx_processing.py        # Parser DOCX (3.4 KB)
├── xml_processing.py         # Parser XML EASA (11 KB)
├── csv_processing.py         # Parser CSV (0.7 KB)
├── confluence_processing.py  # API Confluence (9 KB)
│
├── # === CHUNKING ===
├── chunking.py               # Chunking adaptatif (64 KB, 1865 lignes)
├── semantic_chunking.py      # Chunking sémantique intelligent (NEW)
├── easa_sections.py          # Détection sections EASA (4.3 KB)
│
├── # === REQUÊTE ===
├── rag_query.py              # Pipeline de requête + HyDE + Lost in Middle (22 KB)
├── advanced_search.py        # Recherche avancée (23 KB)
├── hybrid_search.py          # Recherche hybride BM25 + Dense (NEW)
├── query_understanding.py    # Analyse d'intention de requête (NEW)
│
├── # === QUALITÉ RAG ===
├── answer_grounding.py       # Détection hallucinations (NEW)
├── rag_metrics.py            # Métriques RAGAS (NEW)
├── semantic_cache.py         # Cache sémantique des réponses (NEW)
│
├── # === OCR ===
├── llm_ocr.py                # OCR Vision LLM + rotation auto (NEW)
│
├── # === STOCKAGE ===
├── faiss_store.py            # Gestion FAISS (12 KB)
├── feedback_store.py         # Stockage feedbacks (24 KB)
│
├── # === CONFIGURATION ===
├── config_manager.py         # Gestion chemins (14 KB)
├── models_utils.py           # Embeddings/LLM (31 KB)
│
├── # === SCRIPTS ===
├── install.bat               # Installation Windows
├── launch.bat                # Lancement Windows
│
├── # === DOCUMENTATION ===
├── README.md
├── GUIDE_UTILISATEUR.md
├── INSTALLATION_RESEAU.md
├── ARCHITECTURE_TECHNIQUE.md # Ce document
│
├── # === CONFIGURATION (générés) ===
├── config.json               # Configuration utilisateur
├── requirements.txt          # Dépendances Python
└── .gitignore
```

---

## Pipeline d'ingestion

### Flux de données

```
Fichiers sources (PDF, DOCX, XML, TXT, CSV)
          │
          ▼
┌─────────────────────────────────────────┐
│  1. CHARGEMENT PARALLÈLE               │
│  ThreadPoolExecutor (CPU count workers) │
│  load_file_content()                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  2. PARSING PAR FORMAT                  │
│  - PDF: pdfplumber (tableaux) + fallbacks│
│  - DOCX: python-docx                    │
│  - XML: ElementTree + patterns EASA     │
│  - TXT/MD/CSV: lecture native           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. DÉTECTION TYPE DOCUMENT             │
│  detect_sections() → EASA ou Générique  │
└────────────────┬────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│ EASA Parser │     │ Smart Chunk │
│ sections    │     │ générique   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. ANALYSE DENSITÉ                     │
│  _calculate_content_density()           │
│  → very_dense, dense, normal, sparse    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  5. CHUNKING ADAPTATIF                  │
│  _get_adaptive_chunk_size()             │
│  chunk_easa_sections() ou               │
│  smart_chunk_generic()                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  6. AUGMENTATION                        │
│  augment_chunks()                       │
│  - Mots-clés (TF scoring)               │
│  - Phrases clés (shall/must/defined)    │
│  - Densité (type + score)               │
│  add_cross_references_to_chunks()       │
│  - Références CS/AMC/GM/FAR/JAR         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  7. EMBEDDING                           │
│  Snowflake Arctic API (1024 dims)       │
│  Batch size: 32, max 28000 chars/texte  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  8. STOCKAGE FAISS                      │
│  Batch de 4000 chunks                   │
│  index.faiss + metadata.json            │
└─────────────────────────────────────────┘
```

### Fonction principale

```python
# rag_ingestion.py
def ingest_documents(
    file_paths: List[str],
    db_path: str,
    collection_name: str,
    chunk_size: int = 1000,         # Taille de base
    use_easa_sections: bool = True, # Détection EASA
    xml_configs: dict = None,       # Config XML optionnelle
    progress_callback = None        # Callback progression
) -> dict:
    """
    Retourne:
    {
        'ingested': [...],      # Fichiers ingérés
        'skipped': [...],       # Fichiers déjà présents
        'missing': [...],       # Fichiers introuvables
        'attachments': [...],   # Pièces jointes extraites
        'errors': [...]         # Erreurs rencontrées
    }
    """
```

---

## Système de Chunking

### Architecture du chunking (`chunking.py`)

Le fichier `chunking.py` (1865 lignes) implémente un système de chunking sophistiqué.

### 1. Analyse de densité

```python
# Dictionnaire des tailles par densité
CHUNK_SIZES = {
    "very_dense": 800,   # Code, formules, tableaux
    "dense": 1200,       # Spécifications, listes
    "normal": 1500,      # Prose technique
    "sparse": 2000       # Narratif, introductions
}

def _calculate_content_density(text: str) -> dict:
    """
    Analyse le contenu et retourne:
    {
        'technical_term_ratio': float,  # Ratio termes techniques
        'numeric_ratio': float,         # Ratio nombres/formules
        'avg_sentence_length': float,   # Longueur moyenne phrases
        'list_density': float,          # Densité listes/tableaux
        'reference_density': float,     # Densité références
        'acronym_ratio': float,         # Ratio acronymes
        'bracket_density': float,       # Densité parenthèses
        'density_type': str,            # Type résultant
        'density_score': float          # Score 0-1
    }
    """
```

**Mots-clés techniques (80+)** :
```python
TECHNICAL_TERMS = {
    'aircraft', 'airplane', 'aeroplane', 'structure', 'load', 'stress',
    'fatigue', 'damage', 'tolerance', 'inspection', 'maintenance',
    'certification', 'compliance', 'requirement', 'airworthiness',
    'fuel', 'engine', 'propeller', 'system', 'electrical', 'hydraulic',
    'pneumatic', 'avionics', 'flight', 'control', 'landing', 'gear',
    'wing', 'fuselage', 'empennage', 'nacelle', 'pylon', ...
}
```

### 2. Chunking EASA spécialisé

```python
def chunk_easa_sections(
    sections: List[dict],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 200,
    merge_small_sections: bool = True,
    add_context_prefix: bool = True
) -> List[dict]:
    """
    Chunks les sections EASA détectées.

    Format entrée:
    {
        'section_id': 'CS 25.571',
        'section_kind': 'CS',
        'title': 'Damage tolerance and fatigue evaluation',
        'content': '...'
    }

    Format sortie (chunk):
    {
        'text': '[CS 25.571 - Damage tolerance...]\nContenu...',
        'source_file': 'CS-25.pdf',
        'chunk_id': 'CS 25.571__chunk_0',
        'section_id': 'CS 25.571',
        'section_kind': 'CS',
        'section_title': 'Damage tolerance...',
        'is_complete_section': True/False
    }
    """
```

**Patterns de détection** (`easa_sections.py`) :
```python
EASA_PATTERNS = {
    'CS': r'CS[-\s]?25[.\s]?\d+',
    'AMC': r'AMC[-\s]?25[.\s]?\d+',
    'GM': r'GM[-\s]?25[.\s]?\d+',
    'CS-E': r'CS[-\s]?E[-\s]?\d+',
    'CS-APU': r'CS[-\s]?APU[-\s]?\d+'
}
```

### 3. Chunking générique intelligent

```python
def smart_chunk_generic(
    text: str,
    source_file: str,
    chunk_size: int = 1500,
    min_chunk_size: int = 200,
    overlap: int = 100,
    add_source_prefix: bool = True,
    preserve_lists: bool = True,
    preserve_headers: bool = True
) -> List[dict]:
    """
    Chunking intelligent pour documents non-EASA.

    Détection structure via _detect_document_structure():
    - Headers numérotés: "1.1 Overview", "2.3.1 Details"
    - Headers capitalisés: "INTRODUCTION", "REQUIREMENTS"
    - Listes à puces: "- item", "* item", "• item"
    - Listes numérotées: "1. item", "a) item", "(i) item"

    Règles:
    - Headers restent avec leur contenu
    - Listes ne sont jamais coupées
    - Coupure aux fins de phrases (. ! ?)
    - Overlap ajouté après headers/sources
    """
```

### 4. Chunking parent-enfant

```python
def create_parent_child_chunks(
    text: str,
    source_file: str,
    parent_size: int = 3000,
    child_size: int = 800,
    child_overlap: int = 100
) -> Tuple[List[dict], List[dict]]:
    """
    Crée une hiérarchie de chunks:
    - Parent: contexte large (3000+ chars)
    - Children: fragments précis (800 chars)

    Retourne: (parent_chunks, child_chunks)
    Chaque child a un 'parent_id' pour liaison.
    """
```

### 5. Augmentation des chunks

```python
def augment_chunks(
    chunks: List[dict],
    add_keywords: bool = True,
    add_key_phrases: bool = True,
    add_density_info: bool = True
) -> List[dict]:
    """
    Enrichit chaque chunk avec métadonnées.
    """

def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extraction TF-based avec:
    - Filtrage stopwords FR+EN
    - Bonus 2x pour termes techniques aéronautiques
    - Bonus 1.3x pour mots > 8 caractères
    - Extraction codes références (CS, AMC, GM, FAR, JAR)
    """

def _extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extraction phrases clés:
    - Requirements: contient "shall", "must"
    - Définitions: contient "means", "is defined as"
    """
```

### 6. Références croisées

```python
def extract_cross_references(text: str) -> List[dict]:
    """
    Détecte les références inter-sections.

    Patterns:
    - Direct: 'CS 25.571', 'AMC 25.1309'
    - Contexte: 'see CS...', 'refer to AMC...', 'in accordance with...'
    - FAR/JAR: 'FAR 25.571', 'JAR 25.571'
    - Interne: 'paragraph (a)', 'sub-paragraph (1)'

    Retourne:
    [{
        'ref_type': 'CS' | 'AMC' | 'GM' | 'FAR' | 'JAR' | 'internal',
        'ref_id': 'CS 25.571',
        'normalized_id': 'cs_25_571',
        'position': 42
    }, ...]
    """

def add_cross_references_to_chunks(chunks: List[dict], max_refs: int = 5):
    """
    Ajoute chunk['references_to'] = ['CS 25.573', 'AMC 25.571', ...]
    """

def build_reference_index(chunks: List[dict]) -> dict:
    """
    Index inversé pour lookup O(1):
    {
        'cs_25_571': [chunk1, chunk2, ...],
        'amc_25_1309': [chunk3, ...],
        ...
    }
    """
```

### 7. Expansion de contexte (query-time)

```python
def expand_chunk_context(
    chunk: dict,
    all_chunks: List[dict],
    reference_index: dict = None,
    include_neighbors: bool = True,
    include_references: bool = True
) -> dict:
    """
    Enrichit un chunk avec contexte additionnel.

    Ajoute:
    - chunk['context_before']: chunk précédent (même fichier)
    - chunk['context_after']: chunk suivant (même fichier)
    - chunk['referenced_sections']: chunks des sections référencées
    """

def get_neighboring_chunks(
    chunk: dict,
    all_chunks: List[dict],
    before: int = 1,
    after: int = 1
) -> Tuple[List[dict], List[dict]]:
    """
    Retourne les chunks voisins du même fichier source.
    """

def expand_search_results(
    results: List[dict],
    all_chunks: List[dict],
    reference_index: dict = None
) -> List[dict]:
    """
    Enrichit tous les résultats de recherche avec contexte.
    """
```

---

## Parsers de documents

### Parser PDF (`pdf_processing.py`)

**Architecture à triple fallback avec extraction de tableaux :**

```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extraction robuste avec détection de tableaux.
    Ordre: pdfplumber → pdfminer.six → PyMuPDF

    Les tableaux sont formatés en markdown avec alignement.
    """

def _extract_with_pdfplumber(path: str) -> str:
    """Extraction via pdfplumber (principal, avec tableaux)"""

def _extract_text_with_pdfminer(path: str) -> str:
    """Extraction via pdfminer.six (fallback 1)"""

def _extract_text_with_pymupdf(path: str) -> str:
    """Extraction via PyMuPDF (fallback 2)"""

def extract_attachments(pdf_path: str) -> List[dict]:
    """Extraction récursive des pièces jointes"""

def clean_filename(filename: str) -> str:
    """
    Nettoyage nom de fichier:
    - Suppression caractères surrogates
    - Préservation extension (.pdf, .docx, etc.)
    - Caractères invalides → underscore
    """

def detect_and_clean_surrogates(text: str) -> str:
    """
    Nettoyage Unicode:
    - Détection surrogates (0xD800-0xDFFF)
    - Essai encodages: UTF-8, UTF-16, Latin-1, ISO-8859-1, CP1252
    - Suppression caractères non-imprimables
    """
```

### Parser DOCX/DOC (`docx_processing.py`)

```python
def convert_doc_to_docx(doc_path: str) -> str:
    """
    Conversion .doc → .docx via Microsoft Word (Windows).
    Requiert pywin32 et Word installé.
    - Accepte toutes les révisions (track changes)
    - Supprime tous les commentaires
    - Original non modifié (fichier temp supprimé après)
    """

def docx_to_text(path: str) -> str:
    """Extraction texte complet (paragraphes joints par \n)"""

def extract_text_from_docx(path: str) -> str:
    """
    Extraction texte .docx ou .doc.
    Pour .doc: conversion via Word (accepte révisions, supprime commentaires).
    """

def extract_paragraphs_from_docx(path: str) -> List[str]:
    """Liste des paragraphes individuels"""

def extract_sections_from_docx(path: str) -> List[dict]:
    """
    Découpage par headers (Heading 1/2, Titre 1/2).
    Retourne: [{'title': '...', 'content': '...', 'level': 1}, ...]
    """

def extract_text_from_tables(path: str) -> str:
    """Extraction du contenu des tableaux"""

def normalize_whitespace(text: str) -> str:
    """Normalisation espaces (préserve sauts de ligne)"""
```

### Parser XML EASA (`xml_processing.py`)

```python
from enum import Enum

class SectionPattern(Enum):
    CS_STANDARD = r"CS[-\s]?25[.\s]?\d+"
    AMC = r"AMC[-\s]?25[.\s]?\d+"
    GM = r"GM[-\s]?25[.\s]?\d+"
    CS_E = r"CS[-\s]?E[-\s]?\d+"
    CS_APU = r"CS[-\s]?APU[-\s]?\d+"
    ALL_EASA = r"(CS|AMC|GM)[-\s]?(25|E|APU)[-\s.]?\d+"
    CUSTOM = "custom"

@dataclass
class XMLParseConfig:
    pattern_type: SectionPattern = SectionPattern.ALL_EASA
    custom_pattern: str = None
    include_section_title: bool = True
    min_section_length: int = 50
    excluded_tags: List[str] = field(default_factory=lambda: ['note'])

def parse_xml_document(
    path: str,
    config: XMLParseConfig = None
) -> List[dict]:
    """
    Parse XML avec config.
    Retourne sections: [{'section_id': '...', 'content': '...', ...}, ...]
    """

def extract_text_recursive(element, excluded_tags: List[str] = None) -> str:
    """Extraction texte récursive avec gestion namespaces"""
```

### API Confluence (`confluence_processing.py`)

```python
def test_confluence_connection(
    base_url: str,
    username: str,
    password: str,
) -> Dict[str, Any]:
    """
    Teste la connexion à Confluence.
    Retourne: {'success': bool, 'message': str, 'user_info': dict}
    """

def list_spaces(
    base_url: str,
    username: str,
    password: str,
) -> List[Dict[str, str]]:
    """
    Liste tous les espaces accessibles.
    Retourne: [{'key': 'PROJ', 'name': '...', 'type': '...'}, ...]
    """

def get_space_pages(
    base_url: str,
    space_key: str,
    username: str,
    password: str,
    progress_cb: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Récupère toutes les pages d'un espace avec pagination.
    Retourne: [{'id': '...', 'title': '...', 'url': '...', 'html_content': '...'}, ...]
    """

def extract_text_from_confluence_space(
    base_url: str,
    space_key: str,
    username: str,
    password: str,
    progress_cb: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Pipeline complet: extraction pages + conversion HTML → texte.
    Retourne: [{'id': '...', 'title': '...', 'url': '...', 'text': '...'}, ...]
    """
```

**Fonctionnalités :**
- Support Confluence Cloud (atlassian.net) et Server
- Pagination automatique (25 pages/requête)
- Conversion HTML → texte avec BeautifulSoup
- Préservation structure (headers, listes, tableaux)
- URL de page comme chemin logique

**Architecture de connexion :**
```
┌─────────────────────────────┐
│  Détection type Confluence  │
│  (Cloud vs Server)          │
└──────────────┬──────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────────┐    ┌─────────────┐
│   Cloud     │    │   Server    │
│ /wiki/rest/ │    │ /rest/api   │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
┌─────────────────────────────┐
│  API REST Confluence        │
│  (Basic Auth)               │
└─────────────────────────────┘
```

---

## Stockage vectoriel FAISS

### Architecture (`faiss_store.py`)

#### Cache local pour performances réseau

```python
class LocalCacheManager:
    """
    Gestionnaire de cache local pour les bases FAISS réseau.

    Permet de copier les bases FAISS en local pour des performances
    de lecture optimales, tout en validant la fraîcheur du cache.

    Structure cache:
    ~/.cache/ragme_up/
    └── [hash_du_chemin]/
        ├── index.faiss      # Copie locale de l'index
        ├── metadata.json    # Copie locale des métadonnées
        └── .hash            # Hash de validation (taille + mtime)
    """

    def __init__(self, cache_dir: str = None):
        """Par défaut: ~/.cache/ragme_up/"""

    def is_cached(self, network_path: str) -> bool:
        """Vérifie si une collection est en cache"""

    def is_cache_valid(self, network_path: str) -> bool:
        """
        Compare le hash local avec le hash réseau.
        Hash = f"{size}_{mtime}" de index.faiss
        Retourne False si le cache est obsolète.
        """

    def cache_collection(self, network_path: str) -> str:
        """
        Copie une collection réseau en local.
        Retourne le chemin local.
        """

    def get_local_path(self, network_path: str) -> str:
        """Retourne le chemin du cache local"""

    def invalidate_cache(self, network_path: str):
        """Supprime le cache (appelé après ingestion)"""

    def get_cache_info(self, network_path: str) -> dict:
        """
        Retourne: {
            'is_cached': bool,
            'is_valid': bool,
            'local_path': str,
            'cache_size': int
        }
        """

# Singleton global
_cache_manager = None

def get_cache_manager() -> LocalCacheManager:
    """Retourne l'instance singleton du cache manager"""
```

#### Utilisation automatique dans FaissCollection

```python
class FaissCollection:
    def __init__(self, collection_path, name, dimension=1024,
                 use_local_cache=False, lazy_load=True):
        """
        Paramètres:
        - use_local_cache: Active l'utilisation du cache local
        - lazy_load: Diffère le chargement de l'index FAISS

        Attributs ajoutés:
        - self.cache_outdated: True si le cache existe mais est obsolète
        - self.using_cache: True si le cache est utilisé pour cette instance
        """

        if use_local_cache:
            cache_mgr = get_cache_manager()
            if cache_mgr.is_cached(collection_path):
                if cache_mgr.is_cache_valid(collection_path):
                    # Cache valide → utiliser le cache
                    self.collection_path = cache_mgr.get_local_path(collection_path)
                    self.using_cache = True
                else:
                    # Cache obsolète → utiliser le réseau + avertir
                    self.collection_path = collection_path  # réseau
                    self.cache_outdated = True
```

#### Invalidation automatique après ingestion

```python
# Dans rag_ingestion.py
def ingest_documents(...):
    # ... ingestion ...

    # Invalider le cache après ingestion
    cache_mgr = get_cache_manager()
    cache_mgr.invalidate_cache(collection_path)
```

#### Cache Streamlit (TTLs)

| Fonction | TTL | Description |
|----------|-----|-------------|
| `list_bases()` | 5 min | Liste des bases FAISS |
| `list_collections_for_base()` | 5 min | Collections d'une base |
| `get_collection_doc_counts()` | 5 min | Compteurs de documents |
| `get_cached_faiss_store()` | 10 min | Instances FaissStore |
| `cached_rag_query()` | 30 min | Résultats de requêtes RAG |

---

```python
class FAISSStore:
    """
    Gestionnaire de stockage FAISS.

    Structure fichiers:
    base_path/
    └── collection_name/
        ├── index.faiss      # Index vectoriel FAISS
        └── metadata.json    # Métadonnées chunks
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.dimension = 1024  # Snowflake Arctic

    def add_documents(
        self,
        collection: str,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[dict]
    ) -> List[str]:
        """Ajoute documents et retourne IDs"""

    def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Tuple[dict, float]]:
        """Recherche k plus proches voisins"""

    def delete_collection(self, collection: str) -> bool:
        """Supprime une collection"""

    def get_collection_stats(self, collection: str) -> dict:
        """Statistiques: count, dimension, etc."""
```

### Format des métadonnées

```json
{
  "ids": ["chunk_id_1", "chunk_id_2", ...],
  "documents": [
    {
      "id": "CS 25.571__chunk_0__abc123",
      "text": "[CS 25.571 - Damage tolerance...]\n...",
      "source_file": "CS-25.pdf",
      "path": "/path/to/CS-25.pdf",
      "chunk_id": "CS 25.571__chunk_0",
      "section_id": "CS 25.571",
      "section_kind": "CS",
      "section_title": "Damage tolerance and fatigue evaluation",
      "language": "en",
      "keywords": ["fatigue", "damage", "tolerance", "structure"],
      "key_phrases": ["shall be evaluated by analysis..."],
      "density_type": "dense",
      "density_score": 0.62,
      "references_to": ["CS 25.573", "AMC 25.571"],
      "is_complete_section": true
    },
    ...
  ]
}
```

---

## Pipeline de requête

### Flux de recherche (`rag_query.py`)

```
Question utilisateur
        │
        ▼
┌───────────────────────────────┐
│  1. EXPANSION DE REQUÊTE      │
│  (optionnel, via DALLEM API)  │
│  - Synonymes                  │
│  - Reformulations             │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  2. EMBEDDING                 │
│  Snowflake Arctic API         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  3. RECHERCHE FAISS           │
│  k=10 plus proches voisins    │
│  Distance L2                  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4. EXPANSION CONTEXTE        │
│  - Chunks voisins             │
│  - Sections référencées       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  5. RE-RANKING (optionnel)    │
│  - Feedbacks utilisateurs     │
│  - BGE Reranker API           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  6. GÉNÉRATION RÉPONSE        │
│  DALLEM API                   │
│  Prompt + contexte            │
└───────────────────────────────┘
```

### Fonction principale

```python
def query_rag(
    question: str,
    db_path: str,
    collection: str,
    k: int = 10,
    use_reranking: bool = False,
    use_feedback: bool = False,
    expand_context: bool = True
) -> dict:
    """
    Retourne:
    {
        'answer': 'Réponse générée...',
        'sources': [
            {
                'source_file': 'CS-25.pdf',
                'chunk_id': 'CS 25.571__chunk_0',
                'score': 0.85,
                'distance': 0.234,
                'text': '...',
                'section_id': 'CS 25.571',
                'keywords': [...],
                'references_to': [...]
            },
            ...
        ],
        'context_used': '...',  # Debug
        'expanded_context': [...] # Chunks voisins/référencés
    }
    """
```

---

## Améliorations Qualité RAG (v2.0)

### Vue d'ensemble des améliorations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE RAG v2.0                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 1 - RETRIEVAL AMÉLIORÉ                                                │
│  ├── HyDE (Hypothetical Document Embeddings)                                 │
│  ├── Lost in Middle (réordonnancement optimal)                               │
│  └── Détection qualité OCR                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2 - RECHERCHE AVANCÉE                                                 │
│  ├── Hybrid Search (BM25 + Dense + RRF)                                      │
│  ├── Semantic Cache (cache intelligent des réponses)                         │
│  └── RAGAS Metrics (évaluation qualité)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3 - QUALITÉ RÉPONSES                                                  │
│  ├── Answer Grounding (détection hallucinations)                             │
│  ├── Query Understanding (analyse d'intention)                               │
│  └── Semantic Chunking (découpage intelligent)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HyDE - Hypothetical Document Embeddings (`rag_query.py`)

Génère une réponse hypothétique pour enrichir l'embedding de la requête.

```python
def generate_hypothetical_document(question: str, http_client, log=None) -> str:
    """
    Génère un document hypothétique répondant à la question.

    Processus:
    1. LLM génère une réponse hypothétique (style document technique)
    2. Cette réponse est embedée avec la question originale
    3. L'embedding combiné capture mieux l'intention

    Avantage: Améliore le recall de 15-25% sur requêtes complexes
    """
```

### Lost in Middle (`rag_query.py`)

Réordonne les chunks pour placer les plus pertinents en début et fin de contexte.

```python
def reorder_lost_in_middle(chunks: List[dict]) -> List[dict]:
    """
    Réordonne les chunks selon le pattern "Lost in Middle".

    Problème: Les LLM ont tendance à ignorer le contenu au milieu
    Solution: Placer les meilleurs chunks en position 1, 3, 5... et fin

    Pattern résultant:
    [best, 3rd, 5th, ...middle..., 6th, 4th, 2nd]

    Impact: +10-15% de précision sur les réponses longues
    """
```

### Hybrid Search (`hybrid_search.py`)

Combine recherche lexicale BM25 et recherche dense avec fusion RRF.

```python
class HybridSearcher:
    """
    Recherche hybride BM25 + Dense avec Reciprocal Rank Fusion.

    Architecture:
    ┌─────────────┐     ┌─────────────┐
    │   Query     │────►│   BM25      │────► Scores lexicaux
    └─────────────┘     └─────────────┘
           │                  │
           │            ┌─────┴─────┐
           │            │    RRF    │────► Scores fusionnés
           │            └─────┬─────┘
           │                  │
           ▼            ┌─────┴─────┐
    ┌─────────────┐     │           │
    │   Dense     │────►│   FAISS   │────► Scores sémantiques
    └─────────────┘     └───────────┘

    Activation automatique: > 10000 chunks dans la collection
    """

    def __init__(self, chunks: List[dict]):
        """Initialise BM25 sur le corpus"""
        self.bm25 = BM25Okapi([self._tokenize(c['text']) for c in chunks])

    def search(self, query: str, dense_results: List, k: int = 10) -> List[dict]:
        """
        Recherche hybride avec fusion RRF.

        Paramètres:
        - query: Question utilisateur
        - dense_results: Résultats FAISS (déjà calculés)
        - k: Nombre de résultats finaux

        Retourne: Liste triée par score RRF
        """

    def _reciprocal_rank_fusion(self, bm25_ranks, dense_ranks, k=60) -> List[float]:
        """
        RRF = Σ 1/(k + rank_i) pour chaque système

        k=60 est le paramètre standard (favorise les top résultats)
        """
```

### Semantic Cache (`semantic_cache.py`)

Cache intelligent des réponses basé sur similarité sémantique.

```python
class SemanticCache:
    """
    Cache sémantique pour éviter les appels LLM répétitifs.

    Fonctionnement:
    1. Question → embedding
    2. Recherche dans le cache (similarité cosinus)
    3. Si similarité > seuil (0.92) → retourne réponse cachée
    4. Sinon → appel LLM → stocke en cache

    Structure cache:
    {
        "embedding": [float...],  # Embedding de la question
        "question": "...",         # Question originale
        "answer": "...",           # Réponse générée
        "sources": [...],          # Sources utilisées
        "timestamp": "...",        # Date de création
        "collection": "..."        # Collection concernée
    }

    Économie: Jusqu'à 70% des appels LLM en production
    """

    def __init__(self, cache_dir: str, similarity_threshold: float = 0.92):
        """
        Paramètres:
        - cache_dir: Répertoire de stockage (JSON par collection)
        - similarity_threshold: Seuil de similarité (0.92 = très similaire)
        """

    def get(self, question: str, collection: str) -> Optional[dict]:
        """Recherche dans le cache, retourne None si miss"""

    def set(self, question: str, answer: str, sources: List, collection: str):
        """Ajoute une entrée au cache"""

    def invalidate(self, collection: str):
        """Invalide le cache d'une collection (après ingestion)"""
```

### RAGAS Metrics (`rag_metrics.py`)

Métriques d'évaluation de qualité RAG inspirées du framework RAGAS.

```python
class RAGMetrics:
    """
    Métriques de qualité RAG (sans appels API supplémentaires).

    Métriques implémentées:
    ┌────────────────────┬──────────────────────────────────────┐
    │ Métrique           │ Description                           │
    ├────────────────────┼──────────────────────────────────────┤
    │ context_relevancy  │ % chunks pertinents pour la question  │
    │ answer_relevancy   │ Similarité réponse ↔ question         │
    │ faithfulness       │ % claims vérifiables dans le contexte │
    │ context_precision  │ Position des chunks pertinents        │
    └────────────────────┴──────────────────────────────────────┘
    """

    def compute_context_relevancy(self, question: str, chunks: List[dict]) -> float:
        """
        Mesure la pertinence des chunks récupérés.

        Méthode: Overlap tokens (question ∩ chunk) / tokens(question)
        Score: 0.0 (non pertinent) → 1.0 (très pertinent)
        """

    def compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Mesure si la réponse est fidèle au contexte.

        Méthode:
        1. Extraction claims de la réponse (phrases avec faits)
        2. Vérification présence dans le contexte
        3. Score = claims_vérifiés / total_claims
        """

    def compute_all(self, question, answer, chunks) -> dict:
        """Retourne toutes les métriques en un appel"""
```

### Answer Grounding (`answer_grounding.py`)

Détection des hallucinations sans appels API supplémentaires.

```python
class AnswerGrounder:
    """
    Détection d'hallucinations par analyse de grounding.

    Principe: Vérifier que chaque claim de la réponse
    est ancré dans le contexte fourni.

    Types de vérification:
    ┌─────────────────────┬────────────────────────────────────┐
    │ Type                │ Méthode                            │
    ├─────────────────────┼────────────────────────────────────┤
    │ Token overlap       │ Mots partagés réponse/contexte     │
    │ Numeric claims      │ Valeurs numériques vérifiables     │
    │ Reference claims    │ Citations CS/AMC/GM vérifiables    │
    │ Definition claims   │ "X means Y" présent dans contexte  │
    └─────────────────────┴────────────────────────────────────┘
    """

    def __init__(self, context: str, sources: List[dict]):
        """Initialise avec le contexte de la réponse"""

    def analyze(self, answer: str) -> GroundingResult:
        """
        Analyse complète de grounding.

        Retourne:
        {
            'grounding_score': 0.85,      # Score global (0-1)
            'risk_level': 'low',          # low/medium/high
            'verified_claims': [...],     # Claims vérifiés
            'unverified_claims': [...],   # Claims non trouvés
            'suspicious_phrases': [...],  # Phrases à risque
            'recommendations': [...]      # Suggestions
        }
        """

    def extract_claims(self, text: str) -> List[Claim]:
        """Extrait les claims vérifiables du texte"""

    def verify_claim(self, claim: Claim) -> bool:
        """Vérifie si un claim est présent dans le contexte"""
```

### Query Understanding (`query_understanding.py`)

Analyse d'intention pour adapter le comportement du RAG.

```python
class QueryAnalyzer:
    """
    Analyse l'intention et la complexité des requêtes.

    Intents détectés:
    ┌─────────────────┬───────────────────────────────────────┐
    │ Intent          │ Pattern                               │
    ├─────────────────┼───────────────────────────────────────┤
    │ definition      │ "what is", "define", "qu'est-ce que"  │
    │ procedure       │ "how to", "comment", "steps to"       │
    │ requirement     │ "must", "shall", "requirement"        │
    │ comparison      │ "difference between", "vs", "compare" │
    │ troubleshooting │ "error", "issue", "problem", "fix"    │
    │ reference       │ "CS 25.", "AMC", "GM", "FAR"          │
    └─────────────────┴───────────────────────────────────────┘

    Adaptation du RAG selon l'intent:
    - definition: top_k=5, pas de HyDE
    - procedure: top_k=10, avec contexte étendu
    - requirement: top_k=15, sections EASA prioritaires
    - comparison: top_k=20, multi-source
    """

    def analyze(self, question: str) -> QueryAnalysis:
        """
        Analyse complète de la requête.

        Retourne:
        {
            'intent': 'requirement',
            'complexity': 'medium',      # simple/medium/complex
            'entities': ['CS 25.571'],   # Entités détectées
            'suggested_top_k': 15,       # top_k recommandé
            'use_hyde': True,            # Utiliser HyDE?
            'expand_context': True       # Étendre le contexte?
        }
        """

    def get_adaptive_top_k(self, analysis: QueryAnalysis) -> int:
        """Calcule le top_k optimal selon la complexité"""
```

### Semantic Chunking (`semantic_chunking.py`)

Découpage intelligent basé sur les frontières sémantiques.

```python
class SemanticChunker:
    """
    Chunking basé sur la détection de frontières sémantiques.

    Vs chunking classique (taille fixe):
    - Respecte les sections logiques
    - Détecte les changements de sujet
    - Préserve l'intégrité des paragraphes

    Signaux de frontière:
    ┌─────────────────────┬────────────────────────────────────┐
    │ Signal              │ Poids                              │
    ├─────────────────────┼────────────────────────────────────┤
    │ Section EASA        │ 1.0 (frontière forte)              │
    │ Header numéroté     │ 0.9                                │
    │ Double saut ligne   │ 0.7                                │
    │ Changement sujet    │ 0.6 (via embeddings)               │
    │ Fin de liste        │ 0.5                                │
    │ Référence nouvelle  │ 0.4                                │
    └─────────────────────┴────────────────────────────────────┘
    """

    def chunk(self, text: str, source_file: str) -> List[dict]:
        """
        Découpe le texte en chunks sémantiques.

        Processus:
        1. Détection des frontières potentielles
        2. Scoring de chaque frontière
        3. Sélection des meilleures frontières
        4. Découpe et enrichissement métadonnées
        """

    def detect_boundaries(self, text: str) -> List[Boundary]:
        """Détecte toutes les frontières potentielles"""

    def score_boundary(self, boundary: Boundary) -> float:
        """Score de 0 à 1 pour la force de la frontière"""
```

---

## LLM Vision OCR (`llm_ocr.py`)

### Architecture OCR

```
PDF scanné
    │
    ▼
┌─────────────────────────────────────────┐
│  1. DÉTECTION ORIENTATION (LOCAL)       │
│  detect_page_orientation()              │
│  - Analyse métadonnées PDF              │
│  - Direction du texte extractible       │
│  - Retourne: 0°, 90°, 180°, 270°        │
└────────────────┬────────────────────────┘
                 │
                 ▼ (si détection incertaine)
┌─────────────────────────────────────────┐
│  2. DÉTECTION ORIENTATION (LLM)         │
│  detect_orientation_with_llm()          │
│  - Envoie image au LLM Vision           │
│  - Prompt: "Is this page rotated?"      │
│  - Retourne: angle de rotation          │
│  ⚠️ 1 appel API par page               │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. CONVERSION PDF → IMAGE              │
│  pdf_page_to_image()                    │
│  - PyMuPDF (fitz) pour le rendu         │
│  - Rotation automatique si demandée     │
│  - DPI configurable (150 par défaut)    │
│  - Compression PNG optimisée            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. OCR VIA LLM VISION                  │
│  ocr_page_with_llm()                    │
│  - Image encodée en base64              │
│  - Prompt OCR spécialisé                │
│  - DALLEM Vision API                    │
│  ⚠️ 1 appel API par page               │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  5. POST-TRAITEMENT                     │
│  - Nettoyage caractères spéciaux        │
│  - Détection qualité extraction         │
│  - Structuration texte                  │
└─────────────────────────────────────────┘
```

### Fonctions principales

```python
def detect_page_orientation(pdf_path: str, page_number: int, log=None) -> int:
    """
    Détection locale de l'orientation (gratuit, pas d'API).

    Méthodes:
    1. Métadonnées PDF (rotation déclarée)
    2. Analyse direction texte extractible
    3. Heuristiques sur les caractères

    Retourne: Angle de rotation à appliquer (0, 90, 180, 270)
    """

def detect_orientation_with_llm(
    image_bytes: bytes,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    log=None
) -> int:
    """
    Détection orientation via LLM Vision (1 appel API).

    Utilisé quand:
    - La détection locale échoue
    - PDF entièrement scanné (pas de texte extractible)
    - Option smart_rotate=True activée

    Prompt: "Analyze this document image. Is it correctly oriented?
             If rotated, specify the angle (90, 180, 270)."
    """

def pdf_page_to_image(
    pdf_path: str,
    page_number: int,
    dpi: int = 150,
    max_size: int = 4096,
    auto_rotate: bool = True,
    force_rotation: int = None,
    log = None
) -> Optional[bytes]:
    """
    Convertit une page PDF en image PNG.

    Paramètres:
    - dpi: Résolution (150 = bon compromis qualité/taille)
    - max_size: Taille max en pixels (limite LLM Vision)
    - auto_rotate: Applique rotation détectée localement
    - force_rotation: Force un angle spécifique (0/90/180/270)

    Retourne: bytes PNG de l'image
    """

def ocr_page_with_llm(
    image_bytes: bytes,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    log = None
) -> str:
    """
    OCR d'une image via LLM Vision.

    Prompt spécialisé:
    "Extract ALL text from this document image exactly as written.
     Preserve structure, paragraphs, lists, and tables.
     Output only the extracted text, no commentary."

    Retourne: Texte extrait
    """

# Fonction de commodité DALLEM
def ocr_pdf_with_dallem(
    pdf_path: str,
    pages: List[int] = None,
    dpi: int = 150,
    auto_rotate: bool = True,
    smart_rotate: bool = True,
    progress_cb: Callable = None,
    log = None
) -> DocumentOCRResult:
    """
    OCR complet d'un PDF via DALLEM.

    Paramètres:
    - pages: Pages spécifiques (None = toutes)
    - auto_rotate: Rotation locale automatique
    - smart_rotate: Utilise LLM pour orientation si incertain
    - progress_cb: Callback progression (page_num, total)

    Retourne: DocumentOCRResult avec texte par page

    ⚠️ Coût API: 1-2 appels par page (OCR + rotation si smart)
    """
```

### Intégration avec le pipeline d'ingestion

```python
# Dans ingestion_pipeline.py
def extract_text_local(self, file_info: FileInfo) -> ExtractionResult:
    """
    Extrait le texte avec fallback OCR intelligent.

    Stratégie:
    1. Extraction texte classique (pdfplumber/pdfminer)
    2. Si échec ou texte pauvre → OCR Vision
    3. Détection automatique des PDFs scannés
    """

    # Tentative extraction classique
    text = extract_text_from_pdf(file_info.local_path)

    # Détection qualité
    if self._is_poor_extraction(text):
        # Fallback OCR
        result = ocr_pdf_with_dallem(
            file_info.local_path,
            auto_rotate=True,
            smart_rotate=True
        )
        text = result.full_text

    return ExtractionResult(text=text, method='ocr' if used_ocr else 'text')
```

---

## Pipeline d'ingestion optimisé (`ingestion_pipeline.py`)

### Architecture 2 phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 - TRAITEMENT LOCAL                                │
│                    (Minimise les appels réseau)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │ Download    │────►│ Extract     │────►│ Parse       │────►│ Chunk     │ │
│  │ to TEMP     │     │ attachments │     │ text        │     │ documents │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│        │                   │                   │                    │       │
│        ▼                   ▼                   ▼                    ▼       │
│   1 appel/fichier    0 appels            0-N appels*          0 appels     │
│                                          (*OCR si scanné)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 - OPÉRATIONS RÉSEAU                               │
│                    (Batch optimisé)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         BATCH EMBEDDINGS                                 ││
│  │  - Tous les chunks en une fois                                          ││
│  │  - Batch size: 32 textes                                                ││
│  │  - 1 appel API / 32 chunks                                              ││
│  └────────────────────────────────────┬────────────────────────────────────┘│
│                                       │                                      │
│                                       ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         INSERT FAISS                                     ││
│  │  - Écriture locale (ou réseau)                                          ││
│  │  - Batch de 4000 chunks                                                 ││
│  │  - Auto-save après chaque batch                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Classe principale

```python
class OptimizedIngestionPipeline:
    """
    Pipeline d'ingestion optimisé pour minimiser les appels réseau.

    Avantages vs pipeline classique:
    - 70% moins d'appels réseau (batch embeddings)
    - Traitement local parallélisé
    - Gestion intelligente des fichiers temporaires
    - Support OCR intégré avec fallback
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        temp_dir: str = None,
        chunk_size: int = 1000,
        use_semantic_chunking: bool = False,
        log = None
    ):
        """
        Paramètres:
        - db_path: Chemin base FAISS
        - collection_name: Nom de la collection
        - temp_dir: Répertoire temporaire (défaut: système)
        - chunk_size: Taille de base des chunks
        - use_semantic_chunking: Utiliser le chunking sémantique
        """

    def ingest_files(
        self,
        file_paths: List[str],
        rebuild: bool = False,
        progress_cb: Callable = None
    ) -> IngestionResult:
        """
        Ingestion complète avec pipeline optimisé.

        Étapes:
        1. download_to_temp() - Copie locale
        2. extract_attachments() - Pièces jointes
        3. extract_text_local() - Parsing + OCR
        4. chunk_documents() - Découpage
        5. batch_embed_and_insert() - Embeddings + FAISS

        Retourne:
        {
            'ingested': [...],    # Fichiers traités
            'skipped': [...],     # Déjà présents
            'errors': [...],      # Erreurs
            'chunks_created': N,  # Nombre de chunks
            'api_calls': M,       # Appels API effectués
            'duration': X.Xs      # Temps total
        }
        """

    def download_to_temp(self, file_paths: List[str]) -> List[FileInfo]:
        """Télécharge les fichiers en local (réseau → TEMP)"""

    def extract_attachments(self, files: List[FileInfo]) -> List[FileInfo]:
        """Extrait les pièces jointes des PDFs"""

    def extract_text_local(self, files: List[FileInfo]) -> List[ExtractionResult]:
        """Parse le texte localement avec fallback OCR"""

    def chunk_documents(self, extractions: List[ExtractionResult]) -> List[dict]:
        """Découpe en chunks (adaptatif ou sémantique)"""

    def batch_embed_and_insert(
        self,
        chunks: List[dict],
        rebuild: bool
    ) -> int:
        """Embeddings batch + insertion FAISS"""

# Fonctions de commodité
def quick_ingest(file_paths: List[str], db_path: str, collection: str) -> dict:
    """Ingestion rapide en une ligne"""

def ingest_csv(csv_path: str, db_path: str) -> dict:
    """Ingestion depuis un fichier CSV"""
```

---

## Configuration

### Fichier config.json

```json
{
  "base_root_dir": "C:\\Data\\FAISS_DATABASE\\BaseDB",
  "csv_import_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Ingestion",
  "csv_export_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Tracking",
  "feedback_dir": "C:\\Data\\FAISS_DATABASE\\Feedbacks"
}
```

### Gestionnaire de configuration (`config_manager.py`)

```python
@dataclass
class StorageConfig:
    base_root_dir: str
    csv_import_dir: str
    csv_export_dir: str
    feedback_dir: str

def load_config() -> StorageConfig:
    """Charge config.json ou retourne valeurs par défaut"""

def save_config(config: StorageConfig) -> bool:
    """Sauvegarde dans config.json"""

def validate_directory(path: str) -> Tuple[bool, str]:
    """Vérifie existence et permissions écriture"""

def validate_all_directories(config: StorageConfig) -> dict:
    """Valide tous les chemins, retourne erreurs"""
```

---

## API et modèles

Le système utilise exclusivement des APIs externes pour les modèles d'IA.

### Embeddings (`models_utils.py`)

```python
# Configuration
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 32  # Équilibre performance/sécurité
MAX_CHARS_PER_TEXT = 28000  # Limite ~7000 tokens (Snowflake max: 8192)

# Snowflake Arctic (API)
class DirectOpenAIEmbeddings:
    """Client embeddings via API Snowflake (1024 dimensions)"""

def embed_in_batches(texts, role, batch_size, emb_client, log) -> np.ndarray:
    """
    Embedding par batch avec troncature automatique.
    Les textes > 28000 chars sont tronqués pour éviter les erreurs de tokens.
    """
```

### LLM pour génération

```python
# DALLEM (API)
LLM_MODEL = "dallem-val"

def call_dallem_chat(http_client, question, context, log) -> str:
    """Génération via API DALLEM avec retry automatique"""
```

### Reranker

```python
# BGE Reranker (API)
BGE_RERANKER_API_BASE = "https://api.dev.dassault-aviation.pro/bge-reranker-v2-m3"
BGE_RERANKER_ENDPOINT = "/v1/rerank"

def rerank_with_bge(query: str, documents: List[str], top_k: int) -> List[dict]:
    """Re-ranking via API BGE Reranker"""
```

---

## Dépendances

### requirements.txt

```
# Interface
streamlit>=1.28.0
customtkinter>=5.2.0

# FAISS
faiss-cpu>=1.7.4

# Parsing PDF (avec extraction tableaux)
pdfminer.six>=20221105
pymupdf>=1.24.0
pdfplumber>=0.10.0  # Extraction améliorée des tableaux
python-docx>=0.8.11
pywin32>=306       # Conversion .doc → .docx via Word (Windows)

# Traitement texte
langdetect>=1.0.9
chardet>=5.1.0
unidecode>=1.3.6
beautifulsoup4>=4.12.0  # Parsing HTML (Confluence)

# API (Snowflake, DALLEM, BGE Reranker)
openai>=1.0.0
httpx>=0.24.0
requests>=2.31.0

# Core
numpy>=1.24.0
packaging>=23.0
```

### Bibliothèques natives Python utilisées

- `xml.etree.ElementTree` - Parsing XML
- `pathlib.Path` - Manipulation chemins
- `re` - Expressions régulières
- `json` - Sérialisation
- `concurrent.futures` - Parallélisation
- `dataclasses` - Classes de données
- `enum` - Énumérations
- `typing` - Annotations de type

---

## Logs et debug

### Fichier de logs

```
rag_da_debug.log
```

Contient :
- Extraction PDF (succès/fallback/erreurs)
- Traitement Unicode/surrogates
- Chunking (nombre chunks, densité détectée)
- Ingestion FAISS (batches, IDs)
- Requêtes (résultats, scores)
- Erreurs réseau

### Activation debug verbose

Dans `streamlit_RAG.py`, modifier le niveau de logging :

```python
logging.basicConfig(
    level=logging.DEBUG,  # INFO par défaut
    filename='rag_da_debug.log'
)
```

---

**Auteur** : Renaud LOISON

**Version:** 2.0
**Dernière mise à jour:** 2025-11-29
