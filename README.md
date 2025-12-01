# RaGME_UP - PROP

Systeme RAG (Retrieval-Augmented Generation) pour l'indexation et l'interrogation de documents techniques avec FAISS, Snowflake Arctic Embeddings et DALLEM.

---

## Structure du Projet

```
RAG_v7/
|-- core/                    # Infrastructure de base
|   |-- models_utils.py      # Clients API, embeddings Snowflake
|   |-- faiss_store.py       # Stockage vectoriel FAISS
|   |-- semantic_cache.py    # Cache semantique requetes
|   |-- config_manager.py    # Gestion configuration
|
|-- processing/              # Extraction de texte
|   |-- pdf_processing.py    # PDF avec OCR fallback
|   |-- docx_processing.py   # Documents Word
|   |-- pptx_processing.py   # PowerPoint + pieces jointes
|   |-- xml_processing.py    # XML (patterns EASA)
|   |-- csv_processing.py    # Fichiers CSV
|   |-- confluence_processing.py  # API Confluence
|
|-- chunking/                # Decoupage documents
|   |-- chunking.py          # Chunking generique + EASA
|   |-- semantic_chunking.py # Chunking semantique
|   |-- easa_sections.py     # Parser sections EASA
|
|-- search/                  # Recherche et ranking
|   |-- hybrid_search.py     # BM25 + dense (hybride)
|   |-- advanced_search.py   # Query expansion, BGE reranking
|   |-- multi_collection_search.py  # Recherche inter-bases (federe)
|   |-- query_understanding.py      # Analyse intention requete
|
|-- ingestion/               # Pipeline d'ingestion
|   |-- ingestion_pipeline.py  # Pipeline optimise
|   |-- rag_ingestion.py       # API legacy
|
|-- query/                   # Execution requetes RAG
|   |-- rag_query.py         # Fonctions principales RAG
|   |-- answer_grounding.py  # Detection hallucinations
|   |-- rag_metrics.py       # Metriques RAGAS
|
|-- feedback/                # Retours utilisateurs
|   |-- feedback_store.py    # Stockage feedbacks
|
|-- ui/                      # Interfaces utilisateur
|   |-- streamlit_RAG.py     # Application Streamlit
|   |-- csv_generator_gui.py # Generateur CSV (Tkinter)
|
|-- ocr/                     # OCR via LLM
|   |-- llm_ocr.py           # OCR DALLEM vision
|
|-- scripts/                 # Scripts de lancement
|   |-- install.bat          # Installation Windows
|   |-- launch.bat           # Lancement application
|
|-- docs/                    # Documentation
|   |-- ARCHITECTURE_TECHNIQUE.md
|   |-- GUIDE_UTILISATEUR.md
|   |-- INSTALLATION_RESEAU.md
|   |-- PIPELINE_ARCHITECTURE.md
|   |-- GUIDE_RAG.md
```

---

## Documentation

- **[Guide Utilisateur](docs/GUIDE_UTILISATEUR.md)** - Documentation complete pour utiliser l'application
- **[Installation Reseau](docs/INSTALLATION_RESEAU.md)** - Guide de deploiement multi-utilisateurs
- **[Architecture Technique](docs/ARCHITECTURE_TECHNIQUE.md)** - Documentation technique complete

---

## Demarrage rapide

### Installation

```bash
# Windows: double-cliquez sur
scripts/install.bat
```

### Lancement

```bash
# Windows: double-cliquez sur
scripts/launch.bat

# Ou manuellement
streamlit run ui/streamlit_RAG.py
```

L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`

---

## Fonctionnalites principales

- **Mode OFFLINE** : fonctionne sans connexion internet avec modeles locaux (RTX 4090 recommande)
- **Gestion CSV** avec interface GUI moderne
- **Ingestion documents** (PDF, DOCX, PPTX, XML, TXT) avec tracking automatique
- **Ingestion Confluence** : chargement d'espaces entiers via API *(admin)*
- **Mode EASA automatique** : active automatiquement pour la base CERTIFICATION
- **Recherche inter-bases** : requetes federees sur plusieurs collections
- **Questions RAG** avec recherche semantique et generation de reponses
- **Cache local** : copie locale des bases pour performances reseau optimales
- **Feedback utilisateur** : evaluation granulaire des reponses et sources
- **Re-ranking intelligent** : amelioration des resultats bases sur les feedbacks

---

## Modules par fonctionnalite

### Core (`core/`)

| Module | Description |
|--------|-------------|
| `models_utils.py` | Clients API Snowflake/DALLEM, embeddings, logging |
| `faiss_store.py` | Store FAISS avec lazy loading et cache local |
| `semantic_cache.py` | Cache semantique pour requetes similaires |
| `config_manager.py` | Gestion des chemins et configuration |
| `offline_models.py` | **NOUVEAU** - Modeles locaux (BGE-M3, Mistral-7B, BGE-Reranker) |

### Processing (`processing/`)

| Module | Description |
|--------|-------------|
| `pdf_processing.py` | Extraction PDF avec triple fallback + OCR |
| `pptx_processing.py` | **NOUVEAU** - PowerPoint avec pieces jointes et OCR |
| `docx_processing.py` | Documents Word |
| `xml_processing.py` | XML avec patterns EASA configurables |
| `confluence_processing.py` | API REST Confluence |

### Search (`search/`)

| Module | Description |
|--------|-------------|
| `hybrid_search.py` | Recherche hybride BM25 + dense |
| `advanced_search.py` | HyDE, Lost-in-Middle, BGE reranking |
| `multi_collection_search.py` | **NOUVEAU** - Recherche federee inter-bases avec RRF |
| `query_understanding.py` | Detection d'intention et complexite |

### Query (`query/`)

| Module | Description |
|--------|-------------|
| `rag_query.py` | Pipeline RAG complet |
| `answer_grounding.py` | Detection hallucinations |
| `rag_metrics.py` | Metriques RAGAS (faithfulness, relevance) |

---

## Ameliorations Qualite RAG

### Phase 1 - Retrieval ameliore

| Technique | Description | Module |
|-----------|-------------|--------|
| **HyDE** | Genere une reponse hypothetique pour enrichir la requete | `search/advanced_search.py` |
| **Lost in Middle** | Reordonne les chunks (meilleurs en debut/fin) | `search/advanced_search.py` |
| **OCR Quality Detection** | Evalue la qualite d'extraction PDF | `processing/pdf_processing.py` |

### Phase 2 - Recherche hybride & Cache

| Technique | Description | Module |
|-----------|-------------|--------|
| **Hybrid Search (BM25)** | Combine recherche dense + lexicale | `search/hybrid_search.py` |
| **Semantic Cache** | Cache reponses pour requetes similaires | `core/semantic_cache.py` |
| **RAGAS Metrics** | Metriques qualite (faithfulness, relevance) | `query/rag_metrics.py` |

### Phase 3 - Qualite des reponses

| Technique | Description | Module |
|-----------|-------------|--------|
| **Answer Grounding** | Detection d'hallucinations par verification des claims | `query/answer_grounding.py` |
| **Query Understanding** | Analyse d'intention (definition, procedure, requirement) | `search/query_understanding.py` |
| **Semantic Chunking** | Decoupe aux frontieres semantiques | `chunking/semantic_chunking.py` |

### Phase 4 - Multi-collection & PPTX

| Technique | Description | Module |
|-----------|-------------|--------|
| **Multi-Collection Search** | Recherche federee inter-bases avec fusion RRF | `search/multi_collection_search.py` |
| **PPTX Processing** | Extraction PowerPoint avec OCR images et pieces jointes | `processing/pptx_processing.py` |

---

## Utilisation des modules

### Pipeline d'ingestion

```python
from ingestion.ingestion_pipeline import quick_ingest

result = quick_ingest(
    file_paths=["doc1.pdf", "doc2.pptx"],
    db_path="/path/to/faiss",
    collection_name="my_docs",
    quality_threshold=0.5,
)
```

### Requete RAG simple

```python
from query.rag_query import run_rag_query

result = run_rag_query(
    db_path="/path/to/faiss",
    collection_name="my_docs",
    question="Quelles sont les exigences de certification?",
    use_hybrid_search=True,
    use_bge_reranker=True,
)
```

### Recherche inter-bases (federee)

```python
from query.rag_query import run_multi_collection_rag_query

result = run_multi_collection_rag_query(
    db_path="/path/to/faiss",
    collection_names=["collection_A", "collection_B", "collection_C"],
    question="Comment fonctionne le systeme hydraulique?",
    use_hybrid_search=True,
)
# Les sources incluent collection_name pour la provenance
```

### OCR LLM pour PDF scannes

```python
from ocr.llm_ocr import smart_ocr_with_dallem

result = smart_ocr_with_dallem(
    pdf_path="scan.pdf",
    quality_threshold=0.5,
    auto_rotate=True,
)
```

---

## Configuration

### Fichier de configuration

```json
{
  "base_root_dir": "C:\\Data\\FAISS_DATABASE\\BaseDB",
  "csv_import_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Ingestion",
  "csv_export_dir": "C:\\Data\\FAISS_DATABASE\\CSV_Tracking",
  "feedback_dir": "C:\\Data\\FAISS_DATABASE\\Feedbacks"
}
```

### Auto-configuration memoire

Le pipeline detecte automatiquement la RAM disponible et adapte sa configuration :

| RAM | Mode | Workers | Batch |
|-----|------|---------|-------|
| <=8 Go | Ultra-conservateur | 1 | 4 |
| 8-12 Go | Conservateur | 2 | 8 |
| 12-16 Go | Equilibre | 4 | 16 |
| 16-32 Go | Performance | 6 | 32 |
| 32+ Go | Maximum | 8 | 64 |

---

## Mode OFFLINE (v2.1)

Le mode offline permet d'utiliser le systeme RAG sans connexion internet, en utilisant des modeles IA locaux sur GPU.

### Modeles locaux utilises

| Modele | Fonction | VRAM |
|--------|----------|------|
| **BGE-M3** | Embeddings (1024 dim) | ~2 GB |
| **BGE-Reranker-v2-m3** | Re-ranking | ~2 GB |
| **Mistral-7B-Instruct-v0.3** | Generation reponses | ~8 GB |

### Pre-chargement au demarrage

Les modeles sont **pre-charges au demarrage** de l'application pour eviter les temps de chargement pendant les requetes :

```
============================================================
[PRELOAD] Pre-chargement des modeles offline...
============================================================
[PRELOAD] 1/3 Chargement BGE-M3 (embeddings)...
[PRELOAD] ✅ BGE-M3 charge en 3.2s
[PRELOAD] 2/3 Chargement BGE-Reranker...
[PRELOAD] ✅ BGE-Reranker charge en 2.1s
[PRELOAD] 3/3 Chargement Mistral-7B (LLM)...
[PRELOAD] ✅ Mistral-7B charge en 24.5s
============================================================
[PRELOAD] ✅ Tous les modeles charges (29.8s)
[PRELOAD] VRAM utilisee: 12.5/24.0 GB
============================================================
```

### Performance attendue (RTX 4090)

| Operation | Temps |
|-----------|-------|
| Recherche Hybrid (5000 chunks) | ~8s |
| BGE Reranker (30 docs) | ~3s |
| Generation LLM | ~7s |
| **Total par requete** | **~20s** |

### Activation

Dans la sidebar Streamlit, cochez **"🔌 Mode OFFLINE (modeles locaux)"**.

---

## Prerequis

### Mode ONLINE (par defaut)
- Python 3.8 ou superieur
- Windows 10/11 (ou Linux/macOS avec adaptations)
- Acces reseau aux APIs : Snowflake (embeddings), DALLEM (LLM), BGE Reranker

### Mode OFFLINE (sans connexion)
- GPU NVIDIA avec 16+ GB VRAM (RTX 4090 recommande)
- Modeles locaux telecharges :
  - **BGE-M3** : embeddings (1024 dimensions)
  - **Mistral-7B-Instruct-v0.3** : generation de reponses
  - **BGE-Reranker-v2-m3** : re-ranking des resultats
- ~20 GB d'espace disque pour les modeles

---

## Support

Consultez la documentation pour toute question :
- Questions d'utilisation -> [Guide Utilisateur](docs/GUIDE_UTILISATEUR.md)
- Installation reseau -> [Installation Reseau](docs/INSTALLATION_RESEAU.md)
- Developpement/maintenance -> [Architecture Technique](docs/ARCHITECTURE_TECHNIQUE.md)

**Auteur** : Renaud LOISON

---

**Version:** 2.1
**Derniere mise a jour:** 2025-12-01
