# Architecture du Pipeline RAG EASA

> **Version**: RAG v6 | **~20,000 lignes de code** | **24 modules Python**

---

## 1. Vue d'Ensemble Globale

```mermaid
flowchart TB
    subgraph USER["👤 INTERFACE UTILISATEUR"]
        UI[Streamlit Web App]
        CSV[CSV Generator GUI]
    end

    subgraph INGESTION["📥 PIPELINE D'INGESTION"]
        direction TB
        I1[Acquisition Fichiers]
        I2[Extraction Texte]
        I3[OCR LLM]
        I4[Chunking Intelligent]
        I5[Augmentation]
        I6[Embeddings]

        I1 --> I2
        I2 -->|Qualité < 50%| I3
        I2 -->|Qualité OK| I4
        I3 --> I4
        I4 --> I5
        I5 --> I6
    end

    subgraph STORAGE["💾 STOCKAGE"]
        FAISS[(FAISS Vector Store)]
        META[(Metadata JSON)]
        CACHE[(Cache Local)]
    end

    subgraph QUERY["🔍 PIPELINE DE REQUÊTE"]
        direction TB
        Q1[Compréhension Requête]
        Q2[Expansion HyDE]
        Q3[Recherche Hybride]
        Q4[Post-traitement]
        Q5[Génération LLM]
        Q6[Contrôle Qualité]

        Q1 --> Q2
        Q2 --> Q3
        Q3 --> Q4
        Q4 --> Q5
        Q5 --> Q6
    end

    subgraph FEEDBACK["📊 AMÉLIORATION CONTINUE"]
        FB[Feedback Store]
        STATS[Statistiques]
    end

    UI --> INGESTION
    UI --> QUERY
    CSV --> INGESTION

    I6 --> FAISS
    I6 --> META

    FAISS --> CACHE
    CACHE --> Q3
    META --> Q4

    Q6 --> UI
    UI --> FB
    FB --> Q4
    FB --> STATS
```

---

## 2. Pipeline d'Ingestion Détaillé

```mermaid
flowchart TB
    subgraph INPUT["📄 SOURCES"]
        PDF[PDF]
        DOCX[DOCX]
        XML[XML]
        URL[URL HTTP]
        CONF[Confluence]
    end

    subgraph PHASE1["⚡ PHASE 1 - LOCALE (0 appel réseau)"]
        direction TB

        subgraph DOWNLOAD["Téléchargement"]
            DL[Copie vers TEMP local]
        end

        subgraph EXTRACT["Extraction Texte"]
            direction LR
            E1[pdfplumber<br/>Tables + Texte]
            E2[pdfminer.six<br/>Fallback robuste]
            E3[PyMuPDF<br/>Fallback ultime]
            E1 -.->|échec| E2
            E2 -.->|échec| E3
        end

        subgraph QUALITY["Évaluation Qualité"]
            QA{Score > 50%?}
        end

        subgraph OCR["OCR Vision LLM"]
            direction LR
            O1[PDF → Images<br/>150 DPI]
            O2[Détection<br/>Orientation]
            O3[DALLEM Vision<br/>Extraction]
            O1 --> O2 --> O3
        end

        subgraph CHUNK["Chunking Intelligent"]
            direction TB
            C1[Analyse Densité<br/>7 métriques]
            C2{Type Document?}
            C3[Chunking EASA<br/>CS/AMC/GM]
            C4[Chunking Générique<br/>Headers/Listes]
            C1 --> C2
            C2 -->|EASA| C3
            C2 -->|Autre| C4
        end

        subgraph AUGMENT["Augmentation"]
            direction LR
            A1[Keywords<br/>TF + bonus tech]
            A2[Key Phrases<br/>shall/must]
            A3[Références<br/>Cross-links]
        end
    end

    subgraph PHASE2["🌐 PHASE 2 - RÉSEAU (appels minimisés)"]
        direction LR
        EMB[Snowflake Arctic<br/>Embeddings 1024d<br/>Batch 32 textes]
        STORE[FAISS Insert<br/>Batch 4000 chunks]
        EMB --> STORE
    end

    INPUT --> DL
    DL --> EXTRACT
    EXTRACT --> QA
    QA -->|Non| OCR
    QA -->|Oui| CHUNK
    OCR --> CHUNK
    CHUNK --> AUGMENT
    AUGMENT --> PHASE2

    style PHASE1 fill:#e8f5e9
    style PHASE2 fill:#e3f2fd
    style OCR fill:#fff3e0
```

---

## 3. Système de Chunking Adaptatif

```mermaid
flowchart TB
    subgraph DENSITY["📊 ANALYSE DE DENSITÉ"]
        direction LR
        D1[Ratio termes<br/>techniques]
        D2[Densité<br/>numérique]
        D3[Longueur<br/>phrases]
        D4[Densité<br/>listes/tables]
        D5[Références<br/>EASA]
        D6[Ratio<br/>acronymes]
        D7[Densité<br/>crochets]
    end

    subgraph SCORE["🎯 SCORE DE DENSITÉ"]
        S1["≥ 0.5 → Très Dense"]
        S2["0.3-0.5 → Dense"]
        S3["0.15-0.3 → Normal"]
        S4["< 0.15 → Dispersé"]
    end

    subgraph SIZE["📏 TAILLE ADAPTATIVE"]
        T1["800 caractères<br/>Formules, specs"]
        T2["1200 caractères<br/>Technique"]
        T3["1500 caractères<br/>Standard"]
        T4["2000 caractères<br/>Narratif"]
    end

    subgraph EASA["✈️ DÉTECTION EASA"]
        direction LR
        E1["CS 25.xxx"]
        E2["AMC 25.xxx"]
        E3["GM 25.xxx"]
        E4["CS-E xxx"]
        E5["CS-APU xxx"]
    end

    subgraph OUTPUT["📦 CHUNK ENRICHI"]
        direction TB
        OUT["{<br/>  text: '...',<br/>  keywords: [...],<br/>  key_phrases: [...],<br/>  density_type: 'dense',<br/>  density_score: 0.42,<br/>  references_to: [...]<br/>}"]
    end

    D1 & D2 & D3 & D4 & D5 & D6 & D7 --> SCORE
    S1 --> T1
    S2 --> T2
    S3 --> T3
    S4 --> T4
    T1 & T2 & T3 & T4 --> EASA
    EASA --> OUTPUT

    style DENSITY fill:#f3e5f5
    style SIZE fill:#e8f5e9
    style EASA fill:#fff8e1
```

---

## 4. Pipeline de Requête Détaillé

```mermaid
flowchart TB
    Q[/"❓ Question Utilisateur"/]

    subgraph UNDERSTAND["🧠 1. COMPRÉHENSION"]
        direction LR
        U1["Intent Detection<br/>DEFINITION | PROCEDURE<br/>REQUIREMENT | COMPARISON"]
        U2["Complexity<br/>SIMPLE | MODERATE<br/>COMPLEX"]
        U3["Domain<br/>LICENSING | OPERATIONS<br/>AIRWORTHINESS"]
        U4["Adaptive top_k<br/>5 → 30"]
    end

    subgraph EXPAND["🔄 2. EXPANSION (HyDE)"]
        direction TB
        H1[Question originale]
        H2[LLM génère réponse<br/>hypothétique]
        H3[Embedding du<br/>document hypothétique]
        H1 --> H2 --> H3
    end

    subgraph SEARCH["🔍 3. RECHERCHE HYBRIDE"]
        direction TB
        subgraph DENSE["Dense Search"]
            DS[FAISS<br/>Similarité cosinus<br/>Top 50]
        end
        subgraph SPARSE["Sparse Search"]
            SP[BM25<br/>Tokenisation EASA<br/>Top 50]
        end
        FUSION["Fusion<br/>70% Dense + 30% BM25"]
        DS --> FUSION
        SP --> FUSION
    end

    subgraph POST["⚙️ 4. POST-TRAITEMENT"]
        direction TB
        P1["Lost in Middle<br/>Réorganisation extrémités"]
        P2["Context Expansion<br/>Chunks voisins + refs"]
        P3["Feedback Reranking<br/>Boost sources positives"]
        P1 --> P2 --> P3
    end

    subgraph GEN["💬 5. GÉNÉRATION"]
        direction TB
        G1["DALLEM LLM<br/>Temperature: 0.3"]
        G2["Prompt structuré<br/>Contexte + Question"]
        G2 --> G1
    end

    subgraph QA["✅ 6. CONTRÔLE QUALITÉ"]
        direction LR
        QA1["Grounding Analysis<br/>Détection hallucinations"]
        QA2["RAGAS Metrics<br/>Faithfulness, Relevance"]
        QA3["Semantic Cache<br/>Seuil 95%"]
    end

    R[/"📝 Réponse + Sources + Métriques"/]

    Q --> UNDERSTAND
    UNDERSTAND --> EXPAND
    EXPAND --> SEARCH
    SEARCH --> POST
    POST --> GEN
    GEN --> QA
    QA --> R

    style UNDERSTAND fill:#e3f2fd
    style EXPAND fill:#f3e5f5
    style SEARCH fill:#e8f5e9
    style POST fill:#fff3e0
    style GEN fill:#fce4ec
    style QA fill:#e0f2f1
```

---

## 5. Recherche Hybride en Détail

```mermaid
flowchart LR
    subgraph INPUT["Requête"]
        Q["Question embedée"]
    end

    subgraph DENSE["🔵 RECHERCHE DENSE"]
        direction TB
        D1["Snowflake Arctic<br/>Embedding 1024d"]
        D2["FAISS Index<br/>Similarité cosinus"]
        D3["Top 50 résultats<br/>Scores: 0.89, 0.87..."]
        D1 --> D2 --> D3
    end

    subgraph SPARSE["🟢 RECHERCHE SPARSE"]
        direction TB
        S1["Tokenisation<br/>CS 25.571 → cs-25-571"]
        S2["Stopwords FR+EN<br/>Filtrage"]
        S3["BM25 Scoring<br/>Top 50 résultats"]
        S1 --> S2 --> S3
    end

    subgraph FUSION["⚡ FUSION"]
        direction TB
        F1["Score = 0.7×Dense + 0.3×Sparse"]
        F2["Déduplication"]
        F3["Top K final"]
        F1 --> F2 --> F3
    end

    subgraph CONDITION["Activation"]
        C{"> 1000 chunks?"}
    end

    Q --> C
    C -->|Oui| DENSE
    C -->|Oui| SPARSE
    C -->|Non| DENSE
    DENSE --> FUSION
    SPARSE --> FUSION

    style DENSE fill:#e3f2fd
    style SPARSE fill:#e8f5e9
    style FUSION fill:#fff3e0
```

---

## 6. Système de Qualité

```mermaid
flowchart TB
    subgraph GROUNDING["🎯 ANALYSE D'ANCRAGE"]
        direction TB
        G1["Extraction claims<br/>Sujet + Prédicat"]
        G2["Extraction valeurs<br/>Numériques + Unités"]
        G3["Extraction références<br/>Codes EASA"]
        G4["Vérification présence<br/>dans contexte"]
        G5{"Score Ancrage"}
        G1 --> G4
        G2 --> G4
        G3 --> G4
        G4 --> G5
    end

    subgraph RISK["⚠️ NIVEAU DE RISQUE"]
        R1["✅ > 80%<br/>Risque FAIBLE"]
        R2["⚠️ 50-80%<br/>Risque MOYEN"]
        R3["❌ < 50%<br/>Risque ÉLEVÉ"]
    end

    subgraph RAGAS["📊 MÉTRIQUES RAGAS"]
        direction TB
        M1["Faithfulness 25%<br/>Utilise le contexte?"]
        M2["Answer Relevance 25%<br/>Répond à la question?"]
        M3["Context Precision 15%<br/>Sources pertinentes?"]
        M4["Context Utilization 15%<br/>% contexte utilisé"]
        M5["Reference Accuracy 10%<br/>Codes EASA corrects?"]
        M6["Completeness 10%<br/>Réponse complète?"]
        TOTAL["Score Global<br/>Moyenne pondérée"]
        M1 & M2 & M3 & M4 & M5 & M6 --> TOTAL
    end

    subgraph CACHE["💾 CACHE SÉMANTIQUE"]
        direction LR
        C1["Similarité > 95%"]
        C2["TTL: 1 heure"]
        C3["Max: 1000 entrées"]
    end

    G5 --> RISK
    GROUNDING --> RAGAS
    RAGAS --> CACHE

    style GROUNDING fill:#e8f5e9
    style RAGAS fill:#e3f2fd
    style CACHE fill:#f3e5f5
```

---

## 7. Configuration Automatique RAM

```mermaid
flowchart LR
    subgraph DETECT["🔍 Détection"]
        RAM["psutil<br/>RAM disponible"]
    end

    subgraph CONFIG["⚙️ Configuration Auto"]
        direction TB
        C1["≤ 8 Go<br/>Ultra-conservateur"]
        C2["8-12 Go<br/>Conservateur"]
        C3["12-16 Go<br/>Équilibré"]
        C4["16-32 Go<br/>Performance"]
        C5["> 32 Go<br/>Maximum"]
    end

    subgraph PARAMS["📋 Paramètres"]
        direction TB
        P1["Workers: 1<br/>Batch: 4<br/>Streaming: ON"]
        P2["Workers: 2<br/>Batch: 8<br/>Streaming: ON"]
        P3["Workers: 4<br/>Batch: 16<br/>Streaming: OFF"]
        P4["Workers: 6<br/>Batch: 32<br/>Streaming: OFF"]
        P5["Workers: 8<br/>Batch: 64<br/>Streaming: OFF"]
    end

    RAM --> C1 & C2 & C3 & C4 & C5
    C1 --> P1
    C2 --> P2
    C3 --> P3
    C4 --> P4
    C5 --> P5

    style C1 fill:#ffcdd2
    style C2 fill:#ffe0b2
    style C3 fill:#fff9c4
    style C4 fill:#c8e6c9
    style C5 fill:#b2dfdb
```

---

## 8. Architecture des Fichiers

```mermaid
flowchart TB
    subgraph UI["🖥️ Interface"]
        UI1["streamlit_RAG.py<br/>5,800 lignes"]
        UI2["csv_generator_gui.py<br/>1,000 lignes"]
    end

    subgraph INGEST["📥 Ingestion"]
        IN1["ingestion_pipeline.py<br/>800 lignes"]
        IN2["pdf_processing.py<br/>1,400 lignes"]
        IN3["docx_processing.py<br/>300 lignes"]
        IN4["xml_processing.py<br/>400 lignes"]
        IN5["confluence_processing.py<br/>500 lignes"]
    end

    subgraph CHUNK["✂️ Chunking"]
        CH1["chunking.py<br/>1,865 lignes"]
        CH2["semantic_chunking.py<br/>400 lignes"]
        CH3["easa_sections.py<br/>200 lignes"]
    end

    subgraph SEARCH["🔍 Recherche"]
        SE1["rag_query.py<br/>900 lignes"]
        SE2["advanced_search.py<br/>850 lignes"]
        SE3["hybrid_search.py<br/>750 lignes"]
        SE4["faiss_store.py<br/>900 lignes"]
    end

    subgraph QUALITY["✅ Qualité"]
        QA1["answer_grounding.py<br/>600 lignes"]
        QA2["rag_metrics.py<br/>700 lignes"]
        QA3["query_understanding.py<br/>700 lignes"]
        QA4["semantic_cache.py<br/>600 lignes"]
    end

    subgraph UTILS["🔧 Utilitaires"]
        UT1["models_utils.py<br/>700 lignes"]
        UT2["llm_ocr.py<br/>1,200 lignes"]
        UT3["feedback_store.py<br/>700 lignes"]
        UT4["config_manager.py<br/>500 lignes"]
    end

    UI1 --> IN1
    UI1 --> SE1
    IN1 --> IN2 & IN3 & IN4 & IN5
    IN1 --> CH1
    CH1 --> CH2 & CH3
    SE1 --> SE2 & SE3 & SE4
    SE1 --> QA1 & QA2 & QA3 & QA4
    IN1 & SE1 --> UT1 & UT2 & UT3 & UT4

    style UI fill:#e3f2fd
    style INGEST fill:#e8f5e9
    style CHUNK fill:#fff3e0
    style SEARCH fill:#f3e5f5
    style QUALITY fill:#e0f2f1
    style UTILS fill:#fce4ec
```

---

## 9. Flux de Données Complet

```mermaid
sequenceDiagram
    participant U as 👤 Utilisateur
    participant S as 🖥️ Streamlit
    participant I as 📥 Ingestion
    participant C as ✂️ Chunking
    participant E as 🧠 Embeddings
    participant F as 💾 FAISS
    participant Q as 🔍 Query
    participant L as 🤖 LLM
    participant QA as ✅ Qualité

    rect rgb(232, 245, 233)
        Note over U,F: INGESTION
        U->>S: Upload document
        S->>I: Traiter fichier
        I->>I: Extraction texte (triple fallback)
        I->>I: OCR si qualité < 50%
        I->>C: Texte extrait
        C->>C: Analyse densité
        C->>C: Chunking adaptatif
        C->>C: Augmentation (keywords, refs)
        C->>E: Chunks enrichis
        E->>E: Snowflake Arctic (batch 32)
        E->>F: Vecteurs 1024d
        F->>S: ✅ Ingestion terminée
    end

    rect rgb(227, 242, 253)
        Note over U,QA: REQUÊTE
        U->>S: Poser question
        S->>Q: Question
        Q->>Q: Compréhension (intent, complexity)
        Q->>Q: HyDE expansion
        Q->>F: Recherche hybride
        F->>Q: Top K chunks
        Q->>Q: Lost in Middle fix
        Q->>Q: Context expansion
        Q->>L: Contexte + Question
        L->>Q: Réponse générée
        Q->>QA: Vérification
        QA->>QA: Grounding analysis
        QA->>QA: RAGAS metrics
        QA->>S: Réponse + Métriques
        S->>U: 📝 Affichage résultat
    end

    rect rgb(243, 229, 245)
        Note over U,F: FEEDBACK
        U->>S: 👍 / 👎 Feedback
        S->>F: Mise à jour scores
        Note right of F: Améliore le<br/>reranking futur
    end
```

---

## 10. Techniques ML/IA Utilisées

```mermaid
mindmap
  root((RAG Pipeline))
    Retrieval
      HyDE
        Document hypothétique
        Meilleur matching
      Hybrid Search
        Dense FAISS
        Sparse BM25
        Fusion 70/30
      Lost in Middle
        Réorganisation
        Attention LLM
    Chunking
      Adaptatif
        4 tailles
        Densité contenu
      Sémantique
        Frontières naturelles
        Sections/Paragraphes
      EASA
        CS/AMC/GM
        Cross-références
    Qualité
      Grounding
        Hallucinations
        45+ patterns
      RAGAS
        6 métriques
        Score global
      Cache
        Sémantique
        Seuil 95%
    Modèles
      Embeddings
        Snowflake Arctic
        1024 dimensions
      LLM
        DALLEM
        Vision + Chat
      OCR
        Vision LLM
        Auto-rotation
```

---

## Tableau Récapitulatif des Techniques

| Catégorie | Technique | Module | Description |
|-----------|-----------|--------|-------------|
| **Retrieval** | HyDE | `advanced_search.py` | Génère document hypothétique pour meilleur matching |
| | Lost in Middle | `advanced_search.py` | Réorganise résultats pour attention LLM |
| | Recherche Hybride | `hybrid_search.py` | Fusion Dense (70%) + BM25 (30%) |
| | Context Expansion | `chunking.py` | Ajoute chunks voisins et références |
| **Chunking** | Densité Adaptative | `chunking.py` | 4 tailles selon complexité contenu (~750 termes techniques) |
| | Sémantique | `semantic_chunking.py` | Détection frontières naturelles |
| | EASA | `easa_sections.py` | Parsing CS/AMC/GM spécialisé |
| **Qualité** | Grounding | `answer_grounding.py` | Détection hallucinations (45+ patterns réglementaires) |
| | RAGAS | `rag_metrics.py` | 6 métriques d'évaluation |
| | Cache Sémantique | `semantic_cache.py` | Skip requêtes similaires (95%) |
| **Modèles** | Embeddings | `models_utils.py` | Snowflake Arctic 1024d |
| | LLM | `models_utils.py` | DALLEM (Dassault Aviation) |
| | OCR | `llm_ocr.py` | Vision LLM avec auto-rotation |

---

## Dictionnaire Technique Aéronautique (~750 termes)

Le système utilise un dictionnaire étendu de **~750 termes techniques aéronautiques** pour :
- **Analyse de densité** : Détection du contenu technique pour adapter la taille des chunks
- **Extraction de keywords** : Bonus de scoring ×2 pour les termes techniques

### Catégories de Termes

| Catégorie | Termes | Exemples |
|-----------|--------|----------|
| **Réglementaire** | ~70 | CS, AMC, GM, FAR, ATA21-92, ATPL, CPL, AD, SB, STC |
| **Structures** | ~80 | fuselage, longeron, spar, rib, stringer, aileron, flap, slat |
| **Propulsion & APU** | ~90 | turbine, compressor, N1, N2, EGT, APU, thrust reverser, FOD |
| **Systèmes Avion** | ~120 | hydraulic, pneumatic, bleed, pack, IDG, TRU, SSPC |
| **Avionique & Nav** | ~100 | PFD, EFIS, ADIRU, FMS, GPS, ILS, TCAS, EGPWS, FDR |
| **Aérodynamique** | ~80 | lift, drag, stall, Mach, V-speeds (V1, V2, Vref, VMC) |
| **Matériaux** | ~70 | aluminum 2024/7075, titanium Ti6Al4V, CFRP, composite |
| **Maintenance** | ~60 | NDT, inspection, overhaul, A-check, C-check, MEL |
| **Opérations** | ~50 | takeoff, cruise, approach, checklist, SOP, PIC, PF |
| **Sécurité** | ~40 | airworthiness, certification, FMEA, FTA, fail-safe, ETOPS |
| **Hélicoptères** | ~25 | rotor, collective, cyclic, swashplate, autorotation |
| **Infrastructure** | ~30 | runway, taxiway, VASI, PAPI, PCN, VDGS |
| **Termes Français** | ~40 | voilure, nervure, gouverne, centrage, décrochage |

### Utilisation dans le Pipeline

1. **Chunking** (`chunking.py:131`) :
   ```python
   technical_count = sum(1 for w in words if w in TECHNICAL_INDICATORS)
   metrics["technical_ratio"] = technical_count / total_words
   ```

2. **Keyword Scoring** (`chunking.py:245`) :
   ```python
   if word in TECHNICAL_INDICATORS:
       score *= 2.0  # Bonus ×2 pour termes techniques
   ```

### Impact sur la Qualité

| Métrique | Sans termes étendus | Avec ~750 termes |
|----------|---------------------|------------------|
| Détection documents techniques | ~60% | ~95% |
| Pertinence keywords extraits | Moyenne | Haute |
| Chunking adaptatif précision | Basique | Optimisé |

---

*Document généré le 29 novembre 2025 - RAG v6*
