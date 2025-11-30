# Guide du Système RAG EASA

**Auteur** : Renaud LOISON
**Version** : RAG v6
**Date** : Novembre 2025

---

## Introduction

### Définition

**RAG** = **R**etrieval **A**ugmented **G**eneration

Le système RAG EASA est une solution intelligente de recherche et de génération de réponses basée sur des documents techniques aéronautiques. Il permet d'interroger une base documentaire en langage naturel et d'obtenir des réponses précises avec citation des sources.

### Principe de Fonctionnement

1. **Ingestion** : Les documents (PDF, Word, XML, Confluence) sont analysés et indexés
2. **Compréhension** : Le contenu est découpé intelligemment et enrichi de métadonnées
3. **Recherche** : Les passages pertinents sont identifiés lors d'une requête
4. **Génération** : Une réponse est formulée à partir des documents sources

### Bénéfices

| Problématique | Solution Apportée |
|---------------|-------------------|
| Documents éparpillés | Centralisation et indexation unifiée |
| Recherche manuelle chronophage | Réponse en quelques secondes |
| Risque d'omission d'information | Recherche exhaustive automatisée |
| Traçabilité des sources | Citation systématique des références |

---

## Sources de Documents Supportées

### Fichiers Locaux

| Format | Description | Particularités |
|--------|-------------|----------------|
| **PDF** | Documents scannés ou natifs | OCR automatique si nécessaire |
| **DOCX** | Documents Microsoft Word | Extraction des tableaux incluse |
| **XML** | Données structurées | Parsing des sections EASA |

### URL et Liens HTTP

- Documents accessibles via protocole HTTP/HTTPS
- Téléchargement automatique vers un cache local
- Gestion des redirections

### Confluence (Wiki d'Entreprise)

Le système intègre une connectivité native avec Atlassian Confluence permettant :

- **Exploration des espaces** : Liste des espaces accessibles
- **Parcours des pages** : Navigation dans l'arborescence
- **Extraction du contenu** : Texte, tableaux, listes
- **Suivi des liens** : Relations entre pages
- **Synchronisation** : Mise à jour des modifications

#### Configuration Confluence

| Paramètre | Description |
|-----------|-------------|
| URL Confluence | `https://entreprise.atlassian.net` |
| Identifiant | Adresse email du compte |
| Token API | Généré depuis les paramètres Atlassian |

#### Éléments Extraits de Confluence

- Titre et position hiérarchique des pages
- Contenu textuel (paragraphes, titres, sous-titres)
- Tableaux avec conversion en format structuré
- Listes à puces et numérotées
- Métadonnées (auteur, date de modification, espace parent)

---

## Architecture du Pipeline

### Phase 1 : Ingestion des Documents

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│  Extraction │────▶│  Découpage  │
│  Document   │     │    Texte    │     │  Intelligent│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Stockage  │◀────│  Embeddings │◀────│Enrichissement│
│    FAISS    │     │  (vecteurs) │     │  Métadonnées │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Étapes du traitement :**

1. **Extraction du texte**
   - PDF : Triple fallback (pdfplumber → pdfminer → PyMuPDF)
   - Documents scannés : OCR via modèle Vision (DALLEM)
   - Confluence : API REST + parsing HTML

2. **Découpage intelligent (Chunking)**
   - Taille adaptative selon la densité du contenu
   - Préservation des sections réglementaires EASA
   - Respect des frontières sémantiques

3. **Enrichissement des chunks**
   - Extraction des mots-clés techniques
   - Identification des références croisées
   - Calcul du score de densité

4. **Vectorisation (Embeddings)**
   - Modèle : Snowflake Arctic (1024 dimensions)
   - Traitement par lots de 32 textes

5. **Indexation FAISS**
   - Stockage des vecteurs pour recherche rapide
   - Métadonnées associées en JSON

### Phase 2 : Recherche et Génération

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Requête   │────▶│ Analyse     │────▶│  Recherche  │
│ Utilisateur │     │ Intention   │     │   Hybride   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Réponse   │◀────│  Génération │◀────│  Sélection  │
│  + Sources  │     │    LLM      │     │   Top-K     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Étapes du traitement :**

1. **Analyse de la requête**
   - Détection du type : Définition, Procédure, Exigence, Comparaison
   - Évaluation de la complexité
   - Adaptation du nombre de résultats (top_k)

2. **Recherche hybride**
   - Recherche dense : Similarité sémantique (FAISS)
   - Recherche sparse : Correspondance lexicale (BM25)
   - Fusion pondérée : 70% sémantique + 30% lexical

3. **Post-traitement des résultats**
   - Réorganisation "Lost in Middle" pour optimiser l'attention du LLM
   - Expansion contextuelle (chunks adjacents)
   - Re-ranking basé sur l'historique des feedbacks

4. **Génération de la réponse**
   - Modèle LLM : DALLEM (Dassault Aviation)
   - Température : 0.3 (extraction factuelle)
   - Instruction de citation des sources

5. **Contrôle qualité**
   - Analyse d'ancrage (grounding)
   - Détection des hallucinations potentielles
   - Score de confiance

---

## Fonctionnalités Principales

### Spécialisation EASA

Le système reconnaît et traite spécifiquement les documents réglementaires aéronautiques :

| Code | Désignation | Traitement Appliqué |
|------|-------------|---------------------|
| CS 25.xxx | Certification Specifications | Découpage par section réglementaire |
| AMC 25.xxx | Acceptable Means of Compliance | Liaison avec CS correspondant |
| GM 25.xxx | Guidance Material | Préservation du contexte |
| AD | Airworthiness Directive | Extraction structurée |
| SB | Service Bulletin | Indexation des métadonnées |

### Dictionnaire Technique (~750 termes)

Le système intègre un dictionnaire de **750 termes techniques aéronautiques** répartis en 13 catégories :

| Catégorie | Nombre | Exemples |
|-----------|--------|----------|
| Réglementaire | ~70 | CS, AMC, GM, FAR, ATA, ATPL, AD, SB |
| Structures | ~80 | fuselage, longeron, spar, rib, aileron |
| Propulsion & APU | ~90 | turbine, compressor, N1, EGT, APU |
| Systèmes Avion | ~120 | hydraulic, pneumatic, bleed, IDG |
| Avionique & Navigation | ~100 | PFD, ADIRU, FMS, TCAS, EGPWS |
| Aérodynamique | ~80 | lift, drag, stall, V-speeds |
| Matériaux | ~70 | 2024, 7075, Ti6Al4V, CFRP |
| Maintenance | ~60 | NDT, A-check, MEL, overhaul |
| Opérations | ~50 | takeoff, SOP, PIC, checklist |
| Sécurité | ~40 | airworthiness, FMEA, ETOPS |
| Hélicoptères | ~25 | rotor, collective, autorotation |
| Infrastructure | ~30 | runway, VASI, PAPI, PCN |
| Termes Français | ~40 | voilure, nervure, gouverne |

**Impact** : Les termes techniques reçoivent un bonus de scoring ×2 lors de l'extraction des mots-clés.

### Techniques de Recherche Avancées

| Technique | Description | Bénéfice |
|-----------|-------------|----------|
| **HyDE** | Génération d'un document hypothétique avant recherche | Amélioration du rappel |
| **Recherche Hybride** | Combinaison dense (sémantique) + sparse (BM25) | Précision et exhaustivité |
| **Lost in Middle** | Réorganisation des résultats aux extrémités | Meilleure attention LLM |
| **Context Expansion** | Inclusion des chunks adjacents | Contexte complet |

### Contrôle Qualité des Réponses

Chaque réponse générée fait l'objet d'une vérification :

| Métrique | Description | Seuils |
|----------|-------------|--------|
| Score d'ancrage | Pourcentage d'affirmations retrouvées dans les sources | >80% = Fiable |
| Risque hallucination | Évaluation du risque de contenu non sourcé | Faible / Moyen / Élevé |
| Métriques RAGAS | Faithfulness, Relevance, Precision, Utilization | Score global 0-1 |

### Système de Feedback

Le système intègre une boucle d'amélioration continue :

- **Feedback positif** (👍) : La source est valorisée pour les requêtes futures
- **Feedback négatif** (👎) : La source est déprécioriée dans le classement
- **Statistiques** : Suivi des performances par collection et par période

---

## Intégration Confluence

### Processus de Connexion

1. Configuration des paramètres d'authentification
2. Sélection de l'espace Confluence cible
3. Parcours automatique de l'arborescence des pages

### Structure d'Indexation

```
Espace Confluence
├── Page Principale
│   ├── Contenu textuel
│   ├── Tableaux
│   └── Listes
├── Sous-page 1
│   └── ...
└── Sous-page 2
    └── ...
```

### Modes de Mise à Jour

| Mode | Description | Cas d'usage |
|------|-------------|-------------|
| Complète | Réingestion de tout l'espace | Première indexation, restructuration |
| Incrémentale | Mise à jour des pages modifiées | Synchronisation régulière |

### Avantages de l'Intégration Confluence

| Aspect | Bénéfice |
|--------|----------|
| Centralisation | Accès unifié aux connaissances d'entreprise |
| Actualité | Documentation toujours synchronisée |
| Collaboration | Contributions de l'ensemble des équipes |
| Hiérarchie | Structure organisationnelle préservée |

---

## Organisation des Données

### Structure des Collections

Les documents sont organisés en collections thématiques :

```
Base FAISS/
├── certification/           ← Documents CS-25, AMC, GM
│   ├── index.faiss
│   └── metadata.json
├── maintenance/             ← Manuels AMM, SRM, IPC
│   ├── index.faiss
│   └── metadata.json
├── operations/              ← FCOM, QRH, procédures
│   ├── index.faiss
│   └── metadata.json
└── confluence_wiki/         ← Pages Confluence
    ├── index.faiss
    └── metadata.json
```

### Avantages de l'Organisation par Collection

- **Recherche ciblée** : Interrogation d'une collection spécifique
- **Recherche globale** : Interrogation multi-collections
- **Gestion indépendante** : Mise à jour collection par collection
- **Contrôle d'accès** : Permissions par collection

---

## Interface Utilisateur

### Module d'Ingestion

1. Sélection de la source (fichier, URL, Confluence)
2. Choix de la collection de destination
3. Lancement du traitement avec suivi de progression
4. Rapport de résultat (nombre de chunks créés)

### Module de Requête

1. Sélection de la ou des collections à interroger
2. Saisie de la question en langage naturel
3. Affichage de la réponse avec sources et score de confiance
4. Enregistrement du feedback

### Module de Configuration

- Paramétrage des répertoires de travail
- Configuration de la connexion Confluence
- Réglage des paramètres avancés (top_k, seuils, options)

---

## Performances

### Adaptation Automatique aux Ressources

| RAM Disponible | Mode | Configuration |
|----------------|------|---------------|
| ≤ 8 Go | Ultra-conservateur | 1 worker, batch 4, streaming |
| 8-12 Go | Conservateur | 2 workers, batch 8, streaming |
| 12-16 Go | Équilibré | 4 workers, batch 16 |
| 16-32 Go | Performance | 6 workers, batch 32 |
| > 32 Go | Maximum | 8 workers, batch 64 |

### Temps de Traitement Typiques

| Opération | Durée Indicative |
|-----------|------------------|
| Ingestion de 100 pages PDF | 1-2 minutes |
| Requête simple | 1-2 secondes |
| Requête complexe (HyDE activé) | 3-5 secondes |

### Cache Sémantique

Les requêtes similaires (similarité > 95%) bénéficient du cache :
- Réponse instantanée
- Économie de ressources computationnelles
- Durée de validité : 1 heure

---

## Synthèse

```
┌────────────────────────────────────────────────────────────┐
│                    SYSTÈME RAG EASA                        │
│                    Auteur : Renaud LOISON                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  SOURCES           TRAITEMENT         UTILISATION          │
│  ────────          ──────────         ───────────          │
│  • PDF             • Extraction       • Requêtes           │
│  • Word            • OCR si requis    • Réponses sourcées  │
│  • XML             • Chunking EASA    • Feedback           │
│  • Confluence      • Embeddings       • Amélioration       │
│  • URL             • Indexation         continue           │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  CARACTÉRISTIQUES PRINCIPALES                              │
│                                                            │
│  • ~750 termes techniques aéronautiques                    │
│  • Recherche hybride (sémantique + lexicale)               │
│  • Détection des hallucinations                            │
│  • Intégration Confluence native                           │
│  • Adaptation automatique aux ressources système           │
│  • Spécialisation documents EASA (CS, AMC, GM)             │
└────────────────────────────────────────────────────────────┘
```

---

**Document rédigé par Renaud LOISON**
**RAG v6 - Novembre 2025**
