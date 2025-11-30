# 📦 Installation sur un réseau partagé - RaGME_UP - PROP

Ce guide explique comment déployer l'application RaGME_UP - PROP sur un réseau partagé Windows pour un usage multi-utilisateurs.

---

## 🎯 Architecture réseau recommandée

### Pourquoi FAISS est parfait pour le réseau Windows

- ✅ **Pas de SQLite** = pas de problèmes de verrous de fichiers
- ✅ **Fichiers simples** = synchronisation réseau Windows transparente
- ✅ **Rapide** = recherche vectorielle optimisée
- ✅ **Multi-utilisateurs** = accès concurrent sans conflit
- ✅ **Fiable** = sauvegarde automatique après chaque ajout

### Structure des dossiers partagés

```
\\SERVEUR\RAG\
├── streamlit_RAG.py          # Application principale
├── csv_generator_gui.py      # Interface GUI pour CSV
├── # === INGESTION ===
├── rag_ingestion.py          # Ingestion classique
├── ingestion_pipeline.py     # Pipeline optimisé 2 phases (NEW)
├── pdf_processing.py         # Traitement PDF (pdfminer + PyMuPDF)
├── docx_processing.py        # Traitement DOCX (python-docx)
├── xml_processing.py         # Traitement XML EASA
├── # === CHUNKING ===
├── chunking.py               # Chunking adaptatif intelligent
├── semantic_chunking.py      # Chunking sémantique (NEW)
├── easa_sections.py          # Parser sections EASA
├── # === REQUÊTE ===
├── rag_query.py              # Requêtes RAG + HyDE + Lost in Middle
├── hybrid_search.py          # Recherche hybride BM25 + Dense (NEW)
├── query_understanding.py    # Analyse d'intention (NEW)
├── # === QUALITÉ RAG ===
├── answer_grounding.py       # Détection hallucinations (NEW)
├── rag_metrics.py            # Métriques RAGAS (NEW)
├── semantic_cache.py         # Cache sémantique (NEW)
├── # === OCR ===
├── llm_ocr.py                # OCR Vision LLM + rotation (NEW)
├── # === STOCKAGE ===
├── faiss_store.py            # Store FAISS
├── feedback_store.py         # Stockage feedbacks
├── # === CONFIG ===
├── config_manager.py         # Gestion configuration
├── models_utils.py           # Embeddings et LLM
├── requirements.txt          # Dépendances Python
├── install.bat               # Script d'installation
├── launch.bat                # Script de lancement
├── config.json               # Configuration utilisateur (généré)
├── README.md                 # Documentation principale
├── GUIDE_UTILISATEUR.md      # Documentation utilisateur
├── INSTALLATION_RESEAU.md    # Ce document
├── ARCHITECTURE_TECHNIQUE.md # Documentation technique
└── FAISS_DATABASE\           # Dossier partagé pour les données
    ├── BaseDB\               # Bases FAISS (une par projet)
    │   ├── normes_easa\      # Exemple: base normes EASA
    │   │   ├── CS\           # Collection CS
    │   │   │   ├── index.faiss
    │   │   │   └── metadata.json
    │   │   ├── AMC\          # Collection AMC
    │   │   └── GM\           # Collection GM
    │   └── manuels\          # Exemple: base manuels
    ├── CSV_Ingestion\        # CSV pour ingestion
    ├── Fichiers_Tracking_CSV\# CSV de tracking (déduplication)
    └── Feedbacks\            # Feedbacks utilisateurs
```

### ⚠️ Important : Chemins sans espaces

FAISS (bibliothèque C++) ne gère pas les espaces dans les chemins sur Windows réseau.

❌ **Mauvais :**
```
N:\Mon Dossier\Base de données\
```

✅ **Bon :**
```
N:\Mon_Dossier\BaseDB\
```

### Configuration requise

**Sur le serveur :**
- Partage réseau accessible en lecture/écriture
- Espace disque suffisant pour les bases FAISS
- Chemins sans espaces (voir ci-dessus)

**Sur chaque poste client :**
- Windows 10/11
- Python 3.8 ou supérieur (3.11 recommandé)
- Accès au partage réseau
- 4 GB RAM minimum (8 GB recommandé)

---

## 🚀 Installation pour les utilisateurs

### Étape 1 : Installer Python (si pas déjà installé)

1. Téléchargez Python depuis : https://www.python.org/downloads/
2. **Important** : Cochez **"Add Python to PATH"** lors de l'installation
3. Vérifiez l'installation :
   ```cmd
   python --version
   ```

### Étape 2 : Installer les dépendances

1. Ouvrez l'Explorateur Windows
2. Naviguez vers le dossier réseau : `\\SERVEUR\RAG\`
3. Double-cliquez sur **`install.bat`**
4. Attendez la fin de l'installation (peut prendre 5-10 minutes)

**Que fait install.bat ?**
- ✅ Vérifie que Python est installé
- ✅ Met à jour pip
- ✅ Installe toutes les dépendances (Streamlit, FAISS, PyMuPDF, etc.)
- ✅ Installe CustomTkinter pour la GUI
- ✅ Installe faiss-cpu (ou faiss-gpu si GPU disponible)

### Étape 3 : Lancer l'application

1. Dans le dossier réseau `\\SERVEUR\RAG\`
2. Double-cliquez sur **`launch.bat`**
3. L'application s'ouvre automatiquement dans votre navigateur
4. URL : http://localhost:8501

**Pour arrêter l'application :**
- Fermez la fenêtre de commande
- Ou appuyez sur `Ctrl+C`

---

## ⚙️ Configuration des chemins

### Modifier les chemins par défaut

Modifiez les constantes dans `streamlit_RAG.py` (lignes 48-51) :

```python
# ⚠️ IMPORTANT : Utilisez des chemins SANS ESPACES pour compatibilité FAISS C++
BASE_ROOT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\BaseDB"
CSV_IMPORT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\CSV_Ingestion"
CSV_EXPORT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\Fichiers_Tracking_CSV"
```

**Format des chemins réseau :**
- UNC : `\\SERVEUR\PARTAGE\Dossier_Sans_Espaces`
- Lecteur mappé : `N:\Dossier_Sans_Espaces`
- **Évitez absolument les espaces** dans les chemins

### Créer les dossiers nécessaires

⚠️ **Noms sans espaces !**

```cmd
mkdir "\\SERVEUR\RAG\FAISS_DATABASE"
mkdir "\\SERVEUR\RAG\FAISS_DATABASE\BaseDB"
mkdir "\\SERVEUR\RAG\FAISS_DATABASE\CSV_Ingestion"
mkdir "\\SERVEUR\RAG\FAISS_DATABASE\Fichiers_Tracking_CSV"
mkdir "\\SERVEUR\RAG\FAISS_DATABASE\Feedbacks"
```

---

## 🔧 Configuration du Chunking

Le système utilise un chunking adaptatif intelligent. Les paramètres peuvent être personnalisés.

### Paramètres par défaut

| Paramètre | Valeur | Fichier | Description |
|-----------|--------|---------|-------------|
| `base_chunk_size` | 1000 | rag_ingestion.py | Taille de base avant adaptation |
| `min_chunk_size` | 200 | chunking.py | Taille minimale (fusion si inférieur) |
| `max_chunk_size` | 2000-2500 | rag_ingestion.py | Taille maximale après adaptation |
| `overlap` | 100 | chunking.py | Chevauchement entre chunks |
| `merge_small_sections` | True | chunking.py | Fusion sections < 300 caractères |

### Tailles adaptatives par densité de contenu

Le système analyse automatiquement la densité du document :

| Densité | Caractéristiques | Taille chunk |
|---------|------------------|--------------|
| `very_dense` | Code, formules, tableaux | 800 caractères |
| `dense` | Spécifications, listes | 1200 caractères |
| `normal` | Prose technique | 1500 caractères |
| `sparse` | Narratif, introductions | 2000 caractères |

### Métriques analysées

- Densité de termes techniques (80+ mots-clés aéronautiques)
- Ratio nombres/formules dans le texte
- Longueur moyenne des phrases
- Présence de listes et tableaux
- Densité de références (CS, AMC, GM, FAR, JAR)
- Ratio d'acronymes

### Personnalisation avancée

Pour modifier le comportement, éditez `rag_ingestion.py` (ligne ~180) :

```python
adapted_chunk_size = _get_adaptive_chunk_size(
    text,
    base_size=1000,      # Taille de base (modifier ici)
    min_size=600,        # Minimum adaptatif (modifier ici)
    max_size=2000        # Maximum adaptatif (modifier ici)
)
```

---

## 🌐 Avantages FAISS sur réseau

### Avantages de FAISS

| Fonctionnalité | FAISS |
|----------------|-------|
| Base de données | Fichiers simples (.faiss + .json) |
| Verrous réseau | ✅ Aucun problème |
| Performance réseau | 🚀 Rapide |
| Multi-utilisateurs | ✅ Sans conflit |
| Synchronisation | ✅ Immédiate |
| Espaces dans chemins | ❌ Non supportés (C++) |

### Pourquoi FAISS est adapté au réseau

1. **Fichiers indépendants** : Chaque collection = 2 fichiers (index.faiss + metadata.json)
2. **Pas de verrous** : Pas de SQLite = pas de problèmes de "database locked"
3. **Sauvegarde auto** : Après chaque ajout, fichiers synchronisés
4. **Lecture concurrente** : Plusieurs utilisateurs peuvent lire simultanément
5. **Écriture séquentielle** : Une ingestion à la fois (par design)

---

## 🔒 Coordination multi-utilisateurs

### Comment ça fonctionne avec FAISS

✅ **Lectures parallèles** : Illimitées, aucun conflit
✅ **Ingestion séquentielle** : Par design (sauvegarde après chaque batch)
✅ **Pas de corruption** : Fichiers indépendants par collection

### Bonnes pratiques

✅ **À faire :**
- Plusieurs utilisateurs peuvent interroger en même temps
- Ingérer sur différentes bases simultanément (OK)
- Vérifier que les chemins n'ont pas d'espaces

❌ **À éviter :**
- Ingérer simultanément dans la même base (résultats imprévisibles)
- Utiliser des espaces dans les noms de dossiers
- Supprimer manuellement les fichiers .faiss ou .json

### Gestion des conflits

**Si deux utilisateurs ingèrent dans la même base :**
- Dernier sauvegardé gagne (comportement FAISS)
- Pas de corruption de fichiers
- Recommandation : communiquer avant grosse ingestion

---

## 🛠️ Dépannage

### Python n'est pas reconnu

**Symptôme :** `'python' n'est pas reconnu...`

**Solution :**
1. Réinstallez Python en cochant "Add Python to PATH"
2. Ou ajoutez manuellement Python au PATH système

### Erreur d'accès au réseau

**Symptôme :** `Access denied` ou `Permission denied`

**Solution :**
1. Vérifiez les droits d'accès au partage réseau
2. Assurez-vous d'avoir les droits en lecture/écriture
3. Testez avec `dir \\SERVEUR\RAG\`

### Erreur FAISS avec espaces dans le chemin

**Symptôme :** `Error in faiss::FileIOWriter` ou `No such file or directory`

**Solution :**
1. Renommez les dossiers pour supprimer les espaces
2. Mettez à jour les chemins dans `streamlit_RAG.py`
3. Exemples :
   - `Base de données` → `BaseDB`
   - `Fichiers CSV` → `Fichiers_CSV`

### MemoryError lors de l'ingestion

**Symptôme :** `MemoryError` ou processus qui crashent

**Solution :**
- ✅ **Déjà corrigé** : Version actuelle utilise ThreadPoolExecutor
- Si problème persiste : fermez autres applications gourmandes
- Vérifiez RAM disponible (4 GB minimum)

### Caractères spéciaux dans noms de PDF

**Symptôme :** `UnicodeEncodeError: surrogates not allowed`

**Solution :**
- ✅ **Déjà corrigé** : Nettoyage automatique des surrogates
- Version actuelle gère tous les caractères Unicode
- Extensions préservées automatiquement

### La GUI ne s'ouvre pas

**Symptôme :** Erreur au clic sur "Création d'un CSV"

**Solution :**
1. Vérifiez que CustomTkinter est installé : `pip install customtkinter`
2. Relancez `install.bat`
3. Vérifiez que Pillow est installé : `pip install pillow`

### Lenteurs sur le réseau

**Symptôme :** L'application est lente

**Solutions :**
- ✅ **Utilisez le cache local** : Dans la sidebar, cliquez sur "📥 Copier local"
- ✅ FAISS est rapide et déjà optimisé
- Utilisez un lecteur réseau mappé (N:) au lieu de UNC (\\SERVEUR)
- Vérifiez la bande passante réseau
- FAISS charge en RAM = requêtes ultra-rapides après premier chargement

**Cache local automatique :**
- Copie la base FAISS en local (`~/.cache/ragme_up/`)
- Requêtes ultra-rapides sans accès réseau
- Validation automatique de la fraîcheur
- Avertissement si la base réseau a été modifiée

---

## 📊 Monitoring

### Fichiers de logs

Les logs sont créés localement sur chaque poste :
- `rag_da_debug.log` (dans le dossier de l'application)

Logs détaillés pour :
- Extraction PDF (pdfplumber + pdfminer + PyMuPDF fallback)
- Extraction tableaux (détection et formatage)
- Traitement Unicode/surrogates
- Ingestion FAISS (chunks ajoutés)
- Erreurs réseau éventuelles

### Vérifier la santé des bases FAISS

Pour chaque base, vérifiez :
```
BaseDB\[nom_base]\[collection]\
├── index.faiss        # Index vectoriel (taille variable)
└── metadata.json      # Métadonnées (IDs, documents, etc.)
```

**Fichiers corrompus :**
- Très rare avec FAISS
- Si problème : supprimez la collection et réingérez

---

## 🔄 Mises à jour

### Mettre à jour l'application

1. Copiez les nouveaux fichiers Python sur le serveur
2. Les utilisateurs n'ont qu'à relancer `launch.bat`
3. Pas besoin de réinstaller (sauf nouvelles dépendances)

### Mettre à jour les dépendances

Si `requirements.txt` a changé :
1. Chaque utilisateur doit relancer `install.bat`
2. Ou manuellement : `pip install -r requirements.txt --upgrade`

---

## 🎓 Formation des utilisateurs

### Documents à partager

1. **GUIDE_UTILISATEUR.md** : Guide complet d'utilisation
2. **INSTALLATION_RESEAU.md** : Ce document
3. **Quick Start** : Voir section ci-dessous

### Quick Start (1 page)

```
=== RaGME_UP - PROP - Démarrage rapide ===

1. INSTALLER (première fois seulement)
   \\SERVEUR\RAG\install.bat
   → Installe Python + dépendances + FAISS

2. LANCER
   \\SERVEUR\RAG\launch.bat
   → Navigateur s'ouvre automatiquement

3. CRÉER UN CSV
   Onglet "Gestion CSV" → Création d'un CSV
   → Scanner un répertoire → Assigner groupes → Sauvegarder

4. INGÉRER DES DOCUMENTS
   Onglet "Ingestion documents"
   → Uploader le CSV → Lancer ingestion
   → Extraction automatique pièces jointes PDF
   → OCR automatique pour PDFs scannés (LLM Vision)

5. POSER DES QUESTIONS
   Onglet "Questions RAG"
   → Sélectionner base + collection → Taper question
   → Sources cliquables avec bouton "Ouvrir"

NOUVEAUTÉS v2.0 :
✅ HyDE + Lost in Middle (amélioration retrieval)
✅ Hybrid Search (BM25 + Dense + RRF)
✅ Answer Grounding (détection hallucinations)
✅ Query Understanding (analyse d'intention)
✅ Semantic Cache (économie appels LLM)
✅ LLM Vision OCR avec rotation automatique
✅ Pipeline d'ingestion optimisé (70% moins d'appels réseau)
✅ Cache local automatique (performances réseau optimales)
✅ FAISS = compatible réseau Windows

Aide : GUIDE_UTILISATEUR.md (accessible directement depuis l'interface)
```

---

## 💡 Optimisations avancées

### Option 1 : Lecteur réseau mappé (Recommandé)

Plus rapide que UNC + meilleure compatibilité :
```cmd
net use N: \\SERVEUR\RAG /persistent:yes
```

Puis dans `streamlit_RAG.py` :
```python
BASE_ROOT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\BaseDB"
CSV_IMPORT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\CSV_Ingestion"
CSV_EXPORT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\Fichiers_Tracking_CSV"
```

### Option 2 : Cache local pour les requêtes (RECOMMANDÉ)

Le système propose un **cache local automatique** pour des performances optimales :

**Comment activer :**
1. Dans l'onglet "Questions RAG", sélectionnez votre base
2. Dans la sidebar, cliquez sur **"📥 Copier local"**
3. Choisissez : **Base en cours** ou **Toutes les bases** (plus long)
4. Le cache est ensuite utilisé automatiquement

**Avantages :**
- ✅ Requêtes ultra-rapides (lecture locale)
- ✅ Pas d'accès réseau pour les recherches
- ✅ Validation automatique de la fraîcheur
- ✅ Avertissement si le cache devient obsolète

**Fonctionnement :**
- Cache stocké dans `~/.cache/ragme_up/`
- Validation automatique à chaque requête (comparaison hash)
- Si base réseau modifiée → avertissement + fallback réseau
- Invalidation automatique après ingestion locale

**Structure du cache :**
```
~/.cache/ragme_up/
└── [hash_collection]/
    ├── index.faiss      # Index vectoriel local
    ├── metadata.json    # Métadonnées
    └── .hash            # Hash de validation
```

### Option 3 : GPU pour grandes bases (Avancé)

Si bases très volumineuses (100K+ documents) :
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

**Requis :** GPU NVIDIA + CUDA installé

---

## 📞 Support

Pour toute question ou problème :
1. Consultez **GUIDE_UTILISATEUR.md**
2. Consultez les logs : `rag_da_debug.log`
3. Vérifiez les chemins (pas d'espaces)

**Auteur** : Renaud LOISON

---

## ✅ Checklist déploiement

Avant de déployer en production :

- [ ] Python 3.8+ installé sur tous les postes
- [ ] Chemins réseau configurés **sans espaces**
- [ ] Dossiers créés (BaseDB, CSV_Ingestion, Fichiers_Tracking_CSV)
- [ ] Droits lecture/écriture vérifiés
- [ ] install.bat exécuté sur chaque poste
- [ ] launch.bat testé
- [ ] Ingestion test réussie
- [ ] Requêtes test réussies
- [ ] Extraction pièces jointes testée
- [ ] Documentation distribuée (GUIDE_UTILISATEUR.md)

---

**Bonne utilisation de RaGME_UP - PROP avec FAISS ! 🚀**

---

**Version:** 2.0
**Dernière mise à jour:** 2025-11-29
