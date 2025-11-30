"""
Hybrid Search Module - Dense + Sparse (BM25)
Combine la recherche vectorielle (dense) avec BM25 (sparse) pour améliorer le recall.

Activation automatique:
- Si collection > HYBRID_MIN_CHUNKS : active BM25
- Sinon : recherche dense uniquement (plus rapide)

Phase 2 improvements v1.2
"""

import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import pickle
import os
import hashlib

logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Seuil pour activer automatiquement la recherche hybride
HYBRID_MIN_CHUNKS = 500  # Activer BM25 si > 500 chunks (abaissé pour meilleur recall)

# Poids pour la fusion des scores (dense vs sparse)
DEFAULT_DENSE_WEIGHT = 0.8  # 80% dense, 20% BM25 (sémantique prioritaire)
DEFAULT_SPARSE_WEIGHT = 0.2

# Paramètres BM25
BM25_K1 = 1.5  # Term frequency saturation
BM25_B = 0.75  # Document length normalization


# =============================================================================
#  BM25 INDEX
# =============================================================================

class BM25Index:
    """
    Index BM25 pour la recherche sparse.

    Implémentation légère sans dépendances externes (rank_bm25).
    Optimisée pour les documents techniques (EASA, CS, AMC).
    """

    def __init__(
        self,
        k1: float = BM25_K1,
        b: float = BM25_B,
        log=None
    ):
        self.k1 = k1
        self.b = b
        self._log = log or logger

        # Index data
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> num docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # doc_idx -> {term: freq}
        self.vocab: set = set()
        self.n_docs: int = 0
        self._is_built: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenization optimisée pour documents techniques/réglementaires.

        Préserve:
        - Codes EASA complets (CS 25.571, AMC1 25.1309, CS-APU 25.1309, etc.)
        - Termes normatifs (shall, must, may, should) - essentiels en réglementation!
        """
        if not text:
            return []

        # Normaliser
        text = text.lower()

        # Extraire d'abord les codes techniques (les préserver intacts)
        # Patterns améliorés pour capturer tous les formats EASA
        technical_patterns = [
            # CS avec suffixes: CS-E, CS-APU, CS-P, etc.
            r'cs(?:-[a-z]+)?[\s\-]?\d+[a-z]?(?:[.\-]\d+[a-z]?)?(?:\([a-z0-9]+\))*',
            # AMC/GM avec variantes numérotées: AMC, AMC1, AMC2, GM1, GM2
            r'amc\d{0,2}(?:-[a-z]+)?[\s\-]?\d+[a-z]?(?:[.\-]\d+[a-z]?)?(?:\([a-z0-9]+\))*',
            r'gm\d{0,2}(?:-[a-z]+)?[\s\-]?\d+[a-z]?(?:[.\-]\d+[a-z]?)?(?:\([a-z0-9]+\))*',
            # FAR/FAA
            r'(?:far|faa)[\s\-]?\d+[.\-]\d+',
            # FCL: FCL.055, FCL.055.A
            r'fcl[.\-]?\d+(?:\.[a-z]+(?:\.\d+)?)?',
            # CAT, ORO, SPA, NCO, NCC, SPO: CAT.OP.MPA.100, ORO.GEN.105, etc.
            r'(?:cat|oro|spa|nco|ncc|spo)\.[a-z]+(?:\.[a-z]+)*\.\d+',
        ]

        technical_tokens = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normaliser le code (enlever espaces, garder tirets)
                normalized = re.sub(r'\s+', '-', match.strip())
                technical_tokens.append(normalized)

        # Tokenizer le reste (alphanumérique)
        words = re.findall(r'\b[a-zA-Z]{2,}\b|\b\d+\b', text)

        # Stopwords SANS les termes normatifs (shall, must, may, should sont ESSENTIELS
        # en documentation réglementaire - ils indiquent le niveau d'exigence)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'can',  # Garder: shall, must, may, should, might
            'this', 'that', 'these', 'those', 'it', 'its',
            'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou',
            'also', 'any', 'each', 'which', 'who', 'whom', 'whose',
            'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very',
        }

        tokens = technical_tokens + [w for w in words if w not in stopwords and len(w) > 1]
        return tokens

    def build(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Construit l'index BM25 à partir des documents.

        Args:
            documents: Liste des textes à indexer
            doc_ids: IDs optionnels pour chaque document
        """
        self._log.info(f"[BM25] Building index for {len(documents)} documents...")

        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]
        self.n_docs = len(documents)

        if self.n_docs == 0:
            self._is_built = True
            return

        # Tokeniser tous les documents
        self.term_freqs = []
        self.doc_lengths = []
        total_length = 0

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)

            # Compter les fréquences des termes dans ce document
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))

            # Mettre à jour le vocabulaire
            self.vocab.update(tf.keys())

        self.avg_doc_length = total_length / self.n_docs if self.n_docs > 0 else 0

        # Calculer les document frequencies (DF)
        self.doc_freqs = defaultdict(int)
        for tf in self.term_freqs:
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self._is_built = True
        self._log.info(
            f"[BM25] Index built: {self.n_docs} docs, {len(self.vocab)} unique terms, "
            f"avg_len={self.avg_doc_length:.1f}"
        )

    def _compute_idf(self, term: str) -> float:
        """Calcule l'IDF d'un terme."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Formule IDF standard avec smoothing
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Calcule le score BM25 d'un document pour une requête.
        """
        if not self._is_built or doc_idx >= self.n_docs:
            return 0.0

        # Protection contre division par zéro si avg_doc_length == 0
        if self.avg_doc_length == 0:
            return 0.0

        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_doc = self.term_freqs[doc_idx]

        for term in query_tokens:
            if term not in tf_doc:
                continue

            tf = tf_doc[term]
            idf = self._compute_idf(term)

            # Formule BM25
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[Tuple[int, float]]:
        """
        Recherche BM25.

        Args:
            query: Requête textuelle
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de (doc_idx, score) triée par score décroissant
        """
        if not self._is_built or self.n_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Scorer tous les documents
        scores = []
        for doc_idx in range(self.n_docs):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))

        # Trier par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def save(self, path: str) -> None:
        """Sauvegarde l'index BM25."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'doc_freqs': dict(self.doc_freqs),
            'term_freqs': self.term_freqs,
            'vocab': self.vocab,
            'n_docs': self.n_docs,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        self._log.info(f"[BM25] Index saved to {path}")

    @classmethod
    def load(cls, path: str, log=None) -> 'BM25Index':
        """Charge un index BM25 depuis un fichier.

        ATTENTION SÉCURITÉ: pickle.load() peut exécuter du code arbitraire.
        Ne chargez que des fichiers provenant de sources fiables (générés par cette application).
        """
        _log = log or logger
        # Vérifier que le fichier provient d'un répertoire attendu (sécurité basique)
        import os
        abs_path = os.path.abspath(path)
        if not abs_path.endswith('.pkl') and not abs_path.endswith('.pickle'):
            _log.warning(f"[BM25] SÉCURITÉ: Fichier sans extension .pkl/.pickle: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls(k1=data['k1'], b=data['b'], log=_log)
        index.documents = data['documents']
        index.doc_ids = data['doc_ids']
        index.doc_lengths = data['doc_lengths']
        index.avg_doc_length = data['avg_doc_length']
        index.doc_freqs = defaultdict(int, data['doc_freqs'])
        index.term_freqs = data['term_freqs']
        index.vocab = data['vocab']
        index.n_docs = data['n_docs']
        index._is_built = True

        _log.info(f"[BM25] Index loaded from {path} ({index.n_docs} docs)")
        return index


# =============================================================================
#  HYBRID SEARCH
# =============================================================================

def should_use_hybrid_search(
    collection_size: int,
    min_threshold: int = HYBRID_MIN_CHUNKS,
    log=None
) -> bool:
    """
    Détermine automatiquement si la recherche hybride doit être activée.

    Args:
        collection_size: Nombre de chunks dans la collection
        min_threshold: Seuil minimum pour activer l'hybride

    Returns:
        True si la recherche hybride est recommandée
    """
    _log = log or logger
    use_hybrid = collection_size >= min_threshold

    if use_hybrid:
        _log.info(
            f"[HYBRID] Auto-enabled: {collection_size} chunks >= {min_threshold} threshold"
        )
    else:
        _log.debug(
            f"[HYBRID] Disabled: {collection_size} chunks < {min_threshold} threshold"
        )

    return use_hybrid


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalise les scores entre 0 et 1."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_search(
    query: str,
    collection,
    bm25_index: BM25Index,
    embed_func,
    top_k: int = 30,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    log=None
) -> Dict[str, Any]:
    """
    Recherche hybride combinant dense (FAISS) et sparse (BM25).

    La fusion utilise Reciprocal Rank Fusion (RRF) pour combiner les scores.

    Args:
        query: Requête textuelle
        collection: Collection FAISS
        bm25_index: Index BM25 pré-construit
        embed_func: Fonction d'embedding pour la requête
        top_k: Nombre de résultats
        dense_weight: Poids de la recherche dense (0-1)
        sparse_weight: Poids de la recherche sparse (0-1)
        log: Logger

    Returns:
        Résultats fusionnés au format FAISS
    """
    _log = log or logger

    _log.info(f"[HYBRID] Starting hybrid search (dense={dense_weight:.0%}, sparse={sparse_weight:.0%})")

    # 1) Recherche dense (FAISS)
    q_emb = embed_func(query)
    dense_results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k * 2,  # Récupérer plus pour avoir de la marge après fusion
        include=["documents", "metadatas", "distances"],
    )

    # Accès sécurisé aux résultats (évite IndexError si liste vide)
    _dense_docs = dense_results.get("documents", [[]])
    _dense_metas = dense_results.get("metadatas", [[]])
    _dense_dists = dense_results.get("distances", [[]])
    dense_docs = _dense_docs[0] if _dense_docs and _dense_docs[0] else []
    dense_metas = _dense_metas[0] if _dense_metas and _dense_metas[0] else []
    dense_dists = _dense_dists[0] if _dense_dists and _dense_dists[0] else []

    _log.debug(f"[HYBRID] Dense search returned {len(dense_docs)} results")

    # 2) Recherche sparse (BM25)
    bm25_results = bm25_index.search(query, top_k=top_k * 2)

    _log.debug(f"[HYBRID] BM25 search returned {len(bm25_results)} results")

    # 3) Fusion avec Reciprocal Rank Fusion (RRF)
    # RRF score = sum(1 / (k + rank)) pour chaque système
    K = 60  # Constante RRF standard

    # Créer un mapping doc_id -> scores
    doc_scores: Dict[str, Dict[str, Any]] = {}

    # Ajouter les résultats dense
    for rank, (doc, meta, dist) in enumerate(zip(dense_docs, dense_metas, dense_dists)):
        doc_id = meta.get("chunk_id") or str(hash(doc[:100]))

        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                "document": doc,
                "metadata": meta,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": 0,
                "sparse_score": 0,
            }

        doc_scores[doc_id]["dense_rank"] = rank
        # L2 distance pour embeddings normalisés: [0, 2] -> score [1, 0]
        # dist=0 (identique) -> score=1, dist=2 (opposé) -> score=0
        doc_scores[doc_id]["dense_score"] = max(0, 1.0 - dist / 2.0)

    # Récupérer les métadonnées complètes de la collection pour les résultats BM25
    # (permet d'avoir les métadonnées même pour les docs trouvés uniquement par BM25)
    # IMPORTANT: BM25 utilise chunk_id (sans UUID), mais FAISS utilise faiss_id (avec UUID)
    # On doit mapper chunk_id -> metadata
    bm25_doc_ids = [bm25_index.doc_ids[doc_idx] for doc_idx, _ in bm25_results
                   if doc_idx < len(bm25_index.documents)]

    collection_metadata = {}
    if bm25_doc_ids and hasattr(collection, 'get'):
        try:
            all_meta = collection.get(include=["metadatas"])
            all_metas = all_meta.get("metadatas", [])
            # Mapper par chunk_id (pas par faiss_id) car BM25 utilise chunk_id
            for meta in all_metas:
                if meta and "chunk_id" in meta:
                    collection_metadata[str(meta["chunk_id"])] = meta
        except Exception as e:
            _log.debug(f"[HYBRID] Could not fetch metadata from collection: {e}")

    # Ajouter les résultats BM25
    for rank, (doc_idx, bm25_score) in enumerate(bm25_results):
        if doc_idx >= len(bm25_index.documents):
            continue

        doc = bm25_index.documents[doc_idx]
        doc_id = bm25_index.doc_ids[doc_idx]

        if doc_id not in doc_scores:
            # Document trouvé uniquement par BM25 - récupérer les métadonnées complètes
            full_meta = collection_metadata.get(doc_id, {"chunk_id": doc_id})
            doc_scores[doc_id] = {
                "document": doc,
                "metadata": full_meta,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": 0,
                "sparse_score": 0,
            }

        doc_scores[doc_id]["sparse_rank"] = rank
        doc_scores[doc_id]["sparse_score"] = bm25_score

    # Calculer le score RRF final
    for doc_id, data in doc_scores.items():
        rrf_dense = dense_weight / (K + data["dense_rank"]) if data["dense_rank"] is not None else 0
        rrf_sparse = sparse_weight / (K + data["sparse_rank"]) if data["sparse_rank"] is not None else 0
        data["rrf_score"] = rrf_dense + rrf_sparse

        # Bonus si trouvé par les deux systèmes
        if data["dense_rank"] is not None and data["sparse_rank"] is not None:
            data["rrf_score"] *= 1.1  # 10% bonus

    # Trier par score RRF
    sorted_results = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Formater les résultats
    documents = [r["document"] for r in sorted_results[:top_k]]
    metadatas = [r["metadata"] for r in sorted_results[:top_k]]

    # Convertir RRF scores en distances (inverse)
    # Protection contre division par zéro si tous les scores sont 0
    max_rrf = max(r["rrf_score"] for r in sorted_results) if sorted_results else 1.0
    max_rrf = max(max_rrf, 1e-6)  # Évite division par zéro
    distances = [1.0 - (r["rrf_score"] / max_rrf) for r in sorted_results[:top_k]]

    _log.info(
        f"[HYBRID] Fusion complete: {len(documents)} results "
        f"(dense-only: {sum(1 for r in sorted_results[:top_k] if r['sparse_rank'] is None)}, "
        f"sparse-only: {sum(1 for r in sorted_results[:top_k] if r['dense_rank'] is None)}, "
        f"both: {sum(1 for r in sorted_results[:top_k] if r['dense_rank'] is not None and r['sparse_rank'] is not None)})"
    )

    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


# =============================================================================
#  BM25 INDEX MANAGEMENT
# =============================================================================

def get_bm25_index_path(db_path: str, collection_name: str) -> str:
    """Retourne le chemin du fichier d'index BM25 pour une collection."""
    return os.path.join(db_path, collection_name, "bm25_index.pkl")


def build_or_load_bm25_index(
    db_path: str,
    collection_name: str,
    documents: List[str],
    doc_ids: List[str],
    force_rebuild: bool = False,
    log=None
) -> Optional[BM25Index]:
    """
    Charge ou construit l'index BM25 pour une collection.

    Args:
        db_path: Chemin de la base de données
        collection_name: Nom de la collection
        documents: Liste des documents (textes)
        doc_ids: Liste des IDs de documents
        force_rebuild: Force la reconstruction même si l'index existe
        log: Logger

    Returns:
        Index BM25 ou None si erreur
    """
    _log = log or logger

    index_path = get_bm25_index_path(db_path, collection_name)

    # Essayer de charger l'index existant
    if not force_rebuild and os.path.exists(index_path):
        try:
            index = BM25Index.load(index_path, log=_log)

            # Vérifier que l'index est à jour
            if index.n_docs == len(documents):
                _log.info(f"[BM25] Using existing index ({index.n_docs} docs)")
                return index
            else:
                _log.info(
                    f"[BM25] Index outdated ({index.n_docs} vs {len(documents)} docs), rebuilding..."
                )
        except Exception as e:
            _log.warning(f"[BM25] Failed to load index: {e}, rebuilding...")

    # Construire un nouvel index
    try:
        index = BM25Index(log=_log)
        index.build(documents, doc_ids)

        # Sauvegarder l'index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        index.save(index_path)

        return index

    except Exception as e:
        _log.error(f"[BM25] Failed to build index: {e}")
        return None


def get_collection_documents(collection) -> Tuple[List[str], List[str]]:
    """
    Extrait tous les documents et IDs d'une collection FAISS.

    Returns:
        (documents, doc_ids)
    """
    # Récupérer toutes les données de la collection
    all_data = collection.get(include=["documents", "metadatas"])

    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    doc_ids = []
    for i, meta in enumerate(metadatas):
        if meta and "chunk_id" in meta:
            doc_ids.append(str(meta["chunk_id"]))
        else:
            doc_ids.append(str(i))

    return documents, doc_ids
