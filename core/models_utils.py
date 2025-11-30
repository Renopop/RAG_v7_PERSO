# models_utils.py
"""
Module de gestion des modeles et APIs pour le RAG.

Supporte deux modes:
- Mode ONLINE: Utilise les APIs distantes (DALLEM, Snowflake)
- Mode OFFLINE: Utilise les modeles locaux (Mistral, BGE-M3)

Le mode est determine automatiquement via config_manager.is_offline_mode()
"""
import os
import sys
import math
import time
import traceback
from typing import List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import httpx
import logging
from logging import Logger

from openai import OpenAI
import openai

# Import du gestionnaire de configuration pour le mode offline
try:
    from core.config_manager import is_offline_mode, get_effective_paths
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    def is_offline_mode():
        return False
    def get_effective_paths():
        return {}

# Import des modeles offline (lazy import pour eviter les dependances)
OFFLINE_MODELS_AVAILABLE = False
try:
    from core.offline_models import (
        embed_in_batches_offline,
        call_llm_offline,
        rerank_offline,
        get_offline_status,
        get_offline_embeddings,
        get_offline_llm,
    )
    OFFLINE_MODELS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------
#  CONFIG réseau / modèles
# ---------------------------------------------------------------------

LLM_MODEL = "dallem-val"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 32  # taille batch embeddings (équilibre performance/sécurité)
MAX_CHARS_PER_TEXT = 28000  # ~7000 tokens max par texte (limite Snowflake: 8192 tokens)
PARALLEL_EMBEDDING_WORKERS = 8  # Workers parallèles pour appels API (I/O bound, pas CPU)

HARDCODE = {
    "DALLEM_API_BASE": "https://api.dev.dassault-aviation.pro/dallem-pilote/v1",
    "SNOWFLAKE_API_BASE": "https://api.dev.dassault-aviation.pro/snowflake-arctic-embed-l-v2.0/v1",
    "DALLEM_API_KEY": "EMPTY",     # à surcharger par l'env
    "SNOWFLAKE_API_KEY": "token",  # à surcharger par l'env
    "DISABLE_SSL_VERIFY": "false",  # SSL activé par défaut pour la sécurité
}

DALLEM_API_BASE = os.getenv("DALLEM_API_BASE", HARDCODE["DALLEM_API_BASE"]).rstrip("/")
SNOWFLAKE_API_BASE = os.getenv("SNOWFLAKE_API_BASE", HARDCODE["SNOWFLAKE_API_BASE"]).rstrip("/")
DALLEM_API_KEY = os.getenv("DALLEM_API_KEY", HARDCODE["DALLEM_API_KEY"])
SNOWFLAKE_API_KEY = os.getenv("SNOWFLAKE_API_KEY", HARDCODE["SNOWFLAKE_API_KEY"])

VERIFY_SSL = not (
    os.getenv("DISABLE_SSL_VERIFY", HARDCODE["DISABLE_SSL_VERIFY"])
    .lower()
    in ("1", "true", "yes", "on")
)


def _mask(s: Optional[str]) -> str:
    if not s:
        return "<vide>"
    if len(s) <= 6:
        return "***"
    return s[:3] + "…" + s[-3:]


def make_logger(debug: bool) -> Logger:
    log = logging.getLogger("rag_da")

    # Choix des niveaux : console silencieuse en mode non-debug
    if debug:
        level_console = logging.DEBUG
        level_logger = logging.DEBUG
    else:
        level_console = logging.WARNING
        level_logger = logging.WARNING

    log.setLevel(level_logger)

    # Si le logger a déjà des handlers, on met juste à jour leurs niveaux
    if log.handlers:
        for h in log.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level_console)
        return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console)
    ch.setFormatter(fmt)

    # Fichier : on garde tout en DEBUG pour analyse détaillée
    fh = logging.FileHandler("rag_da_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)

    # Ces logs seront visibles au moins dans le fichier
    log.info("=== Configuration RAG_DA ===")
    log.info(f"SNOWFLAKE_API_BASE = {SNOWFLAKE_API_BASE}")
    log.info(f"DALLEM_API_BASE    = {DALLEM_API_BASE}")
    log.info(f"VERIFY_SSL         = {VERIFY_SSL}")
    log.info(f"EMBED_MODEL        = {EMBED_MODEL}")
    log.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    log.info(
        "API_KEYS           = snowflake={} | dallem={}".format(
            _mask(SNOWFLAKE_API_KEY),
            _mask(DALLEM_API_KEY),
        )
    )
    return log


def create_http_client() -> httpx.Client:
    """
    Client HTTP configuré (timeout, SSL) pour Snowflake + DALLEM.
    """
    return httpx.Client(
        verify=VERIFY_SSL,
        timeout=httpx.Timeout(300.0),
    )


# ---------------------------------------------------------------------
#  Client embeddings Snowflake (OpenAI v1-compatible)
# ---------------------------------------------------------------------


class DirectOpenAIEmbeddings:
    """
    Client embeddings minimal (OpenAI v1-compatible).
    role_prefix=True -> "passage:" / "query:".
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        http_client: Optional[httpx.Client] = None,
        role_prefix: bool = True,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.role_prefix = role_prefix
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.log = logger or logging.getLogger("rag_da")

    def _apply_prefix(self, items: List[str], role: str) -> List[str]:
        if not self.role_prefix:
            return items
        pref = "query: " if role == "query" else "passage: "
        return [pref + (x or "") for x in items]

    def _retry_request(self, func, max_retries: int = 5, base_delay: float = 1.0):
        """
        Exécute func() avec retry exponentiel adapté au type d'erreur.

        - RateLimitError: backoff agressif (facteur 4x) car l'API est surchargée
        - APIConnectionError: backoff standard (facteur 2x) pour erreurs réseau
        - APIError: backoff standard pour autres erreurs API
        """
        for attempt in range(max_retries):
            try:
                return func()
            except openai.RateLimitError as e:
                # Backoff plus agressif pour RateLimit (facteur 4x au lieu de 2x)
                if attempt == max_retries - 1:
                    self.log.error(
                        f"[embeddings] Échec après {max_retries} tentatives — RateLimitError: {e}"
                    )
                    raise
                wait_time = base_delay * (4 ** attempt)  # 1s, 4s, 16s, 64s...
                self.log.warning(
                    f"[embeddings] RateLimit - tentative {attempt + 1}/{max_retries} "
                    f"— retry dans {wait_time:.1f}s (backoff agressif)"
                )
                time.sleep(wait_time)
            except (openai.APIConnectionError, openai.APIError) as e:
                # Backoff standard pour erreurs réseau/API
                if attempt == max_retries - 1:
                    self.log.error(
                        f"[embeddings] Échec après {max_retries} tentatives — {type(e).__name__}: {e}"
                    )
                    raise
                wait_time = base_delay * (2 ** attempt)  # 1s, 2s, 4s, 8s...
                self.log.warning(
                    f"[embeddings] Tentative {attempt + 1}/{max_retries} échouée "
                    f"({type(e).__name__}: {e}) — retry dans {wait_time:.1f}s"
                )
                time.sleep(wait_time)

    def _create_embeddings(self, inputs: List[str]) -> List[List[float]]:
        t0 = time.time()
        self.log.debug(
            f"[embeddings] POST {self.client.base_url} | model={self.model} "
            f"| n_inputs={len(inputs)} | len0={len(inputs[0]) if inputs else 0}"
        )

        def _do_request():
            return self.client.embeddings.create(model=self.model, input=inputs)

        try:
            resp = self._retry_request(_do_request)
            dur = (time.time() - t0) * 1000
            self.log.debug(
                f"[embeddings] OK in {dur:.1f} ms | items={len(resp.data)} "
                f"| dim≈{len(resp.data[0].embedding) if resp.data else 'n/a'}"
            )
            return [d.embedding for d in resp.data]

        except openai.NotFoundError as e:
            self.log.error(f"[embeddings] NotFoundError (modèle='{self.model}' ?) : {e}")
            self.log.debug(traceback.format_exc())
            raise
        except openai.AuthenticationError as e:
            self.log.error("[embeddings] AuthenticationError — clé invalide ?")
            self.log.debug(traceback.format_exc())
            raise
        except Exception as e:
            self.log.error(f"[embeddings] Exception — {e}")
            self.log.debug(traceback.format_exc())
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="passage")
        return self._create_embeddings(inputs)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="query")
        return self._create_embeddings(inputs)


def _embed_single_batch(
    batch_info: Tuple[int, List[str]],
    role: str,
    emb_client: DirectOpenAIEmbeddings,
    dry_run: bool,
) -> Tuple[int, List[List[float]]]:
    """
    Embed un seul batch de textes (fonction worker pour le parallélisme).
    Retourne (batch_index, embeddings).
    """
    batch_idx, chunk = batch_info
    if dry_run:
        dim = 1024
        fake = np.random.rand(len(chunk), dim).astype(np.float32) - 0.5
        return (batch_idx, fake.tolist())
    else:
        if role == "query":
            return (batch_idx, emb_client.embed_queries(chunk))
        else:
            return (batch_idx, emb_client.embed_documents(chunk))


def _embed_sequential(
    batches: List[Tuple[int, List[str]]],
    role: str,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool,
) -> List[List[float]]:
    """
    Méthode séquentielle d'embedding (fallback).
    """
    out: List[List[float]] = []
    total_batches = len(batches)

    for batch_idx, chunk in batches:
        log.debug(
            f"[emb-seq] batch {batch_idx + 1}/{total_batches} "
            f"| size={len(chunk)}"
        )
        try:
            _, embeddings = _embed_single_batch((batch_idx, chunk), role, emb_client, dry_run)
            out.extend(embeddings)
        except Exception as e:
            log.error(f"[emb-seq] échec batch {batch_idx} — {e}")
            raise

    return out


def _embed_parallel(
    batches: List[Tuple[int, List[str]]],
    role: str,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool,
    max_workers: int,
) -> List[List[float]]:
    """
    Méthode parallèle d'embedding avec ThreadPoolExecutor.
    """
    total_batches = len(batches)
    results: dict = {}  # batch_idx -> embeddings

    log.info(f"[emb-parallel] Démarrage avec {max_workers} workers pour {total_batches} batches")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les batches
        future_to_batch = {
            executor.submit(_embed_single_batch, batch, role, emb_client, dry_run): batch[0]
            for batch in batches
        }

        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                idx, embeddings = future.result()
                results[idx] = embeddings
                completed += 1
                log.debug(f"[emb-parallel] batch {idx + 1}/{total_batches} terminé ({completed}/{total_batches})")
            except Exception as e:
                log.error(f"[emb-parallel] échec batch {batch_idx} — {e}")
                raise

    # Reconstruire la liste ordonnée
    out: List[List[float]] = []
    for i in range(total_batches):
        out.extend(results[i])

    return out


def embed_in_batches(
    texts: List[str],
    role: str,
    batch_size: int,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool = False,
    use_parallel: bool = True,
    force_offline: bool = False,
) -> np.ndarray:
    """
    Découpe en batches, appelle le client embeddings, normalise les vecteurs (L2).
    Tronque automatiquement les textes trop longs pour éviter les erreurs de tokens.

    Supporte le mode offline automatique via config_manager.

    Args:
        texts: Liste de textes a encoder
        role: "query" ou "passage"
        batch_size: Taille des batches
        emb_client: Client embeddings (ignore en mode offline)
        log: Logger
        dry_run: Si True, genere des embeddings aleatoires
        use_parallel: Si True, utilise le traitement parallèle (multicoeur).
                     Si erreur, fallback automatique sur séquentiel.
        force_offline: Si True, force le mode offline

    Returns:
        Array numpy d'embeddings normalises (n_texts, dimension)
    """
    # Verifier si on est en mode offline
    offline_mode = force_offline or (CONFIG_MANAGER_AVAILABLE and is_offline_mode())

    if offline_mode and OFFLINE_MODELS_AVAILABLE:
        log.info(f"[emb] Mode OFFLINE actif - utilisation BGE-M3 local")
        return embed_in_batches_offline(
            texts=texts,
            role=role,
            batch_size=batch_size,
            log=log,
        )

    # Mode online (API Snowflake)
    # Tronquer les textes trop longs (limite Snowflake: 8192 tokens ≈ 28000 chars)
    truncated_count = 0
    safe_texts = []
    for t in texts:
        if len(t) > MAX_CHARS_PER_TEXT:
            safe_texts.append(t[:MAX_CHARS_PER_TEXT])
            truncated_count += 1
        else:
            safe_texts.append(t)

    if truncated_count > 0:
        log.warning(f"[emb] {truncated_count} texte(s) tronqué(s) à {MAX_CHARS_PER_TEXT} caractères")

    n = len(safe_texts)

    # Préparer les batches
    batches: List[Tuple[int, List[str]]] = []
    batch_idx = 0
    for i in range(0, n, batch_size):
        chunk = safe_texts[i: i + batch_size]
        batches.append((batch_idx, chunk))
        batch_idx += 1

    total_batches = len(batches)
    mode = "parallel" if use_parallel and total_batches > 1 else "sequential"
    log.info(
        f"[emb] start role={role} | n={n} | batch_size={batch_size} | "
        f"batches={total_batches} | mode={mode} | workers={PARALLEL_EMBEDDING_WORKERS} | dry_run={dry_run}"
    )

    out: List[List[float]] = []

    # Essayer le mode parallèle si activé et plusieurs batches
    if use_parallel and total_batches > 1:
        try:
            out = _embed_parallel(batches, role, emb_client, log, dry_run, PARALLEL_EMBEDDING_WORKERS)
            log.info(f"[emb] Mode parallèle OK ({total_batches} batches, {PARALLEL_EMBEDDING_WORKERS} workers)")
        except Exception as e:
            log.warning(f"[emb] Mode parallèle échoué, fallback séquentiel: {e}")
            out = _embed_sequential(batches, role, emb_client, log, dry_run)
    else:
        # Mode séquentiel direct
        out = _embed_sequential(batches, role, emb_client, log, dry_run)

    M = np.asarray(out, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] != n:
        log.error(f"[emb] shape inattendue: {M.shape} (attendu ({n}, d))")

    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    if np.any(np.isnan(denom)):
        log.warning("[emb] NaN détecté dans la norme, correction appliquée.")
        denom = np.nan_to_num(denom, nan=1.0)
    M = M / denom
    log.info(
        f"[emb] terminé | shape={M.shape} | d={M.shape[1] if M.ndim == 2 else 'n/a'}"
    )
    return M


# ---------------------------------------------------------------------
#  Appel LLM DALLEM
# ---------------------------------------------------------------------


def call_dallem_chat(
    http_client: httpx.Client,
    question: str,
    context: str,
    log: Logger,
    force_offline: bool = False,
) -> str:
    """
    Appel au LLM pour generer une reponse RAG.

    En mode offline, utilise Mistral-7B local.
    En mode online, utilise l'API DALLEM.

    Args:
        http_client: Client HTTP (ignore en mode offline)
        question: Question de l'utilisateur
        context: Contexte documentaire
        log: Logger
        force_offline: Si True, force le mode offline

    Returns:
        Reponse generee
    """
    # Verifier si on est en mode offline
    offline_mode = force_offline or (CONFIG_MANAGER_AVAILABLE and is_offline_mode())

    if offline_mode and OFFLINE_MODELS_AVAILABLE:
        log.info("[RAG] Mode OFFLINE actif - utilisation Mistral-7B local")
        return call_llm_offline(
            question=question,
            context=context,
            log=log,
        )

    # Mode online (API DALLEM)
    if not DALLEM_API_KEY or DALLEM_API_KEY in ("toto", "EMPTY"):
        raise RuntimeError("DALLEM_API_KEY manquant ou de test. Impossible d'utiliser le LLM.")

    system_msg = (
        "Tu es un assistant expert en réglementation aéronautique EASA. "
        "Tu maîtrises parfaitement les documents de certification:\n"
        "- CS (Certification Specifications): exigences réglementaires obligatoires\n"
        "- AMC (Acceptable Means of Compliance): moyens acceptables pour démontrer la conformité\n"
        "- GM (Guidance Material): explications et interprétations non contraignantes\n"
        "Tu dois répondre en te basant UNIQUEMENT sur le CONTEXTE fourni."
    )

    import textwrap
    user_msg = textwrap.dedent(f"""
    === CONTEXTE DOCUMENTAIRE ===
    {context}
    === FIN DU CONTEXTE ===

    QUESTION : {question}

    INSTRUCTIONS :
    - Réponds dans la même langue que la question (français si question en français, anglais sinon)
    - Cite TOUJOURS les références exactes (ex: "CS 25.613", "AMC1 25.1309") présentes dans le contexte
    - Distingue clairement ce qui est une exigence (CS/shall) de ce qui est un moyen de conformité (AMC) ou une guidance (GM)
    - Pour les termes "shall", "must", "may", "should": précise le niveau d'obligation
    - Si le contexte ne contient AUCUNE information pertinente, réponds : "Je n'ai pas l'information pour répondre à cette question."
    """)

    url = DALLEM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {DALLEM_API_KEY}",
        "Content-Type": "application/json",
    }

    # Log du contexte pour diagnostic
    log.info(f"[RAG] Contexte: {len(context)} chars, {context.count('[source=')} sources")
    if not context.strip():
        log.warning("[RAG] ⚠️ CONTEXTE VIDE - pas de chunks trouvés!")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 2000,
        "temperature": 0.3,
    }

    log.info("[RAG] Appel DALLEM /chat/completions pour réponse RAG")

    # Retry logic avec backoff exponentiel
    max_retries = 4
    base_delay = 2  # secondes

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = http_client.post(url, headers=headers, json=payload, timeout=180.0)
            resp.raise_for_status()
            data = resp.json()

            # Vérifier que la réponse contient bien du contenu
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError("Réponse LLM vide")

            log.info(f"[RAG] ✅ Réponse DALLEM reçue (attempt {attempt + 1}/{max_retries})")
            return content

        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16 secondes

            if attempt < max_retries - 1:
                log.warning(f"[RAG] ⚠️ Erreur DALLEM (attempt {attempt + 1}/{max_retries}): {e}")
                log.info(f"[RAG] Retry dans {delay}s...")
                time.sleep(delay)
            else:
                log.error(f"[RAG] ❌ Échec DALLEM après {max_retries} tentatives: {e}")

    # Toutes les tentatives ont échoué - retourner un message d'erreur spécial
    error_msg = (
        "⚠️ **ERREUR DE COMMUNICATION AVEC LE LLM**\n\n"
        f"Le serveur n'a pas pu répondre après {max_retries} tentatives.\n\n"
        f"**Erreur technique:** {str(last_error)[:200]}\n\n"
        "👉 **Veuillez reposer votre question** ou réessayer dans quelques instants."
    )
    return error_msg


# ---------------------------------------------------------------------
#  HELPERS POUR LE MODE OFFLINE
# ---------------------------------------------------------------------

def get_current_mode() -> str:
    """
    Retourne le mode actuel (online ou offline).

    Returns:
        "online" ou "offline"
    """
    if CONFIG_MANAGER_AVAILABLE and is_offline_mode():
        return "offline"
    return "online"


def is_offline_available() -> bool:
    """
    Verifie si le mode offline est disponible (modeles installes).

    Returns:
        True si le mode offline est disponible
    """
    return OFFLINE_MODELS_AVAILABLE


def get_models_status() -> dict:
    """
    Retourne le statut des modeles (online et offline).

    Returns:
        Dict avec informations sur les modeles disponibles
    """
    status = {
        "current_mode": get_current_mode(),
        "online": {
            "available": True,
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL,
            "api_base_llm": DALLEM_API_BASE,
            "api_base_embed": SNOWFLAKE_API_BASE,
        },
        "offline": {
            "available": OFFLINE_MODELS_AVAILABLE,
            "models": {},
        },
    }

    if OFFLINE_MODELS_AVAILABLE:
        try:
            offline_status = get_offline_status()
            status["offline"]["models"] = offline_status.get("models", {})
            status["offline"]["gpu"] = offline_status.get("gpu", {})
        except Exception:
            pass

    return status


def create_embeddings_client(
    log: Optional[Logger] = None,
    force_offline: bool = False,
):
    """
    Cree un client embeddings selon le mode actuel.

    En mode offline, retourne un wrapper pour OfflineEmbeddings.
    En mode online, retourne DirectOpenAIEmbeddings.

    Args:
        log: Logger optionnel
        force_offline: Si True, force le mode offline

    Returns:
        Client embeddings
    """
    _log = log or logging.getLogger("rag_da")

    offline_mode = force_offline or (CONFIG_MANAGER_AVAILABLE and is_offline_mode())

    if offline_mode and OFFLINE_MODELS_AVAILABLE:
        _log.info("[EMB] Creation client embeddings OFFLINE (BGE-M3)")
        return get_offline_embeddings(log=_log)

    _log.info("[EMB] Creation client embeddings ONLINE (Snowflake)")
    http_client = create_http_client()
    return DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=_log,
    )


def create_llm_client(
    log: Optional[Logger] = None,
    force_offline: bool = False,
):
    """
    Cree un client LLM selon le mode actuel.

    En mode offline, retourne OfflineLLM.
    En mode online, retourne None (utiliser call_dallem_chat directement).

    Args:
        log: Logger optionnel
        force_offline: Si True, force le mode offline

    Returns:
        Client LLM ou None
    """
    _log = log or logging.getLogger("rag_da")

    offline_mode = force_offline or (CONFIG_MANAGER_AVAILABLE and is_offline_mode())

    if offline_mode and OFFLINE_MODELS_AVAILABLE:
        _log.info("[LLM] Creation client LLM OFFLINE (Mistral-7B)")
        return get_offline_llm(log=_log)

    _log.info("[LLM] Mode ONLINE - utiliser call_dallem_chat()")
    return None
