# feedback_store.py
# Module de gestion des retours utilisateurs pour le RAG
# Stockage JSON avec feedbacks granulaires et statistiques

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import getpass


@dataclass
class SourceFeedback:
    """Feedback sur une source individuelle."""
    source_file: str
    chunk_id: str
    relevance_score: int  # 1-5 (1=non pertinent, 5=très pertinent)
    is_correct: Optional[bool] = None  # Contenu correct/incorrect
    comment: str = ""


@dataclass
class QueryFeedback:
    """Feedback simplifié sur une requête RAG."""
    feedback_id: str
    timestamp: str
    user: str

    # Contexte de la requête
    base_name: str
    collection_name: str
    question: str

    # Évaluation simplifiée de la réponse
    is_positive: bool  # True = pouce haut, False = pouce bas

    # Réponse attendue (pour améliorer les futures recherches)
    suggested_answer: str = ""

    # Métadonnées
    answer_text: str = ""  # Réponse générée (pour référence)
    top_k_used: int = 10
    synthesize_all: bool = False

    # Champs legacy pour compatibilité avec anciens feedbacks
    answer_quality: int = 0
    answer_completeness: int = 0
    answer_accuracy: int = 0
    sources_feedback: List[Dict] = field(default_factory=list)
    general_comment: str = ""


class FeedbackStore:
    """Gestionnaire de stockage des feedbacks utilisateurs."""

    def __init__(self, feedback_dir: str):
        """
        Initialise le store de feedbacks.

        Args:
            feedback_dir: Répertoire racine pour stocker les fichiers JSON de feedback
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def _get_feedback_file(self, base_name: str) -> Path:
        """Retourne le chemin du fichier de feedback pour une base donnée."""
        return self.feedback_dir / f"feedbacks_{base_name}.json"

    def _load_feedbacks(self, base_name: str) -> List[Dict]:
        """Charge les feedbacks existants pour une base."""
        filepath = self._get_feedback_file(base_name)
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_feedbacks(self, base_name: str, feedbacks: List[Dict]) -> None:
        """Sauvegarde les feedbacks pour une base."""
        filepath = self._get_feedback_file(base_name)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)

    def add_feedback(self, feedback: QueryFeedback) -> str:
        """
        Ajoute un nouveau feedback.

        Args:
            feedback: L'objet QueryFeedback à enregistrer

        Returns:
            L'ID du feedback créé
        """
        feedbacks = self._load_feedbacks(feedback.base_name)
        feedback_dict = asdict(feedback)
        feedbacks.append(feedback_dict)
        self._save_feedbacks(feedback.base_name, feedbacks)
        return feedback.feedback_id

    def get_feedbacks_for_base(self, base_name: str) -> List[Dict]:
        """Récupère tous les feedbacks pour une base."""
        return self._load_feedbacks(base_name)

    def get_all_feedbacks(self) -> Dict[str, List[Dict]]:
        """Récupère tous les feedbacks de toutes les bases."""
        all_feedbacks = {}
        for filepath in self.feedback_dir.glob("feedbacks_*.json"):
            base_name = filepath.stem.replace("feedbacks_", "")
            all_feedbacks[base_name] = self._load_feedbacks(base_name)
        return all_feedbacks

    def get_statistics(self, base_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule les statistiques des feedbacks (format simplifié).

        Args:
            base_name: Nom de la base (None pour toutes les bases)

        Returns:
            Dictionnaire de statistiques
        """
        if base_name:
            feedbacks = self._load_feedbacks(base_name)
            bases_data = {base_name: feedbacks}
        else:
            bases_data = self.get_all_feedbacks()

        total_feedbacks = 0
        positive_feedbacks = 0
        negative_feedbacks = 0

        negative_questions = []
        collection_stats = {}
        user_stats = {}
        daily_stats = {}

        for base, feedbacks in bases_data.items():
            for fb in feedbacks:
                total_feedbacks += 1

                # Nouveau format simplifié (pouce haut/bas)
                is_positive = fb.get("is_positive")

                # Compatibilité avec ancien format (scores 1-5)
                if is_positive is None:
                    # Ancien format : calculer si positif à partir des scores
                    quality = fb.get("answer_quality", 0)
                    completeness = fb.get("answer_completeness", 0)
                    accuracy = fb.get("answer_accuracy", 0)
                    if quality + completeness + accuracy > 0:
                        avg_score = (quality + completeness + accuracy) / 3
                        is_positive = avg_score >= 3
                    else:
                        is_positive = True  # Par défaut positif si pas de données

                if is_positive:
                    positive_feedbacks += 1
                else:
                    negative_feedbacks += 1
                    # Stocker les questions avec feedback négatif
                    negative_questions.append({
                        "base": base,
                        "collection": fb.get("collection_name", ""),
                        "question": fb.get("question", ""),
                        "timestamp": fb.get("timestamp", ""),
                        "suggested_answer": fb.get("suggested_answer", "")
                    })

                # Stats par collection
                coll = fb.get("collection_name", "unknown")
                if coll not in collection_stats:
                    collection_stats[coll] = {
                        "count": 0,
                        "positive": 0,
                        "negative": 0
                    }
                collection_stats[coll]["count"] += 1
                if is_positive:
                    collection_stats[coll]["positive"] += 1
                else:
                    collection_stats[coll]["negative"] += 1

                # Stats par utilisateur
                user = fb.get("user", "unknown")
                if user not in user_stats:
                    user_stats[user] = 0
                user_stats[user] += 1

                # Stats journalières
                timestamp = fb.get("timestamp", "")
                if timestamp:
                    day = timestamp[:10]  # YYYY-MM-DD
                    if day not in daily_stats:
                        daily_stats[day] = {"positive": 0, "negative": 0}
                    if is_positive:
                        daily_stats[day]["positive"] += 1
                    else:
                        daily_stats[day]["negative"] += 1

        # Calcul du taux de satisfaction
        satisfaction_rate = round(positive_feedbacks / total_feedbacks * 100, 1) if total_feedbacks > 0 else 0

        # Taux par collection
        for coll, stats in collection_stats.items():
            if stats["count"] > 0:
                stats["satisfaction_rate"] = round(stats["positive"] / stats["count"] * 100, 1)

        return {
            "total_feedbacks": total_feedbacks,
            "positive_feedbacks": positive_feedbacks,
            "negative_feedbacks": negative_feedbacks,
            "satisfaction_rate": satisfaction_rate,
            "collection_stats": collection_stats,
            "user_stats": user_stats,
            "daily_stats": daily_stats,
            "negative_questions": negative_questions[:20],  # Top 20
        }

    def get_feedback_trends(self, base_name: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Calcule les tendances des feedbacks sur une période (format simplifié).

        Args:
            base_name: Nom de la base (None pour toutes)
            days: Nombre de jours à analyser

        Returns:
            Données de tendance
        """
        from datetime import datetime, timedelta

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        if base_name:
            feedbacks = self._load_feedbacks(base_name)
        else:
            feedbacks = []
            for fb_list in self.get_all_feedbacks().values():
                feedbacks.extend(fb_list)

        # Filtrer par date
        recent_feedbacks = [
            fb for fb in feedbacks
            if fb.get("timestamp", "") >= cutoff_date
        ]

        # Calculer l'évolution par jour
        daily_data = {}
        for fb in recent_feedbacks:
            timestamp = fb.get("timestamp", "")
            if timestamp:
                day = timestamp[:10]
                if day not in daily_data:
                    daily_data[day] = {"positive": 0, "negative": 0}

                # Nouveau format
                is_positive = fb.get("is_positive")
                # Compatibilité ancien format
                if is_positive is None:
                    quality = fb.get("answer_quality", 0)
                    completeness = fb.get("answer_completeness", 0)
                    accuracy = fb.get("answer_accuracy", 0)
                    if quality + completeness + accuracy > 0:
                        is_positive = (quality + completeness + accuracy) / 3 >= 3
                    else:
                        is_positive = True

                if is_positive:
                    daily_data[day]["positive"] += 1
                else:
                    daily_data[day]["negative"] += 1

        # Données par jour
        trend_data = []
        for day in sorted(daily_data.keys()):
            data = daily_data[day]
            total = data["positive"] + data["negative"]
            trend_data.append({
                "date": day,
                "positive": data["positive"],
                "negative": data["negative"],
                "total": total,
                "satisfaction_rate": round(data["positive"] / total * 100, 1) if total > 0 else 0
            })

        return {
            "period_days": days,
            "total_feedbacks_in_period": len(recent_feedbacks),
            "trend_data": trend_data
        }

    def export_feedbacks_csv(self, base_name: Optional[str] = None) -> str:
        """
        Exporte les feedbacks au format CSV.

        Args:
            base_name: Nom de la base (None pour toutes)

        Returns:
            Contenu CSV
        """
        import csv
        import io

        if base_name:
            feedbacks = self._load_feedbacks(base_name)
        else:
            feedbacks = []
            for fb_list in self.get_all_feedbacks().values():
                feedbacks.extend(fb_list)

        output = io.StringIO()
        writer = csv.writer(output, delimiter=";")

        # En-tête (format simplifié)
        writer.writerow([
            "feedback_id", "timestamp", "user", "base_name", "collection_name",
            "question", "is_positive", "suggested_answer"
        ])

        for fb in feedbacks:
            # Compatibilité ancien format
            is_positive = fb.get("is_positive")
            if is_positive is None:
                quality = fb.get("answer_quality", 0)
                completeness = fb.get("answer_completeness", 0)
                accuracy = fb.get("answer_accuracy", 0)
                if quality + completeness + accuracy > 0:
                    is_positive = (quality + completeness + accuracy) / 3 >= 3
                else:
                    is_positive = True

            writer.writerow([
                fb.get("feedback_id", ""),
                fb.get("timestamp", ""),
                fb.get("user", ""),
                fb.get("base_name", ""),
                fb.get("collection_name", ""),
                fb.get("question", ""),
                "👍" if is_positive else "👎",
                fb.get("suggested_answer", "")
            ])

        return output.getvalue()

    def delete_feedback(self, base_name: str, feedback_id: str) -> bool:
        """
        Supprime un feedback par son ID.

        Args:
            base_name: Nom de la base
            feedback_id: ID du feedback à supprimer

        Returns:
            True si supprimé, False sinon
        """
        feedbacks = self._load_feedbacks(base_name)
        initial_count = len(feedbacks)
        feedbacks = [fb for fb in feedbacks if fb.get("feedback_id") != feedback_id]

        if len(feedbacks) < initial_count:
            self._save_feedbacks(base_name, feedbacks)
            return True
        return False

    # ========================================================================
    #   MÉTHODES POUR LE RE-RANKING BASÉ SUR LES FEEDBACKS
    # ========================================================================

    def get_source_relevance_scores(self, base_name: str, collection_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calcule les scores moyens de pertinence par source (fichier).

        Args:
            base_name: Nom de la base
            collection_name: Nom de la collection (optionnel, None pour toutes)

        Returns:
            Dict {source_file: {avg_relevance, count, boost_factor}}
        """
        feedbacks = self._load_feedbacks(base_name)
        source_scores: Dict[str, List[int]] = {}

        for fb in feedbacks:
            # Filtrer par collection si spécifié
            if collection_name and fb.get("collection_name") != collection_name:
                continue

            for src_fb in fb.get("sources_feedback", []):
                source_file = src_fb.get("source_file", "")
                relevance = src_fb.get("relevance_score", 0)

                if source_file and relevance > 0:
                    if source_file not in source_scores:
                        source_scores[source_file] = []
                    source_scores[source_file].append(relevance)

        # Calculer les moyennes et facteurs de boost
        result = {}
        for source_file, scores in source_scores.items():
            if not scores:  # Protection défensive
                continue
            avg_relevance = sum(scores) / len(scores)
            count = len(scores)

            # Facteur de boost : -0.2 (score=1) à +0.2 (score=5)
            # Formule : (avg - 3) / 10 => range [-0.2, +0.2]
            boost_factor = (avg_relevance - 3) / 10

            result[source_file] = {
                "avg_relevance": round(avg_relevance, 2),
                "count": count,
                "boost_factor": round(boost_factor, 3)
            }

        return result

    def get_chunk_relevance_scores(self, base_name: str, collection_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calcule les scores de pertinence par chunk_id (plus précis que par source).

        Args:
            base_name: Nom de la base
            collection_name: Nom de la collection (optionnel)

        Returns:
            Dict {chunk_id: {avg_relevance, count, boost_factor}}
        """
        feedbacks = self._load_feedbacks(base_name)
        chunk_scores: Dict[str, List[int]] = {}

        for fb in feedbacks:
            if collection_name and fb.get("collection_name") != collection_name:
                continue

            for src_fb in fb.get("sources_feedback", []):
                chunk_id = src_fb.get("chunk_id", "")
                relevance = src_fb.get("relevance_score", 0)

                if chunk_id and relevance > 0:
                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = []
                    chunk_scores[chunk_id].append(relevance)

        result = {}
        for chunk_id, scores in chunk_scores.items():
            if not scores:  # Protection défensive
                continue
            avg_relevance = sum(scores) / len(scores)
            count = len(scores)
            boost_factor = (avg_relevance - 3) / 10

            result[chunk_id] = {
                "avg_relevance": round(avg_relevance, 2),
                "count": count,
                "boost_factor": round(boost_factor, 3)
            }

        return result

    def find_similar_questions(
        self,
        question: str,
        base_name: str,
        collection_name: Optional[str] = None,
        similarity_threshold: float = 0.6,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Trouve des questions similaires dans l'historique des feedbacks.
        Utilise une similarité basée sur les mots-clés (Jaccard).

        Args:
            question: La question à comparer
            base_name: Nom de la base
            collection_name: Nom de la collection (optionnel)
            similarity_threshold: Seuil de similarité (0-1)
            max_results: Nombre maximum de résultats

        Returns:
            Liste de feedbacks similaires avec leur score de similarité
        """
        import re

        def tokenize(text: str) -> set:
            """Tokenise le texte en mots normalisés."""
            text = text.lower()
            # Garder uniquement les mots alphanumériques
            words = re.findall(r'\b[a-zàâäéèêëïîôùûüç0-9]+\b', text)
            # Filtrer les mots trop courts (< 3 caractères)
            return set(w for w in words if len(w) >= 3)

        def jaccard_similarity(set1: set, set2: set) -> float:
            """Calcule la similarité de Jaccard entre deux ensembles."""
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        question_tokens = tokenize(question)
        if not question_tokens:
            return []

        feedbacks = self._load_feedbacks(base_name)
        similar_questions = []

        for fb in feedbacks:
            # Filtrer par collection si spécifié
            if collection_name and fb.get("collection_name") != collection_name:
                continue

            fb_question = fb.get("question", "")
            fb_tokens = tokenize(fb_question)

            similarity = jaccard_similarity(question_tokens, fb_tokens)

            if similarity >= similarity_threshold:
                # Nouveau format simplifié
                is_positive = fb.get("is_positive")
                # Compatibilité ancien format
                if is_positive is None:
                    quality = fb.get("answer_quality", 0)
                    completeness = fb.get("answer_completeness", 0)
                    accuracy = fb.get("answer_accuracy", 0)
                    if quality + completeness + accuracy > 0:
                        is_positive = (quality + completeness + accuracy) / 3 >= 3
                    else:
                        is_positive = True

                similar_questions.append({
                    "question": fb_question,
                    "similarity": round(similarity, 3),
                    "is_positive": is_positive,
                    "suggested_answer": fb.get("suggested_answer", ""),
                    "collection_name": fb.get("collection_name", ""),
                    "timestamp": fb.get("timestamp", "")
                })

        # Trier par similarité décroissante
        similar_questions.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_questions[:max_results]

    def compute_reranking_factors(
        self,
        sources: List[Dict],
        base_name: str,
        collection_name: Optional[str] = None,
        question: Optional[str] = None,
        alpha: float = 0.3
    ) -> List[Dict]:
        """
        Calcule les facteurs de re-ranking pour une liste de sources.
        Version simplifiée basée sur les feedbacks positifs/négatifs.

        Args:
            sources: Liste des sources retournées par la recherche RAG
            base_name: Nom de la base
            collection_name: Nom de la collection
            question: Question posée (pour trouver des questions similaires)
            alpha: Facteur d'influence du feedback (0-1)

        Returns:
            Liste des sources avec scores ajustés
        """
        # Trouver des questions similaires si une question est fournie
        similar_q_boost = 0.0
        suggested_answers = []

        if question:
            similar_questions = self.find_similar_questions(
                question, base_name, collection_name,
                similarity_threshold=0.5, max_results=3
            )

            # Calculer un boost global basé sur les feedbacks des questions similaires
            for sq in similar_questions:
                similarity = sq["similarity"]
                is_positive = sq.get("is_positive", True)

                # Boost pondéré par la similarité
                # Positif = +0.1, Négatif = -0.1 (pondéré par similarité)
                boost = (0.1 if is_positive else -0.1) * similarity
                similar_q_boost += boost

                # Collecter les réponses suggérées pour améliorer
                if sq.get("suggested_answer"):
                    suggested_answers.append(sq["suggested_answer"])

        # Appliquer le boost à toutes les sources
        reranked_sources = []
        for src in sources:
            original_score = src.get("score", 0.0)

            # Appliquer le facteur alpha au boost des questions similaires
            adjusted_score = original_score * (1 + alpha * similar_q_boost)

            # Créer la source enrichie
            enriched_src = src.copy()
            enriched_src["original_score"] = original_score
            enriched_src["score"] = round(max(0.0, min(1.0, adjusted_score)), 4)
            enriched_src["feedback_boost"] = round(similar_q_boost, 4)

            reranked_sources.append(enriched_src)

        # Re-trier par score ajusté
        reranked_sources.sort(key=lambda x: x["score"], reverse=True)

        return reranked_sources


def create_feedback(
    base_name: str,
    collection_name: str,
    question: str,
    is_positive: bool,
    suggested_answer: str = "",
    answer_text: str = "",
    top_k_used: int = 10,
    synthesize_all: bool = False
) -> QueryFeedback:
    """
    Fonction utilitaire pour créer un objet QueryFeedback (format simplifié).

    Args:
        base_name: Nom de la base
        collection_name: Nom de la collection
        question: Question posée
        is_positive: True = pouce haut, False = pouce bas
        suggested_answer: Réponse attendue par l'utilisateur
        answer_text: Texte de la réponse générée
        top_k_used: Nombre de sources utilisées
        synthesize_all: Mode synthèse activé

    Returns:
        Objet QueryFeedback
    """
    return QueryFeedback(
        feedback_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        user=getpass.getuser(),
        base_name=base_name,
        collection_name=collection_name,
        question=question,
        is_positive=is_positive,
        suggested_answer=suggested_answer,
        answer_text=answer_text,
        top_k_used=top_k_used,
        synthesize_all=synthesize_all
    )
