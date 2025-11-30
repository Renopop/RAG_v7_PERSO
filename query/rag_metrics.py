"""
RAG Metrics Module - Évaluation de la qualité du pipeline RAG

Implémente des métriques inspirées de RAGAS (Retrieval Augmented Generation Assessment):
- Faithfulness: La réponse est-elle fidèle au contexte?
- Answer Relevance: La réponse répond-elle à la question?
- Context Precision: Les sources récupérées sont-elles pertinentes?
- Context Recall: A-t-on récupéré toutes les informations nécessaires?

Phase 2 improvements v1.2
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class RAGMetrics:
    """Métriques de qualité pour une requête RAG."""

    # Métriques principales
    faithfulness: float  # 0-1: Réponse fidèle au contexte
    answer_relevance: float  # 0-1: Réponse pertinente à la question
    context_precision: float  # 0-1: Sources pertinentes
    context_utilization: float  # 0-1: Contexte utilisé dans la réponse

    # Métriques secondaires
    source_coverage: float  # 0-1: Diversité des sources
    reference_accuracy: float  # 0-1: Références citées correctement
    response_completeness: float  # 0-1: Réponse complète

    # Score global
    overall_score: float  # Moyenne pondérée

    # Métadonnées
    question: str
    answer: str
    num_sources: int
    context_length: int
    timestamp: str

    # Détails pour diagnostic
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGMetrics':
        """Crée depuis un dictionnaire."""
        return cls(**data)


# =============================================================================
#  METRICS CALCULATION
# =============================================================================

class RAGEvaluator:
    """
    Évaluateur de qualité pour les réponses RAG.

    Calcule les métriques sans dépendance externe (pas de LLM pour l'évaluation).
    Utilise des heuristiques et des analyses textuelles.
    """

    def __init__(self, log=None):
        self._log = log or logger

        # Patterns pour détecter les références EASA
        self.reference_patterns = [
            r'CS[\s\-]?\d+\.\d+',
            r'AMC[\s\-]?\d+\.\d+',
            r'GM[\s\-]?\d+\.\d+',
            r'CS[\s\-]?[A-Z]+[\s\-]?\d+',
            r'FAR[\s\-]?\d+\.\d+',
        ]

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        sources: List[Dict[str, Any]],
    ) -> RAGMetrics:
        """
        Évalue la qualité d'une réponse RAG.

        Args:
            question: Question posée
            answer: Réponse générée
            context: Contexte fourni au LLM
            sources: Sources utilisées

        Returns:
            RAGMetrics avec tous les scores
        """
        self._log.info(f"[METRICS] Evaluating response quality...")

        details = {}

        # 1. Faithfulness - La réponse utilise-t-elle le contexte?
        faithfulness, faith_details = self._compute_faithfulness(answer, context)
        details["faithfulness"] = faith_details

        # 2. Answer Relevance - La réponse répond-elle à la question?
        answer_relevance, rel_details = self._compute_answer_relevance(question, answer)
        details["answer_relevance"] = rel_details

        # 3. Context Precision - Les sources sont-elles pertinentes à la question?
        context_precision, prec_details = self._compute_context_precision(question, sources)
        details["context_precision"] = prec_details

        # 4. Context Utilization - Combien du contexte est utilisé?
        context_utilization, util_details = self._compute_context_utilization(answer, context)
        details["context_utilization"] = util_details

        # 5. Source Coverage - Diversité des sources
        source_coverage = self._compute_source_coverage(sources)
        details["source_coverage"] = {"score": source_coverage}

        # 6. Reference Accuracy - Références correctement citées
        reference_accuracy, ref_details = self._compute_reference_accuracy(answer, context)
        details["reference_accuracy"] = ref_details

        # 7. Response Completeness - Réponse complète
        response_completeness = self._compute_response_completeness(question, answer)
        details["response_completeness"] = {"score": response_completeness}

        # Score global (moyenne pondérée)
        weights = {
            "faithfulness": 0.25,
            "answer_relevance": 0.25,
            "context_precision": 0.15,
            "context_utilization": 0.15,
            "reference_accuracy": 0.10,
            "response_completeness": 0.10,
        }

        overall_score = (
            faithfulness * weights["faithfulness"] +
            answer_relevance * weights["answer_relevance"] +
            context_precision * weights["context_precision"] +
            context_utilization * weights["context_utilization"] +
            reference_accuracy * weights["reference_accuracy"] +
            response_completeness * weights["response_completeness"]
        )

        metrics = RAGMetrics(
            faithfulness=round(faithfulness, 3),
            answer_relevance=round(answer_relevance, 3),
            context_precision=round(context_precision, 3),
            context_utilization=round(context_utilization, 3),
            source_coverage=round(source_coverage, 3),
            reference_accuracy=round(reference_accuracy, 3),
            response_completeness=round(response_completeness, 3),
            overall_score=round(overall_score, 3),
            question=question[:200],
            answer=answer[:500],
            num_sources=len(sources),
            context_length=len(context),
            timestamp=datetime.now().isoformat(),
            details=details,
        )

        self._log.info(
            f"[METRICS] ✅ Evaluation complete: overall={overall_score:.2f}, "
            f"faith={faithfulness:.2f}, relevance={answer_relevance:.2f}, "
            f"precision={context_precision:.2f}"
        )

        return metrics

    def _compute_faithfulness(
        self,
        answer: str,
        context: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule la fidélité de la réponse au contexte.

        Méthode: Vérifie que les affirmations de la réponse sont présentes dans le contexte.
        """
        if not answer or not context:
            return 0.0, {"reason": "empty_input"}

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Extraire les phrases clés de la réponse
        sentences = re.split(r'[.!?]\s+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return 0.5, {"reason": "no_sentences", "sentence_count": 0}

        # Vérifier combien de phrases ont des éléments dans le contexte
        supported_count = 0
        for sentence in sentences:
            # Extraire les mots clés de la phrase
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
            if not words:
                continue

            # Compter combien de mots clés sont dans le contexte
            matches = sum(1 for w in words if w in context_lower)
            match_ratio = matches / len(words) if words else 0

            if match_ratio > 0.5:  # Plus de 50% des mots trouvés
                supported_count += 1

        faithfulness = supported_count / len(sentences) if sentences else 0.5

        return faithfulness, {
            "sentence_count": len(sentences),
            "supported_count": supported_count,
            "ratio": faithfulness,
        }

    def _compute_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule la pertinence de la réponse par rapport à la question.

        Méthode: Vérifie que la réponse adresse les éléments clés de la question.
        """
        if not question or not answer:
            return 0.0, {"reason": "empty_input"}

        question_lower = question.lower()
        answer_lower = answer.lower()

        # Extraire les mots clés de la question (sans stopwords)
        stopwords = {'what', 'how', 'why', 'when', 'where', 'which', 'who',
                    'is', 'are', 'the', 'a', 'an', 'in', 'on', 'for', 'to', 'of',
                    'quoi', 'comment', 'pourquoi', 'quand', 'où', 'quel', 'quelle',
                    'est', 'sont', 'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du'}

        question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question_lower))
        question_keywords = question_words - stopwords

        if not question_keywords:
            return 0.5, {"reason": "no_keywords"}

        # Compter combien de mots clés sont dans la réponse
        matches = sum(1 for kw in question_keywords if kw in answer_lower)
        keyword_coverage = matches / len(question_keywords)

        # Bonus si la réponse n'est pas trop courte ni trop longue
        answer_length = len(answer)
        length_score = 1.0
        if answer_length < 50:
            length_score = 0.5
        elif answer_length > 2000:
            length_score = 0.8

        # Pénalité si "I don't know" ou équivalent
        uncertainty_patterns = [
            r"i don't have",
            r"i cannot",
            r"no information",
            r"je ne sais pas",
            r"pas d'information",
        ]
        has_uncertainty = any(re.search(p, answer_lower) for p in uncertainty_patterns)
        uncertainty_penalty = 0.3 if has_uncertainty else 0.0

        relevance = min(1.0, keyword_coverage * length_score - uncertainty_penalty)
        relevance = max(0.0, relevance)

        return relevance, {
            "question_keywords": list(question_keywords)[:10],
            "keyword_matches": matches,
            "keyword_coverage": keyword_coverage,
            "length_score": length_score,
            "has_uncertainty": has_uncertainty,
        }

    def _compute_context_precision(
        self,
        question: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule la précision du contexte (sources pertinentes).

        Méthode: Vérifie que les sources contiennent des éléments de la question.
        """
        if not question or not sources:
            return 0.0, {"reason": "empty_input"}

        question_lower = question.lower()
        question_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', question_lower))

        if not question_words:
            return 0.5, {"reason": "no_question_words"}

        relevant_sources = 0
        source_scores = []

        for source in sources:
            text = source.get("text", "").lower()
            if not text:
                source_scores.append(0.0)
                continue

            # Compter les mots de la question présents dans la source
            matches = sum(1 for w in question_words if w in text)
            score = matches / len(question_words) if question_words else 0
            source_scores.append(score)

            if score > 0.3:  # Au moins 30% des mots clés
                relevant_sources += 1

        precision = relevant_sources / len(sources) if sources else 0.0

        return precision, {
            "total_sources": len(sources),
            "relevant_sources": relevant_sources,
            "source_scores": source_scores[:10],  # Top 10
        }

    def _compute_context_utilization(
        self,
        answer: str,
        context: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule le taux d'utilisation du contexte.

        Méthode: Mesure combien d'informations du contexte apparaissent dans la réponse.
        """
        if not answer or not context:
            return 0.0, {"reason": "empty_input"}

        # Extraire les phrases significatives du contexte
        context_sentences = re.split(r'[.!?\n]+', context)
        context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 30]

        if not context_sentences:
            return 0.5, {"reason": "no_context_sentences"}

        answer_lower = answer.lower()

        # Compter les phrases du contexte qui semblent utilisées
        used_count = 0
        for sentence in context_sentences[:50]:  # Limiter à 50 phrases
            # Extraire quelques mots clés de la phrase
            words = re.findall(r'\b[a-zA-Z]{5,}\b', sentence.lower())
            if len(words) < 3:
                continue

            # Vérifier si ces mots apparaissent ensemble dans la réponse
            key_words = words[:5]
            matches = sum(1 for w in key_words if w in answer_lower)

            if matches >= 2:  # Au moins 2 mots clés trouvés
                used_count += 1

        utilization = min(1.0, used_count / min(len(context_sentences), 50) * 2)

        return utilization, {
            "context_sentences": min(len(context_sentences), 50),
            "used_count": used_count,
        }

    def _compute_source_coverage(
        self,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calcule la diversité des sources.

        Méthode: Mesure le nombre de fichiers différents utilisés.
        """
        if not sources:
            return 0.0

        # Extraire les fichiers sources uniques
        unique_files = set()
        for source in sources:
            file_name = source.get("source_file", source.get("path", ""))
            if file_name:
                unique_files.add(file_name)

        # Score basé sur la diversité
        if len(unique_files) >= 5:
            return 1.0
        elif len(unique_files) >= 3:
            return 0.8
        elif len(unique_files) >= 2:
            return 0.6
        elif len(unique_files) == 1:
            return 0.4
        else:
            return 0.2

    def _compute_reference_accuracy(
        self,
        answer: str,
        context: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calcule la précision des références citées.

        Méthode: Vérifie que les références (CS, AMC) dans la réponse sont dans le contexte.
        """
        if not answer:
            return 0.5, {"reason": "no_answer"}

        # Extraire les références de la réponse
        answer_refs = set()
        for pattern in self.reference_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            answer_refs.update(m.upper().replace(" ", "-") for m in matches)

        if not answer_refs:
            return 1.0, {"reason": "no_references_to_check", "refs_found": 0}

        # Vérifier lesquelles sont dans le contexte
        context_upper = context.upper()
        valid_refs = sum(1 for ref in answer_refs if ref in context_upper)

        accuracy = valid_refs / len(answer_refs) if answer_refs else 1.0

        return accuracy, {
            "refs_in_answer": list(answer_refs),
            "refs_validated": valid_refs,
            "total_refs": len(answer_refs),
        }

    def _compute_response_completeness(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Calcule si la réponse semble complète.

        Méthode: Heuristiques basées sur la longueur et la structure.
        """
        if not answer:
            return 0.0

        # Longueur minimum attendue
        if len(answer) < 50:
            return 0.3

        # Vérifier la présence de structure
        has_structure = bool(re.search(r'[-•*]\s|\d+[.)]\s|:\n', answer))

        # Vérifier la présence de références
        has_references = bool(re.search(r'CS[\s-]?\d+|AMC[\s-]?\d+', answer, re.IGNORECASE))

        # Score de base sur la longueur
        if len(answer) > 500:
            length_score = 1.0
        elif len(answer) > 200:
            length_score = 0.8
        elif len(answer) > 100:
            length_score = 0.6
        else:
            length_score = 0.4

        # Bonus pour structure et références
        bonus = 0.1 if has_structure else 0
        bonus += 0.1 if has_references else 0

        return min(1.0, length_score + bonus)


# =============================================================================
#  METRICS STORAGE
# =============================================================================

class MetricsStore:
    """
    Stockage et agrégation des métriques RAG.
    """

    def __init__(self, storage_path: Optional[str] = None, log=None):
        self.storage_path = storage_path
        self._log = log or logger
        self._metrics: List[RAGMetrics] = []

        if storage_path and os.path.exists(storage_path):
            self._load()

    def add(self, metrics: RAGMetrics) -> None:
        """Ajoute une métrique."""
        self._metrics.append(metrics)
        if self.storage_path:
            self._save()

    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Retourne un résumé des métriques.

        Args:
            last_n: Limiter aux N dernières métriques
        """
        metrics = self._metrics[-last_n:] if last_n else self._metrics

        if not metrics:
            return {"count": 0}

        def avg(values):
            return sum(values) / len(values) if values else 0

        return {
            "count": len(metrics),
            "avg_overall": round(avg([m.overall_score for m in metrics]), 3),
            "avg_faithfulness": round(avg([m.faithfulness for m in metrics]), 3),
            "avg_relevance": round(avg([m.answer_relevance for m in metrics]), 3),
            "avg_precision": round(avg([m.context_precision for m in metrics]), 3),
            "avg_utilization": round(avg([m.context_utilization for m in metrics]), 3),
            "avg_ref_accuracy": round(avg([m.reference_accuracy for m in metrics]), 3),
            "period_start": metrics[0].timestamp if metrics else None,
            "period_end": metrics[-1].timestamp if metrics else None,
        }

    def get_all(self) -> List[Dict[str, Any]]:
        """Retourne toutes les métriques."""
        return [m.to_dict() for m in self._metrics]

    def _save(self) -> None:
        """Sauvegarde les métriques."""
        if not self.storage_path:
            return
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump([m.to_dict() for m in self._metrics], f, indent=2)
        except Exception as e:
            self._log.warning(f"[METRICS] Failed to save: {e}")

    def _load(self) -> None:
        """Charge les métriques."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._metrics = [RAGMetrics.from_dict(d) for d in data]
            self._log.info(f"[METRICS] Loaded {len(self._metrics)} metrics")
        except Exception as e:
            self._log.warning(f"[METRICS] Failed to load: {e}")


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def quick_evaluate(
    question: str,
    answer: str,
    context: str,
    sources: List[Dict[str, Any]],
    log=None
) -> Dict[str, float]:
    """
    Évaluation rapide retournant juste les scores.

    Returns:
        Dict avec les scores principaux
    """
    evaluator = RAGEvaluator(log=log)
    metrics = evaluator.evaluate(question, answer, context, sources)

    return {
        "overall": metrics.overall_score,
        "faithfulness": metrics.faithfulness,
        "relevance": metrics.answer_relevance,
        "precision": metrics.context_precision,
        "utilization": metrics.context_utilization,
    }


def format_metrics_report(metrics: RAGMetrics) -> str:
    """
    Formate un rapport lisible des métriques.
    """
    def score_bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled)

    def score_emoji(score: float) -> str:
        if score >= 0.8:
            return "✅"
        elif score >= 0.6:
            return "⚠️"
        else:
            return "❌"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    RAG QUALITY METRICS                        ║
╠══════════════════════════════════════════════════════════════╣
║  Overall Score: {metrics.overall_score:.0%}  {score_bar(metrics.overall_score)}  {score_emoji(metrics.overall_score)}
╠──────────────────────────────────────────────────────────────╣
║  Faithfulness:      {metrics.faithfulness:.0%}  {score_bar(metrics.faithfulness, 15)}  {score_emoji(metrics.faithfulness)}
║  Answer Relevance:  {metrics.answer_relevance:.0%}  {score_bar(metrics.answer_relevance, 15)}  {score_emoji(metrics.answer_relevance)}
║  Context Precision: {metrics.context_precision:.0%}  {score_bar(metrics.context_precision, 15)}  {score_emoji(metrics.context_precision)}
║  Context Util.:     {metrics.context_utilization:.0%}  {score_bar(metrics.context_utilization, 15)}  {score_emoji(metrics.context_utilization)}
║  Reference Acc.:    {metrics.reference_accuracy:.0%}  {score_bar(metrics.reference_accuracy, 15)}  {score_emoji(metrics.reference_accuracy)}
║  Completeness:      {metrics.response_completeness:.0%}  {score_bar(metrics.response_completeness, 15)}  {score_emoji(metrics.response_completeness)}
╠──────────────────────────────────────────────────────────────╣
║  Sources: {metrics.num_sources}  │  Context: {metrics.context_length:,} chars
╚══════════════════════════════════════════════════════════════╝
"""
    return report
