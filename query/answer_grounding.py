"""
Answer Grounding Module - Détection d'hallucinations

Vérifie que chaque affirmation dans la réponse LLM est supportée
par le contexte fourni. Détecte les hallucinations sans appel API.

Phase 3 Quality Improvements v1.0
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import string

logger = logging.getLogger(__name__)


# =============================================================================
#  CONFIGURATION
# =============================================================================

# Seuil minimum de couverture pour considérer une phrase comme "groundée"
MIN_GROUNDING_SCORE = 0.3

# Mots à ignorer dans l'analyse (stop words français + anglais + technique)
STOP_WORDS = {
    # Français
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais",
    "donc", "car", "ni", "que", "qui", "quoi", "dont", "où", "ce", "cette",
    "ces", "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes", "ton",
    "ta", "tes", "notre", "nos", "votre", "vos", "il", "elle", "ils", "elles",
    "je", "tu", "nous", "vous", "on", "se", "en", "y", "ne", "pas", "plus",
    "moins", "très", "bien", "peu", "trop", "aussi", "ainsi", "alors", "après",
    "avant", "avec", "sans", "sous", "sur", "dans", "pour", "par", "vers",
    "chez", "entre", "être", "avoir", "faire", "dire", "aller", "voir", "pouvoir",
    "vouloir", "devoir", "falloir", "est", "sont", "était", "sera", "serait",
    "peut", "doit", "fait", "dit", "va", "ont", "avait", "aura", "aurait",
    # Anglais
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could", "might",
    "must", "shall", "may", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "what", "which", "who", "whom",
    # Technique aviation
    "doit", "doivent", "peut", "peuvent", "selon", "conformément", "applicable",
    "requis", "exigé", "nécessaire", "approprié", "adéquat", "suivant",
}

# Patterns pour extraire les valeurs numériques avec unités
NUMERIC_PATTERN = re.compile(
    r'\b(\d+(?:[.,]\d+)?)\s*'
    r'(heures?|hours?|h|'
    r'minutes?|min|'
    r'secondes?|seconds?|s|'
    r'jours?|days?|'
    r'mois|months?|'
    r'ans?|years?|'
    r'kg|kilogrammes?|kilograms?|'
    r'lb|lbs|livres?|pounds?|'
    r'km|kilomètres?|kilometers?|'
    r'nm|nautical\s*miles?|milles?\s*nautiques?|'
    r'm|mètres?|meters?|'
    r'ft|feet|pieds?|'
    r'kts?|knots?|noeuds?|'
    r'mph|'
    r'%|pour\s*cent|percent|'
    r'°|degrés?|degrees?)\b',
    re.IGNORECASE
)

# Patterns pour les références réglementaires
REFERENCE_PATTERN = re.compile(
    r'\b('
    r'(?:EASA\s*)?Part[- ]?\d+|'
    r'(?:EASA\s*)?FCL[- ]?\d+|'
    r'(?:FAR|FAA)\s*\d+[.-]\d+|'
    r'CS[- ]?\d+|'
    r'AMC\d*[- ]?\d*|'
    r'GM\d*[- ]?\d*|'
    r'JAR[- ]?\d+|'
    r'OPS\s*\d+|'
    r'CAT\.[A-Z]+\.\d+|'
    r'ORO\.[A-Z]+\.\d+|'
    r'SPA\.[A-Z]+\.\d+|'
    r'NCO\.[A-Z]+\.\d+|'
    r'NCC\.[A-Z]+\.\d+|'
    r'SPO\.[A-Z]+\.\d+|'
    r'Annexe?\s*[IVXLCDM]+|'
    r'Annex\s*[IVXLCDM]+|'
    r'Article\s*\d+|'
    r'Section\s*\d+|'
    r'Chapitre\s*\d+|'
    r'Chapter\s*\d+'
    r')\b',
    re.IGNORECASE
)


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class GroundingResult:
    """Résultat de l'analyse de grounding pour une phrase."""
    sentence: str
    score: float  # 0-1, 1 = parfaitement supporté
    is_grounded: bool
    supporting_evidence: List[str]  # Extraits du contexte qui supportent
    ungrounded_claims: List[str]  # Éléments non trouvés dans le contexte
    numeric_claims: List[Tuple[str, bool]]  # (valeur, trouvée dans contexte)
    reference_claims: List[Tuple[str, bool]]  # (référence, trouvée dans contexte)


@dataclass
class AnswerGroundingReport:
    """Rapport complet de grounding pour une réponse."""
    overall_score: float  # Score global 0-1
    is_grounded: bool  # True si score >= seuil
    sentence_results: List[GroundingResult]
    total_sentences: int
    grounded_sentences: int
    ungrounded_sentences: int
    hallucination_risk: str  # "low", "medium", "high"
    flagged_claims: List[str]  # Claims potentiellement hallucinées


# =============================================================================
#  ANSWER GROUNDING ANALYZER
# =============================================================================

class AnswerGroundingAnalyzer:
    """
    Analyseur de grounding pour les réponses RAG.

    Vérifie que les affirmations dans la réponse LLM sont supportées
    par le contexte source, sans nécessiter d'appels API supplémentaires.
    """

    def __init__(
        self,
        min_grounding_score: float = MIN_GROUNDING_SCORE,
        log: Optional[logging.Logger] = None
    ):
        """
        Args:
            min_grounding_score: Seuil minimum pour considérer une phrase groundée
            log: Logger optionnel
        """
        self.min_grounding_score = min_grounding_score
        self._log = log or logger

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize et normalise le texte."""
        # Lowercase et suppression ponctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Split et filtrage
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

        return tokens

    def _split_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases."""
        # Pattern pour fin de phrase
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

        # Split
        sentences = sentence_endings.split(text)

        # Nettoyer et filtrer
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s) > 20]  # Ignorer phrases trop courtes

        return sentences

    def _extract_numeric_claims(self, text: str) -> List[str]:
        """Extrait les valeurs numériques avec leurs unités."""
        matches = NUMERIC_PATTERN.findall(text)
        return [f"{value} {unit}".strip() for value, unit in matches]

    def _extract_reference_claims(self, text: str) -> List[str]:
        """Extrait les références réglementaires."""
        matches = REFERENCE_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _compute_token_overlap(
        self,
        sentence_tokens: List[str],
        context_tokens: set
    ) -> float:
        """Calcule le ratio de tokens de la phrase présents dans le contexte."""
        if not sentence_tokens:
            return 1.0  # Phrase vide = pas de claim

        found = sum(1 for t in sentence_tokens if t in context_tokens)
        return found / len(sentence_tokens)

    def _find_supporting_evidence(
        self,
        sentence: str,
        context_sentences: List[str],
        context_tokens_list: List[set]
    ) -> List[str]:
        """Trouve les phrases du contexte qui supportent la phrase."""
        sentence_tokens = set(self._tokenize(sentence))

        if not sentence_tokens:
            return []

        supporting = []
        for ctx_sentence, ctx_tokens in zip(context_sentences, context_tokens_list):
            # Calculer overlap
            overlap = len(sentence_tokens & ctx_tokens) / len(sentence_tokens)
            if overlap >= 0.3:  # Au moins 30% de mots en commun
                supporting.append(ctx_sentence)

        return supporting[:3]  # Max 3 preuves

    def _check_claim_in_context(self, claim: str, context: str) -> bool:
        """Vérifie si un claim spécifique est présent dans le contexte."""
        # Normaliser
        claim_lower = claim.lower().strip()
        context_lower = context.lower()

        # Vérification directe
        if claim_lower in context_lower:
            return True

        # Pour les nombres, chercher avec variations
        numbers = re.findall(r'\d+(?:[.,]\d+)?', claim)
        if numbers:
            for num in numbers:
                # Chercher le nombre avec différents formats
                variations = [
                    num,
                    num.replace('.', ','),
                    num.replace(',', '.'),
                ]
                if any(v in context_lower for v in variations):
                    return True

        return False

    def analyze_sentence(
        self,
        sentence: str,
        context: str,
        context_sentences: List[str],
        context_tokens: set,
        context_tokens_list: List[set]
    ) -> GroundingResult:
        """Analyse le grounding d'une phrase individuelle."""
        # Tokenizer la phrase
        sentence_tokens = self._tokenize(sentence)

        # Calculer le score de base (overlap de tokens)
        base_score = self._compute_token_overlap(sentence_tokens, context_tokens)

        # Extraire et vérifier les claims numériques
        numeric_claims = self._extract_numeric_claims(sentence)
        numeric_results = [
            (claim, self._check_claim_in_context(claim, context))
            for claim in numeric_claims
        ]

        # Extraire et vérifier les références
        reference_claims = self._extract_reference_claims(sentence)
        reference_results = [
            (ref, self._check_claim_in_context(ref, context))
            for ref in reference_claims
        ]

        # Ajuster le score selon les claims vérifiables
        if numeric_results or reference_results:
            verifiable_claims = numeric_results + reference_results
            verified = sum(1 for _, found in verifiable_claims if found)
            claim_score = verified / len(verifiable_claims) if verifiable_claims else 1.0

            # Score final = moyenne pondérée (claims vérifiables plus importants)
            score = 0.4 * base_score + 0.6 * claim_score
        else:
            score = base_score

        # Trouver les preuves
        supporting_evidence = self._find_supporting_evidence(
            sentence, context_sentences, context_tokens_list
        )

        # Identifier les claims non supportés
        ungrounded_claims = []
        for claim, found in numeric_results:
            if not found:
                ungrounded_claims.append(f"Valeur: {claim}")
        for ref, found in reference_results:
            if not found:
                ungrounded_claims.append(f"Référence: {ref}")

        return GroundingResult(
            sentence=sentence,
            score=score,
            is_grounded=score >= self.min_grounding_score,
            supporting_evidence=supporting_evidence,
            ungrounded_claims=ungrounded_claims,
            numeric_claims=numeric_results,
            reference_claims=reference_results,
        )

    def analyze(
        self,
        answer: str,
        context: str
    ) -> AnswerGroundingReport:
        """
        Analyse complète du grounding d'une réponse.

        Args:
            answer: Réponse générée par le LLM
            context: Contexte source utilisé pour la génération

        Returns:
            Rapport de grounding détaillé
        """
        # Préparer le contexte
        context_sentences = self._split_sentences(context)
        context_tokens = set(self._tokenize(context))
        context_tokens_list = [set(self._tokenize(s)) for s in context_sentences]

        # Découper la réponse en phrases
        answer_sentences = self._split_sentences(answer)

        if not answer_sentences:
            return AnswerGroundingReport(
                overall_score=1.0,
                is_grounded=True,
                sentence_results=[],
                total_sentences=0,
                grounded_sentences=0,
                ungrounded_sentences=0,
                hallucination_risk="low",
                flagged_claims=[],
            )

        # Analyser chaque phrase
        sentence_results = []
        for sentence in answer_sentences:
            result = self.analyze_sentence(
                sentence, context, context_sentences, context_tokens, context_tokens_list
            )
            sentence_results.append(result)

        # Calculer les statistiques
        grounded_count = sum(1 for r in sentence_results if r.is_grounded)
        ungrounded_count = len(sentence_results) - grounded_count

        # Score global
        if sentence_results:
            overall_score = sum(r.score for r in sentence_results) / len(sentence_results)
        else:
            overall_score = 1.0

        # Collecter les claims flaggés
        flagged_claims = []
        for result in sentence_results:
            if not result.is_grounded:
                flagged_claims.extend(result.ungrounded_claims)
                if not result.ungrounded_claims:
                    # Phrase non groundée mais sans claim spécifique
                    flagged_claims.append(f"Phrase non supportée: {result.sentence[:100]}...")

        # Déterminer le niveau de risque
        if overall_score >= 0.7:
            hallucination_risk = "low"
        elif overall_score >= 0.4:
            hallucination_risk = "medium"
        else:
            hallucination_risk = "high"

        report = AnswerGroundingReport(
            overall_score=overall_score,
            is_grounded=overall_score >= self.min_grounding_score,
            sentence_results=sentence_results,
            total_sentences=len(sentence_results),
            grounded_sentences=grounded_count,
            ungrounded_sentences=ungrounded_count,
            hallucination_risk=hallucination_risk,
            flagged_claims=flagged_claims,
        )

        self._log.info(
            f"[GROUNDING] Score: {overall_score:.2f}, "
            f"Risk: {hallucination_risk}, "
            f"Grounded: {grounded_count}/{len(sentence_results)}"
        )

        return report


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def analyze_grounding(
    answer: str,
    context: str,
    min_score: float = MIN_GROUNDING_SCORE
) -> AnswerGroundingReport:
    """
    Fonction utilitaire pour analyser le grounding d'une réponse.

    Args:
        answer: Réponse du LLM
        context: Contexte source
        min_score: Seuil minimum de grounding

    Returns:
        Rapport de grounding
    """
    analyzer = AnswerGroundingAnalyzer(min_grounding_score=min_score)
    return analyzer.analyze(answer, context)


def format_grounding_report(report: AnswerGroundingReport) -> str:
    """
    Formate le rapport de grounding pour affichage.

    Args:
        report: Rapport de grounding

    Returns:
        Rapport formaté en texte
    """
    lines = [
        "=" * 60,
        "ANSWER GROUNDING REPORT",
        "=" * 60,
        "",
        f"Overall Score: {report.overall_score:.2%}",
        f"Hallucination Risk: {report.hallucination_risk.upper()}",
        f"Sentences: {report.grounded_sentences}/{report.total_sentences} grounded",
        "",
    ]

    if report.flagged_claims:
        lines.append("FLAGGED CLAIMS (potential hallucinations):")
        for claim in report.flagged_claims[:10]:  # Max 10
            lines.append(f"  ! {claim}")
        lines.append("")

    # Détails par phrase (optionnel)
    if report.sentence_results:
        lines.append("SENTENCE ANALYSIS:")
        for i, result in enumerate(report.sentence_results[:5], 1):  # Max 5
            status = "OK" if result.is_grounded else "!!"
            lines.append(f"  [{status}] {result.score:.0%}: {result.sentence[:80]}...")

        if len(report.sentence_results) > 5:
            lines.append(f"  ... and {len(report.sentence_results) - 5} more sentences")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def get_grounding_warning(report: AnswerGroundingReport) -> Optional[str]:
    """
    Génère un avertissement si le risque d'hallucination est élevé.

    Args:
        report: Rapport de grounding

    Returns:
        Message d'avertissement ou None
    """
    if report.hallucination_risk == "high":
        return (
            "ATTENTION: Cette réponse présente un risque élevé d'hallucination. "
            f"Seulement {report.grounded_sentences}/{report.total_sentences} phrases "
            "sont supportées par les sources. Veuillez vérifier les informations."
        )
    elif report.hallucination_risk == "medium":
        return (
            "Note: Certaines informations dans cette réponse ne sont pas "
            "directement vérifiables dans les sources fournies."
        )

    return None
