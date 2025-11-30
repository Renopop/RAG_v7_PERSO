"""
Query Understanding Module - Compréhension intelligente des requêtes

Détecte l'intention et le type de question pour adapter la stratégie
de recherche et améliorer la pertinence des résultats.

Phase 3 Quality Improvements v1.0
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
#  QUERY TYPES
# =============================================================================

class QueryIntent(Enum):
    """Types d'intentions de requête."""
    DEFINITION = "definition"       # Qu'est-ce que X ?
    PROCEDURE = "procedure"         # Comment faire X ?
    REQUIREMENT = "requirement"     # Quelles sont les exigences pour X ?
    COMPARISON = "comparison"       # Différence entre X et Y ?
    REFERENCE = "reference"         # Où trouver / Quel texte régit X ?
    FACTUAL = "factual"            # Valeur numérique, date, durée
    ELIGIBILITY = "eligibility"     # Qui peut / Conditions pour
    TROUBLESHOOT = "troubleshoot"   # Que faire si / En cas de
    LIST = "list"                   # Lister les / Quels sont les
    GENERAL = "general"             # Autre


class QueryComplexity(Enum):
    """Niveau de complexité de la requête."""
    SIMPLE = "simple"       # Réponse directe attendue
    MODERATE = "moderate"   # Nécessite synthèse
    COMPLEX = "complex"     # Multi-aspects, comparaison


class QueryDomain(Enum):
    """Domaine thématique de la requête."""
    LICENSING = "licensing"         # FCL, licences, qualifications
    OPERATIONS = "operations"       # OPS, procédures opérationnelles
    AIRWORTHINESS = "airworthiness" # Part 21, maintenance, certification
    MEDICAL = "medical"             # Certificats médicaux, aptitude
    TRAINING = "training"           # Formation, ATO, simulateurs
    SAFETY = "safety"               # SMS, occurrences, gestion risques
    GENERAL = "general"             # Non spécifique


# =============================================================================
#  PATTERN DEFINITIONS
# =============================================================================

# Patterns pour détecter l'intention
INTENT_PATTERNS = {
    QueryIntent.DEFINITION: [
        r"\bqu['']?est[- ]ce que?\b",
        r"\bdéfini(?:r|tion|ssez)\b",
        r"\bsignifie\b",
        r"\bwhat is\b",
        r"\bdefine\b",
        r"\bmeaning of\b",
        r"\bc['']est quoi\b",
        r"\bexplique[rz]?\s+(?:le concept|la notion)\b",
    ],
    QueryIntent.PROCEDURE: [
        r"\bcomment\s+(?:faire|procéder|effectuer|réaliser)\b",
        r"\bétapes?\s+(?:pour|de|à suivre)\b",
        r"\bprocédure\s+(?:pour|de)\b",
        r"\bhow to\b",
        r"\bsteps to\b",
        r"\bprocess for\b",
        r"\bméthode\s+pour\b",
        r"\bmanière\s+de\b",
    ],
    QueryIntent.REQUIREMENT: [
        r"\bexigences?\s+(?:pour|de|relatives?)\b",
        r"\bconditions?\s+(?:pour|de|requises?)\b",
        r"\bprérequis\b",
        r"\brequirements?\s+(?:for|to)\b",
        r"\bnécessaires?\s+pour\b",
        r"\bqu['']?(?:est[- ]ce qui est|faut[- ]il)\s+(?:requis|exigé|nécessaire)\b",
        r"\bobligatoire\b",
        r"\bdoit[- ]on\b",
        r"\bfaut[- ]il\b",
    ],
    QueryIntent.COMPARISON: [
        r"\bdifférence\s+entre\b",
        r"\bcomparer?\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdistinguer?\b",
        r"\bdifference between\b",
        r"\bcompare\b",
        r"\bpar rapport à\b",
        r"\bcontrairement à\b",
    ],
    QueryIntent.REFERENCE: [
        r"\boù\s+(?:trouver|est[- ]ce|se trouve)\b",
        r"\bquel\s+(?:texte|règlement|article|paragraphe)\b",
        r"\bréférence\s+(?:du|de la|réglementaire)\b",
        r"\bsource\b",
        r"\bwhere\s+(?:to find|is|can)\b",
        r"\bwhich\s+(?:regulation|rule|document)\b",
        r"\bquelle?\s+(?:partie|part|annexe|annex)\b",
    ],
    QueryIntent.FACTUAL: [
        r"\bcombien\b",
        r"\bquelle?\s+(?:durée|nombre|quantité|valeur)\b",
        r"\bà partir de\s+(?:quel|combien)\b",
        r"\bhow (?:many|much|long)\b",
        r"\bminimum\b",
        r"\bmaximum\b",
        r"\blimite\b",
        r"\bdurée\s+(?:de validité|minimale|maximale)\b",
        r"\b\d+\s*(?:heures?|h|ans?|mois|jours?)\b",
    ],
    QueryIntent.ELIGIBILITY: [
        r"\bqui peut\b",
        r"\béligible\b",
        r"\bpour être\b",
        r"\bconditions?\s+(?:d['']accès|pour accéder)\b",
        r"\bwho can\b",
        r"\beligib(?:le|ility)\b",
        r"\bpermis\s+de\b",
        r"\bautor[iu]s[ée]\s+à\b",
    ],
    QueryIntent.TROUBLESHOOT: [
        r"\bque\s+faire\s+(?:si|en cas)\b",
        r"\ben\s+cas\s+de\b",
        r"\bsi\s+(?:je|on|l['']on)\b",
        r"\bwhat\s+(?:to do|if)\b",
        r"\bproblème\b",
        r"\bincident\b",
        r"\bpanne\b",
        r"\béchoue\b",
    ],
    QueryIntent.LIST: [
        r"\blist(?:er?|e[zs]?)\b",
        r"\bénumérer?\b",
        r"\bquels?\s+sont\b",
        r"\bquelles?\s+sont\b",
        r"\btous\s+les\b",
        r"\btoutes\s+les\b",
        r"\blist\s+(?:of|all|the)\b",
        r"\bwhat are\b",
    ],
}

# Patterns pour détecter le domaine
DOMAIN_PATTERNS = {
    QueryDomain.LICENSING: [
        r"\bfcl\b", r"\blicence\b", r"\blicense\b", r"\bpilote\b", r"\bpilot\b",
        r"\bqualification\b", r"\brating\b", r"\btype\s+rating\b",
        r"\batpl\b", r"\bcpl\b", r"\bppl\b", r"\blapl\b", r"\bir\b",
        r"\binstructeur\b", r"\binstructor\b", r"\bexaminateur\b", r"\bexaminer\b",
        r"\bcheck\b", r"\btest\b", r"\bexamen\b", r"\bexam\b",
    ],
    QueryDomain.OPERATIONS: [
        r"\bops\b", r"\bopérations?\b", r"\boperations?\b",
        r"\bcat\b", r"\bncc\b", r"\bnco\b", r"\bspo\b",
        r"\bvol\b", r"\bflight\b", r"\bmission\b",
        r"\bfrm\b", r"\bfatigue\b", r"\bftl\b", r"\bfdm\b",
        r"\bminima\b", r"\bmeteo\b", r"\bmétéo\b",
    ],
    QueryDomain.AIRWORTHINESS: [
        r"\bpart[- ]?21\b", r"\bpart[- ]?m\b", r"\bpart[- ]?145\b",
        r"\bmaintenance\b", r"\bnavigabilité\b", r"\bairworthiness\b",
        r"\bcen\b", r"\barc\b", r"\bcertificat\s+de\s+type\b",
        r"\bmodification\b", r"\bstc\b", r"\bad\b", r"\bairworthiness directive\b",
    ],
    QueryDomain.MEDICAL: [
        r"\bmédical\b", r"\bmedical\b", r"\bclasse\s+[12]\b", r"\bclass\s+[12]\b",
        r"\bame\b", r"\baero[-\s]?medical\b", r"\blapl\s+medical\b",
        r"\baptitude\b", r"\bfitness\b", r"\bincapacit[ée]\b",
    ],
    QueryDomain.TRAINING: [
        r"\bato\b", r"\bformation\b", r"\btraining\b", r"\bentraînement\b",
        r"\bsimulateur\b", r"\bsimulator\b", r"\bfstd\b", r"\bffs\b",
        r"\bcours\b", r"\bcourse\b", r"\bmodule\b", r"\bprogramme\b",
    ],
    QueryDomain.SAFETY: [
        r"\bsms\b", r"\bsécurité\b", r"\bsafety\b",
        r"\boccurrence\b", r"\bincident\b", r"\baccident\b",
        r"\brisque\b", r"\brisk\b", r"\bhazard\b",
        r"\benquête\b", r"\binvestigation\b",
    ],
}

# Indicateurs de complexité
COMPLEXITY_INDICATORS = {
    QueryComplexity.COMPLEX: [
        r"\bet\s+aussi\b",
        r"\bainsi\s+que\b",
        r"\ben\s+tenant\s+compte\b",
        r"\bplusieurs?\b",
        r"\btou(?:s|tes)\s+les\b",
        r"\bcomparer?\b",
        r"\bdifférence\b",
        r"\ben\s+détail\b",
        r"\bexhaustive?ment\b",
    ],
    QueryComplexity.MODERATE: [
        r"\bpourquoi\b",
        r"\bexpliquer?\b",
        r"\bdécrire\b",
        r"\bcomment\b",
        r"\bquelles?\s+sont\b",
    ],
}


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class QueryAnalysis:
    """Résultat de l'analyse d'une requête."""
    original_query: str
    normalized_query: str
    intent: QueryIntent
    intent_confidence: float
    secondary_intents: List[Tuple[QueryIntent, float]]
    domain: QueryDomain
    complexity: QueryComplexity
    entities: List[str]  # Entités extraites (références, termes techniques)
    search_hints: Dict[str, Any]  # Hints pour optimiser la recherche


# =============================================================================
#  QUERY ANALYZER
# =============================================================================

class QueryAnalyzer:
    """
    Analyseur de requêtes pour comprendre l'intention et le contexte.

    Permet d'adapter la stratégie de recherche et de génération
    en fonction du type de question posée.
    """

    def __init__(self, log: Optional[logging.Logger] = None):
        """
        Args:
            log: Logger optionnel
        """
        self._log = log or logger

        # Compiler les patterns
        self._intent_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }
        self._domain_patterns = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in DOMAIN_PATTERNS.items()
        }
        self._complexity_patterns = {
            complexity: [re.compile(p, re.IGNORECASE) for p in patterns]
            for complexity, patterns in COMPLEXITY_INDICATORS.items()
        }

        # Pattern pour extraire les références réglementaires
        # Amélioré pour capturer les références EASA complètes (CS 25.613, AMC1 25.613, etc.)
        self._reference_pattern = re.compile(
            r'\b('
            r'(?:EASA\s*)?Part[- ]?\d+|'
            r'FCL[.\-]?\d+(?:\.[A-Z]\.\d+)?|'
            r'(?:FAR|FAA)\s*\d+[.-]\d+|'
            # Références CS complètes: CS 25.613, CS-E 510, CS-APU 25.1309, CS 25.613(a)(1)
            r'CS(?:-[A-Z]+)?\s*\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?(?:\([a-z0-9]+\))*|'
            # AMC/GM avec variantes numérotées: AMC 25.613, AMC1 25.613, AMC2 25.1309
            r'AMC\d{0,2}(?:-[A-Z]+)?\s*\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?(?:\([a-z0-9]+\))*|'
            r'GM\d{0,2}(?:-[A-Z]+)?\s*\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?(?:\([a-z0-9]+\))*|'
            # CAT, ORO, SPA, NCO, NCC, SPO avec segments multiples: CAT.OP.MPA.100, ORO.GEN.105
            r'(?:CAT|ORO|SPA|NCO|NCC|SPO)\.[A-Z]+(?:\.[A-Z]+)*\.\d+|'
            r'Annex(?:e)?\s*[IVXLCDM]+|'
            r'Article\s*\d+'
            r')\b',
            re.IGNORECASE
        )

    def _normalize_query(self, query: str) -> str:
        """Normalise la requête."""
        # Supprimer les espaces multiples
        query = re.sub(r'\s+', ' ', query).strip()

        # Supprimer la ponctuation finale redondante
        query = re.sub(r'[?!.]+$', '', query).strip()

        return query

    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float, List[Tuple[QueryIntent, float]]]:
        """Détecte l'intention de la requête."""
        scores = {}

        for intent, patterns in self._intent_patterns.items():
            match_count = sum(1 for p in patterns if p.search(query))
            if match_count > 0:
                # Score = nombre de matches / nombre de patterns (max 1.0)
                scores[intent] = min(1.0, match_count / max(1, len(patterns) * 0.3))

        if not scores:
            return QueryIntent.GENERAL, 0.5, []

        # Trier par score décroissant
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary_intent, primary_score = sorted_intents[0]
        secondary_intents = sorted_intents[1:3]  # Top 2 secondaires

        return primary_intent, primary_score, secondary_intents

    def _detect_domain(self, query: str) -> QueryDomain:
        """Détecte le domaine thématique."""
        scores = {}

        for domain, patterns in self._domain_patterns.items():
            match_count = sum(1 for p in patterns if p.search(query))
            if match_count > 0:
                scores[domain] = match_count

        if not scores:
            return QueryDomain.GENERAL

        return max(scores, key=scores.get)

    def _detect_complexity(self, query: str) -> QueryComplexity:
        """Détecte la complexité de la requête."""
        # Vérifier les indicateurs de complexité
        for complexity, patterns in self._complexity_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return complexity

        # Par défaut, basé sur la longueur
        word_count = len(query.split())
        if word_count > 20:
            return QueryComplexity.MODERATE
        return QueryComplexity.SIMPLE

    def _extract_entities(self, query: str) -> List[str]:
        """Extrait les entités (références réglementaires, termes techniques)."""
        entities = []

        # Extraire les références réglementaires
        refs = self._reference_pattern.findall(query)
        entities.extend(refs)

        return entities

    def _generate_search_hints(
        self,
        query: str,
        intent: QueryIntent,
        domain: QueryDomain,
        complexity: QueryComplexity,
        entities: List[str]
    ) -> Dict[str, Any]:
        """Génère des hints pour optimiser la recherche."""
        hints = {
            "boost_exact_match": False,
            "expand_query": False,
            "top_k_multiplier": 1.0,
            "prefer_recent": False,
            "filter_sections": [],
            "required_keywords": [],
        }

        # Selon l'intention
        if intent == QueryIntent.DEFINITION:
            hints["boost_exact_match"] = True
            hints["filter_sections"] = ["définition", "definition", "glossaire", "glossary"]

        elif intent == QueryIntent.PROCEDURE:
            hints["filter_sections"] = ["procédure", "procedure", "méthode", "method"]
            hints["top_k_multiplier"] = 1.5  # Plus de contexte pour les procédures

        elif intent == QueryIntent.REQUIREMENT:
            hints["boost_exact_match"] = True
            hints["filter_sections"] = ["exigences", "requirements", "conditions"]

        elif intent == QueryIntent.COMPARISON:
            hints["expand_query"] = True
            hints["top_k_multiplier"] = 2.0  # Besoin de plusieurs sources pour comparer

        elif intent == QueryIntent.REFERENCE:
            hints["boost_exact_match"] = True
            hints["filter_sections"] = ["référence", "reference", "applicable"]

        elif intent == QueryIntent.FACTUAL:
            hints["boost_exact_match"] = True
            hints["prefer_recent"] = True  # Les valeurs peuvent changer

        elif intent == QueryIntent.LIST:
            hints["top_k_multiplier"] = 2.0  # Besoin de couvrir tous les éléments

        # Selon la complexité
        if complexity == QueryComplexity.COMPLEX:
            hints["top_k_multiplier"] = max(hints["top_k_multiplier"], 2.0)
            hints["expand_query"] = True

        # Ajouter les entités comme mots-clés requis
        hints["required_keywords"] = entities

        return hints

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyse complète d'une requête.

        Args:
            query: Question de l'utilisateur

        Returns:
            Analyse détaillée de la requête
        """
        normalized = self._normalize_query(query)

        # Détecter l'intention
        intent, confidence, secondary_intents = self._detect_intent(normalized)

        # Détecter le domaine
        domain = self._detect_domain(normalized)

        # Détecter la complexité
        complexity = self._detect_complexity(normalized)

        # Extraire les entités
        entities = self._extract_entities(normalized)

        # Générer les hints de recherche
        search_hints = self._generate_search_hints(
            normalized, intent, domain, complexity, entities
        )

        analysis = QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            intent_confidence=confidence,
            secondary_intents=secondary_intents,
            domain=domain,
            complexity=complexity,
            entities=entities,
            search_hints=search_hints,
        )

        self._log.info(
            f"[QUERY] Intent: {intent.value} ({confidence:.0%}), "
            f"Domain: {domain.value}, Complexity: {complexity.value}"
        )

        return analysis


# =============================================================================
#  QUERY EXPANSION
# =============================================================================

def _expand_easa_references(query: str) -> List[str]:
    """
    Étend les références EASA pour chercher les documents connexes.

    Si la requête contient "CS 25.613", génère aussi:
    - "AMC 25.613" (guidance)
    - "GM 25.613" (guidance material)

    Idem dans l'autre sens: AMC -> CS, GM.
    """
    variants = []

    # Pattern pour détecter les références EASA
    easa_pattern = re.compile(
        r'\b(CS|AMC|GM)(\d{0,2})?\s*(\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?)',
        re.IGNORECASE
    )

    matches = list(easa_pattern.finditer(query))
    if not matches:
        return variants

    for match in matches:
        ref_type = match.group(1).upper()
        ref_number = match.group(2) or ""  # Pour AMC1, GM2, etc.
        section = match.group(3)

        original_ref = f"{ref_type}{ref_number} {section}".strip()

        # Générer les variantes connexes
        related_types = []
        if ref_type == "CS":
            related_types = ["AMC", "GM"]
        elif ref_type.startswith("AMC"):
            related_types = ["CS", "GM"]
        elif ref_type.startswith("GM"):
            related_types = ["CS", "AMC"]

        for related in related_types:
            new_ref = f"{related} {section}"
            new_query = query.replace(match.group(0), new_ref)
            if new_query != query and new_query not in variants:
                variants.append(new_query)

    return variants[:2]  # Max 2 variantes EASA


def expand_query_for_intent(
    query: str,
    intent: QueryIntent,
    domain: QueryDomain
) -> List[str]:
    """
    Génère des variantes de la requête basées sur l'intention.

    Args:
        query: Requête originale
        intent: Intention détectée
        domain: Domaine détecté

    Returns:
        Liste de requêtes variantes pour améliorer le recall
    """
    variants = [query]

    # Expansion EASA: CS 25.613 -> AMC 25.613, GM 25.613
    easa_variants = _expand_easa_references(query)
    variants.extend(easa_variants)

    # Variantes selon l'intention
    if intent == QueryIntent.DEFINITION:
        # Ajouter des formulations alternatives
        if "qu'est-ce que" not in query.lower():
            variants.append(f"définition {query}")
            variants.append(f"qu'est-ce que {query}")

    elif intent == QueryIntent.REQUIREMENT:
        variants.append(f"exigences {query}")
        variants.append(f"conditions requises {query}")

    elif intent == QueryIntent.PROCEDURE:
        variants.append(f"procédure {query}")
        variants.append(f"étapes {query}")

    # Variantes selon le domaine
    if domain == QueryDomain.LICENSING:
        if "fcl" not in query.lower():
            variants.append(f"FCL {query}")

    elif domain == QueryDomain.OPERATIONS:
        if "ops" not in query.lower():
            variants.append(f"OPS {query}")

    return variants[:5]  # Max 5 variantes (augmenté pour EASA)


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def analyze_query(query: str) -> QueryAnalysis:
    """
    Fonction utilitaire pour analyser une requête.

    Args:
        query: Question à analyser

    Returns:
        Analyse de la requête
    """
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)


def get_adaptive_top_k(
    base_top_k: int,
    query_analysis: QueryAnalysis
) -> int:
    """
    Calcule le top_k adapté à la requête.

    Args:
        base_top_k: Valeur de base de top_k
        query_analysis: Analyse de la requête

    Returns:
        top_k adapté
    """
    multiplier = query_analysis.search_hints.get("top_k_multiplier", 1.0)
    return max(3, int(base_top_k * multiplier))


def get_hybrid_search_weights(
    query_analysis: QueryAnalysis
) -> tuple:
    """
    Calcule les poids optimaux pour la recherche hybride selon l'intent.

    Certains types de requêtes bénéficient plus de la recherche sémantique (dense),
    d'autres de la recherche par mots-clés (sparse/BM25).

    Args:
        query_analysis: Analyse de la requête

    Returns:
        Tuple (dense_weight, sparse_weight)
    """
    intent = query_analysis.intent

    # DEFINITION: besoin de termes exacts + sémantique équilibrée
    if intent == QueryIntent.DEFINITION:
        return (0.5, 0.5)

    # PROCEDURE: comprendre le "comment" - favoriser sémantique
    elif intent == QueryIntent.PROCEDURE:
        return (0.75, 0.25)

    # REQUIREMENT: mots-clés réglementaires importants - équilibré
    elif intent == QueryIntent.REQUIREMENT:
        return (0.55, 0.45)

    # COMPARISON: comprendre les nuances - favoriser sémantique
    elif intent == QueryIntent.COMPARISON:
        return (0.7, 0.3)

    # REFERENCE: recherche de références exactes - favoriser BM25
    elif intent == QueryIntent.REFERENCE:
        return (0.4, 0.6)

    # FACTUAL: valeurs numériques, dates - favoriser BM25
    elif intent == QueryIntent.FACTUAL:
        return (0.45, 0.55)

    # LIST: énumérations - équilibré
    elif intent == QueryIntent.LIST:
        return (0.55, 0.45)

    # ELIGIBILITY: conditions et critères - favoriser sémantique
    elif intent == QueryIntent.ELIGIBILITY:
        return (0.65, 0.35)

    # TROUBLESHOOT: résolution problèmes - favoriser sémantique
    elif intent == QueryIntent.TROUBLESHOOT:
        return (0.7, 0.3)

    # GENERAL: par défaut équilibré avec légère préférence dense
    else:
        return (0.6, 0.4)


def format_query_analysis(analysis: QueryAnalysis) -> str:
    """
    Formate l'analyse pour affichage.

    Args:
        analysis: Analyse de la requête

    Returns:
        Texte formaté
    """
    lines = [
        f"Query: {analysis.original_query}",
        f"Intent: {analysis.intent.value} ({analysis.intent_confidence:.0%})",
        f"Domain: {analysis.domain.value}",
        f"Complexity: {analysis.complexity.value}",
    ]

    if analysis.entities:
        lines.append(f"Entities: {', '.join(analysis.entities)}")

    if analysis.secondary_intents:
        secondary = ", ".join(
            f"{i.value} ({c:.0%})" for i, c in analysis.secondary_intents
        )
        lines.append(f"Secondary intents: {secondary}")

    return "\n".join(lines)
