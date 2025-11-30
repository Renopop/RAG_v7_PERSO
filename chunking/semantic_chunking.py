"""
Semantic Chunking Module - Découpage sémantique intelligent

Découpe les documents aux frontières sémantiques au lieu d'utiliser
un simple comptage de caractères. Préserve la cohérence du contexte.

Phase 3 Quality Improvements v1.0
"""

import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
#  CONFIGURATION
# =============================================================================

# Taille cible des chunks (caractères)
DEFAULT_TARGET_SIZE = 1500

# Taille minimum d'un chunk
DEFAULT_MIN_SIZE = 200

# Taille maximum d'un chunk
DEFAULT_MAX_SIZE = 3000

# Overlap entre chunks (caractères)
DEFAULT_OVERLAP = 150


# =============================================================================
#  BOUNDARY TYPES
# =============================================================================

class BoundaryType(Enum):
    """Types de frontières sémantiques."""
    SECTION = "section"           # Nouvelle section majeure
    SUBSECTION = "subsection"     # Sous-section
    PARAGRAPH = "paragraph"       # Nouveau paragraphe
    SENTENCE = "sentence"         # Fin de phrase
    LIST_ITEM = "list_item"       # Élément de liste
    REGULATORY_REF = "reg_ref"    # Référence réglementaire
    NONE = "none"                 # Pas de frontière claire


# Poids des frontières (plus élevé = meilleure rupture)
BOUNDARY_WEIGHTS = {
    BoundaryType.SECTION: 1.0,
    BoundaryType.SUBSECTION: 0.8,
    BoundaryType.PARAGRAPH: 0.6,
    BoundaryType.REGULATORY_REF: 0.5,
    BoundaryType.LIST_ITEM: 0.4,
    BoundaryType.SENTENCE: 0.3,
    BoundaryType.NONE: 0.0,
}


# =============================================================================
#  PATTERN DEFINITIONS
# =============================================================================

# Patterns pour les sections
SECTION_PATTERNS = [
    # Titres numérotés
    r'^(?:\d+\.)+\s+[A-ZÀÉÈÊËÏÎÔÙÛÜ]',
    # Articles
    r'^Article\s+\d+',
    # Chapitres
    r'^Chapitre\s+\d+',
    r'^Chapter\s+\d+',
    # Sections EASA avec espace: CS 25.613, AMC 25.1309, AMC1 25.631, CS-E 510, CS-APU 25.1309
    r'^(?:CS|AMC|GM)(?:\d{0,2}|-[A-Z]+)?[\s\-]+\d+[A-Z]?(?:[.\-]\d+[A-Za-z]?)?',
    # Sections réglementaires avec points: CAT.OP.MPA.100, ORO.GEN.105
    r'^(?:CAT|ORO|NCO|NCC|SPO|SPA|FCL)\.[A-Z]+\.[A-Z]*\.?\d+',
    # Titres en majuscules
    r'^[A-ZÀÉÈÊËÏÎÔÙÛÜ][A-ZÀÉÈÊËÏÎÔÙÛÜ\s]{10,}$',
    # Annexes
    r'^Annex(?:e)?\s+[IVXLCDM\d]+',
    # Appendix
    r'^(?:Appendix|APPENDIX)\s+[A-Z0-9]+',
]

# Patterns pour les sous-sections
SUBSECTION_PATTERNS = [
    # Sous-numérotation
    r'^\([a-z]\)',
    r'^\(\d+\)',
    r'^[a-z]\)',
    r'^\d+\)',
    # Tirets/puces
    r'^[-–—•]\s+',
    # Lettres avec point
    r'^[a-z]\.\s+',
]

# Patterns pour les références réglementaires (début de texte réglementaire)
REGULATORY_PATTERNS = [
    r"^(?:Le|La|Les|L['\u2019])\s+(?:exploitant|pilote|commandant|titulaire)",
    r'^(?:Un|Une)\s+(?:licence|qualification|certificat)',
    r'^(?:Tout|Toute)\s+(?:personne|aéronef|exploitant)',
    r'^(?:The|A|An)\s+(?:operator|pilot|holder)',
    r'^(?:Pour|Afin de|En vue de)',
    r'^(?:Conformément|Selon|En application de)',
]

# Pattern pour fin de phrase
SENTENCE_END_PATTERN = re.compile(r'[.!?]\s+(?=[A-ZÀÉÈÊËÏÎÔÙÛÜ])')


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class SemanticBoundary:
    """Représente une frontière sémantique dans le texte."""
    position: int
    boundary_type: BoundaryType
    weight: float
    context: str  # Quelques caractères autour de la frontière


@dataclass
class SemanticChunk:
    """Un chunk sémantique."""
    text: str
    start_position: int
    end_position: int
    boundary_type: BoundaryType
    metadata: Dict[str, Any]


# =============================================================================
#  SEMANTIC CHUNKER
# =============================================================================

class SemanticChunker:
    """
    Découpeur sémantique intelligent.

    Identifie les frontières sémantiques dans le texte et découpe
    en préservant la cohérence du contexte.
    """

    def __init__(
        self,
        target_size: int = DEFAULT_TARGET_SIZE,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        log: Optional[logging.Logger] = None
    ):
        """
        Args:
            target_size: Taille cible des chunks (caractères)
            min_size: Taille minimum d'un chunk
            max_size: Taille maximum d'un chunk
            overlap: Overlap entre chunks
            log: Logger optionnel
        """
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap
        self._log = log or logger

        # Compiler les patterns
        self._section_patterns = [re.compile(p, re.MULTILINE) for p in SECTION_PATTERNS]
        self._subsection_patterns = [re.compile(p, re.MULTILINE) for p in SUBSECTION_PATTERNS]
        self._regulatory_patterns = [re.compile(p, re.MULTILINE) for p in REGULATORY_PATTERNS]

    def _find_boundaries(self, text: str) -> List[SemanticBoundary]:
        """Trouve toutes les frontières sémantiques dans le texte."""
        boundaries = []

        # Trouver les sections
        for pattern in self._section_patterns:
            for match in pattern.finditer(text):
                boundaries.append(SemanticBoundary(
                    position=match.start(),
                    boundary_type=BoundaryType.SECTION,
                    weight=BOUNDARY_WEIGHTS[BoundaryType.SECTION],
                    context=text[max(0, match.start()-10):match.end()+10],
                ))

        # Trouver les sous-sections
        for pattern in self._subsection_patterns:
            for match in pattern.finditer(text):
                boundaries.append(SemanticBoundary(
                    position=match.start(),
                    boundary_type=BoundaryType.SUBSECTION,
                    weight=BOUNDARY_WEIGHTS[BoundaryType.SUBSECTION],
                    context=text[max(0, match.start()-10):match.end()+10],
                ))

        # Trouver les références réglementaires
        for pattern in self._regulatory_patterns:
            for match in pattern.finditer(text):
                boundaries.append(SemanticBoundary(
                    position=match.start(),
                    boundary_type=BoundaryType.REGULATORY_REF,
                    weight=BOUNDARY_WEIGHTS[BoundaryType.REGULATORY_REF],
                    context=text[max(0, match.start()-10):match.end()+10],
                ))

        # Trouver les paragraphes (double saut de ligne)
        for match in re.finditer(r'\n\n+', text):
            boundaries.append(SemanticBoundary(
                position=match.end(),  # Après les sauts de ligne
                boundary_type=BoundaryType.PARAGRAPH,
                weight=BOUNDARY_WEIGHTS[BoundaryType.PARAGRAPH],
                context=text[max(0, match.start()-5):min(len(text), match.end()+20)],
            ))

        # Trouver les fins de phrase
        for match in SENTENCE_END_PATTERN.finditer(text):
            boundaries.append(SemanticBoundary(
                position=match.end(),
                boundary_type=BoundaryType.SENTENCE,
                weight=BOUNDARY_WEIGHTS[BoundaryType.SENTENCE],
                context=text[max(0, match.start()-10):min(len(text), match.end()+10)],
            ))

        # Trier par position
        boundaries.sort(key=lambda b: b.position)

        # Dédupliquer avec clustering intelligent
        # On garde toutes les frontières suffisamment espacées,
        # mais on fusionne celles qui sont vraiment superposées (même position exacte ou ±3 chars)
        if not boundaries:
            return []

        # Algorithme: parcourir les frontières triées et fusionner les clusters
        MERGE_THRESHOLD = 3  # Fusionner seulement les frontières à ±3 caractères
        deduped = []
        cluster_start = 0

        for i in range(1, len(boundaries) + 1):
            # Fin du cluster si on atteint la fin ou si la prochaine frontière est trop loin
            if i == len(boundaries) or boundaries[i].position - boundaries[cluster_start].position > MERGE_THRESHOLD:
                # Trouver la meilleure frontière dans ce cluster
                cluster = boundaries[cluster_start:i]
                best = max(cluster, key=lambda b: b.weight)
                deduped.append(best)
                cluster_start = i

        return deduped

    def _find_best_split_point(
        self,
        text: str,
        start: int,
        target_end: int,
        boundaries: List[SemanticBoundary]
    ) -> Tuple[int, BoundaryType]:
        """
        Trouve le meilleur point de découpe proche de la cible.

        Args:
            text: Texte complet
            start: Position de début
            target_end: Position cible de fin
            boundaries: Liste des frontières

        Returns:
            (position, type de frontière)
        """
        # Zone de recherche autour de la cible
        search_start = max(start + self.min_size, target_end - self.target_size // 3)
        search_end = min(len(text), target_end + self.target_size // 3)

        # Chercher la meilleure frontière dans la zone
        best_boundary = None
        best_score = -1

        for boundary in boundaries:
            if search_start <= boundary.position <= search_end:
                # Score = poids * proximité à la cible
                distance = abs(boundary.position - target_end)
                proximity_score = 1.0 - (distance / (search_end - search_start + 1))
                score = boundary.weight * 0.7 + proximity_score * 0.3

                if score > best_score:
                    best_score = score
                    best_boundary = boundary

        if best_boundary:
            return best_boundary.position, best_boundary.boundary_type

        # Si pas de frontière trouvée, chercher une fin de phrase
        for i in range(target_end, search_start, -1):
            if i < len(text) and text[i-1:i+1] in ['. ', '! ', '? ']:
                return i + 1, BoundaryType.SENTENCE

        # Fallback: découpe à target_end
        return min(target_end, len(text)), BoundaryType.NONE

    def _extract_metadata(self, chunk_text: str) -> Dict[str, Any]:
        """Extrait les métadonnées du chunk."""
        metadata = {}

        # Détecter le type de contenu
        if any(p.search(chunk_text) for p in self._section_patterns):
            metadata["has_section_header"] = True

        # Compter les éléments de liste
        list_items = len(re.findall(r'^\s*[-–—•]\s+', chunk_text, re.MULTILINE))
        list_items += len(re.findall(r'^\s*\([a-z]\)\s+', chunk_text, re.MULTILINE))
        list_items += len(re.findall(r'^\s*\d+\)\s+', chunk_text, re.MULTILINE))
        if list_items > 0:
            metadata["list_items_count"] = list_items

        # Extraire les références réglementaires
        refs = re.findall(
            r'\b(?:FCL|CAT|ORO|NCO|AMC|GM|CS|Part)[.\-]?\d+(?:\.[A-Z]+\.\d+)?',
            chunk_text,
            re.IGNORECASE
        )
        if refs:
            metadata["regulatory_refs"] = list(set(refs))

        # Densité du texte (ratio mots/caractères)
        words = len(chunk_text.split())
        chars = len(chunk_text)
        metadata["word_density"] = words / max(1, chars)

        return metadata

    def chunk(self, text: str) -> List[SemanticChunk]:
        """
        Découpe le texte en chunks sémantiques.

        Args:
            text: Texte à découper

        Returns:
            Liste de chunks sémantiques
        """
        if not text or len(text) < self.min_size:
            if text:
                return [SemanticChunk(
                    text=text,
                    start_position=0,
                    end_position=len(text),
                    boundary_type=BoundaryType.NONE,
                    metadata=self._extract_metadata(text),
                )]
            return []

        # Trouver toutes les frontières
        boundaries = self._find_boundaries(text)

        chunks = []
        current_start = 0

        while current_start < len(text):
            # Calculer la position cible de fin
            target_end = current_start + self.target_size

            if target_end >= len(text):
                # Dernier chunk
                chunk_text = text[current_start:].strip()
                if chunk_text:
                    chunks.append(SemanticChunk(
                        text=chunk_text,
                        start_position=current_start,
                        end_position=len(text),
                        boundary_type=BoundaryType.NONE,
                        metadata=self._extract_metadata(chunk_text),
                    ))
                break

            # Trouver le meilleur point de découpe
            split_pos, boundary_type = self._find_best_split_point(
                text, current_start, target_end, boundaries
            )

            # Vérifier la taille minimum (ne pas dépasser la fin du texte)
            if split_pos - current_start < self.min_size:
                split_pos = min(current_start + self.min_size, len(text))
            # Vérifier la taille maximum (elif pour éviter conflit si min_size > max_size)
            elif split_pos - current_start > self.max_size:
                split_pos = min(current_start + self.max_size, len(text))

            # Extraire le chunk
            chunk_text = text[current_start:split_pos].strip()

            if chunk_text:
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    start_position=current_start,
                    end_position=split_pos,
                    boundary_type=boundary_type,
                    metadata=self._extract_metadata(chunk_text),
                ))

            # Avancer avec overlap
            current_start = max(current_start + 1, split_pos - self.overlap)

        self._log.info(
            f"[CHUNKING] Created {len(chunks)} semantic chunks "
            f"(avg size: {sum(len(c.text) for c in chunks) // max(1, len(chunks))} chars)"
        )

        return chunks

    def chunk_with_context(
        self,
        text: str,
        context_size: int = 100
    ) -> List[SemanticChunk]:
        """
        Découpe avec contexte enrichi (début du chunk précédent).

        Args:
            text: Texte à découper
            context_size: Taille du contexte à ajouter

        Returns:
            Liste de chunks avec contexte
        """
        chunks = self.chunk(text)

        # Ajouter le contexte du chunk précédent
        for i in range(1, len(chunks)):
            prev_text = chunks[i-1].text
            context = prev_text[-context_size:] if len(prev_text) > context_size else prev_text

            # Marquer le contexte
            chunks[i].text = f"[Context: ...{context}]\n\n{chunks[i].text}"
            chunks[i].metadata["has_prev_context"] = True

        return chunks


# =============================================================================
#  ADAPTIVE CHUNKER
# =============================================================================

class AdaptiveSemanticChunker(SemanticChunker):
    """
    Chunker adaptatif qui ajuste la taille selon le contenu.

    - Texte dense (définitions, exigences) : chunks plus petits
    - Texte narratif : chunks plus grands
    - Listes : garde ensemble les éléments liés
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze_content_density(self, text: str) -> float:
        """
        Analyse la densité du contenu.

        Returns:
            Score 0-1 (1 = très dense)
        """
        # Indicateurs de densité
        indicators = 0

        # Présence de références réglementaires
        refs = len(re.findall(r'\b(?:FCL|CAT|ORO|Part|AMC)\.\w+', text, re.IGNORECASE))
        if refs > 2:
            indicators += 0.3

        # Présence de valeurs numériques avec unités
        nums = len(re.findall(r'\d+\s*(?:heures?|h|ans?|jours?|mois|%)', text, re.IGNORECASE))
        if nums > 3:
            indicators += 0.2

        # Présence de listes
        lists = len(re.findall(r'^\s*[-–—•(]\s*', text, re.MULTILINE))
        if lists > 5:
            indicators += 0.2

        # Ratio phrases courtes
        sentences = re.split(r'[.!?]', text)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 15)
        if sentences and short_sentences / len(sentences) > 0.5:
            indicators += 0.3

        return min(1.0, indicators)

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Découpe adaptative basée sur la densité du contenu."""
        # Analyser la densité globale
        density = self._analyze_content_density(text)

        # Ajuster la taille cible
        original_target = self.target_size
        if density > 0.7:
            # Contenu très dense : chunks plus petits
            self.target_size = int(original_target * 0.7)
        elif density < 0.3:
            # Contenu narratif : chunks plus grands
            self.target_size = int(original_target * 1.3)

        # Appeler le chunker parent
        chunks = super().chunk(text)

        # Restaurer la taille originale
        self.target_size = original_target

        return chunks


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def semantic_chunk(
    text: str,
    target_size: int = DEFAULT_TARGET_SIZE,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    adaptive: bool = True
) -> List[SemanticChunk]:
    """
    Fonction utilitaire pour le découpage sémantique.

    Args:
        text: Texte à découper
        target_size: Taille cible des chunks
        min_size: Taille minimum
        max_size: Taille maximum
        overlap: Overlap entre chunks
        adaptive: Utiliser le chunker adaptatif

    Returns:
        Liste de chunks sémantiques
    """
    ChunkerClass = AdaptiveSemanticChunker if adaptive else SemanticChunker

    chunker = ChunkerClass(
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        overlap=overlap,
    )

    return chunker.chunk(text)


def chunks_to_texts(chunks: List[SemanticChunk]) -> List[str]:
    """
    Convertit une liste de SemanticChunk en liste de textes.

    Args:
        chunks: Liste de chunks sémantiques

    Returns:
        Liste de textes
    """
    return [chunk.text for chunk in chunks]


def format_chunk_stats(chunks: List[SemanticChunk]) -> str:
    """
    Formate les statistiques des chunks.

    Args:
        chunks: Liste de chunks

    Returns:
        Statistiques formatées
    """
    if not chunks:
        return "No chunks"

    sizes = [len(c.text) for c in chunks]
    boundary_types = {}
    for c in chunks:
        bt = c.boundary_type.value
        boundary_types[bt] = boundary_types.get(bt, 0) + 1

    lines = [
        f"Total chunks: {len(chunks)}",
        f"Size range: {min(sizes)} - {max(sizes)} chars",
        f"Average size: {sum(sizes) // len(sizes)} chars",
        f"Boundary types: {boundary_types}",
    ]

    return "\n".join(lines)
