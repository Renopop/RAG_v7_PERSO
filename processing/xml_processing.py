"""
XML Processing Module - Parser pour fichiers XML de normes EASA
Supporte plusieurs patterns de découpage: CS xx.xxx, AMC, CS-E, CS-APU, ou custom
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class SectionPattern(Enum):
    """Patterns de découpage prédéfinis"""
    CS_STANDARD = "cs_standard"      # CS 25.101, CS-25.101, CS25.101
    AMC = "amc"                       # AMC 25.101, AMC-25.101
    CS_E = "cs_e"                     # CS-E 100, CS-E 200
    CS_APU = "cs_apu"                 # CS-APU 100, CS-APU 200
    GM = "gm"                         # GM 25.101 (Guidance Material)
    ALL_EASA = "all_easa"             # Tous les patterns EASA combinés
    CUSTOM = "custom"                 # Pattern regex personnalisé


# Patterns regex prédéfinis
PATTERNS = {
    SectionPattern.CS_STANDARD: re.compile(
        r'(CS[-\s]?\d+[A-Z]?[-.]?\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        re.IGNORECASE
    ),
    SectionPattern.AMC: re.compile(
        r'(AMC[-\s]?\d+[A-Z]?[-.]?\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        re.IGNORECASE
    ),
    SectionPattern.CS_E: re.compile(
        r'(CS[-\s]?E[-\s]?\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        re.IGNORECASE
    ),
    SectionPattern.CS_APU: re.compile(
        r'(CS[-\s]?APU[-\s]?\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        re.IGNORECASE
    ),
    SectionPattern.GM: re.compile(
        r'(GM[-\s]?\d+[A-Z]?[-.]?\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        re.IGNORECASE
    ),
    SectionPattern.ALL_EASA: re.compile(
        r'((CS[-\s]?\d+[A-Z]?[-.]?\d+|AMC[-\s]?\d+[A-Z]?[-.]?\d+|CS[-\s]?E[-\s]?\d+|CS[-\s]?APU[-\s]?\d+|GM[-\s]?\d+[A-Z]?[-.]?\d+)(?:\.\d+)?(?:\s*\([a-z]\))?)',
        re.IGNORECASE
    ),
}

# Descriptions pour l'UI
PATTERN_DESCRIPTIONS = {
    SectionPattern.CS_STANDARD: "CS xx.xxx - Certification Specifications (ex: CS 25.101, CS-25.102)",
    SectionPattern.AMC: "AMC xx.xxx - Acceptable Means of Compliance (ex: AMC 25.101)",
    SectionPattern.CS_E: "CS-E xxx - Engine Certification (ex: CS-E 100, CS-E 210)",
    SectionPattern.CS_APU: "CS-APU xxx - APU Certification (ex: CS-APU 100)",
    SectionPattern.GM: "GM xx.xxx - Guidance Material (ex: GM 25.101)",
    SectionPattern.ALL_EASA: "Tous EASA - Détecte CS, AMC, CS-E, CS-APU, GM",
    SectionPattern.CUSTOM: "Custom - Pattern regex personnalisé",
}


@dataclass
class XMLParseConfig:
    """Configuration pour le parsing XML"""
    pattern_type: SectionPattern = SectionPattern.ALL_EASA
    custom_pattern: Optional[str] = None  # Regex personnalisé si pattern_type == CUSTOM
    include_section_title: bool = True
    min_section_length: int = 50
    excluded_tags: List[str] = field(default_factory=list)


@dataclass
class Section:
    """Une section de document"""
    code: str
    title: str
    content: str
    start_pos: int


def get_pattern_regex(config: XMLParseConfig) -> Optional[re.Pattern]:
    """Retourne le pattern regex selon la configuration"""
    if config.pattern_type == SectionPattern.CUSTOM:
        if config.custom_pattern:
            try:
                return re.compile(config.custom_pattern, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Pattern regex invalide: {e}")
                return None
        return None
    return PATTERNS.get(config.pattern_type)


def extract_text_from_xml(xml_path: str, config: Optional[XMLParseConfig] = None) -> str:
    """
    Extrait le texte d'un fichier XML.
    """
    if config is None:
        config = XMLParseConfig()

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return _extract_all_text(root, config)
    except ET.ParseError as e:
        logger.error(f"Erreur de parsing XML {xml_path}: {e}")
        try:
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction XML {xml_path}: {e}")
        return ""


def _strip_namespace(tag: str) -> str:
    """Retire le namespace d'un tag XML"""
    if "}" in tag:
        return tag.split("}")[1]
    return tag


def _extract_all_text(root: ET.Element, config: XMLParseConfig) -> str:
    """Extrait tout le texte d'un élément XML"""
    texts = []

    def recurse(elem):
        tag_name = _strip_namespace(elem.tag)
        if tag_name.lower() in [t.lower() for t in config.excluded_tags]:
            return
        if elem.text:
            text = elem.text.strip()
            if text:
                texts.append(text)
        for child in elem:
            recurse(child)
        if elem.tail:
            tail = elem.tail.strip()
            if tail:
                texts.append(tail)

    recurse(root)
    return "\n".join(texts)


def detect_sections(text: str, config: Optional[XMLParseConfig] = None) -> List[Section]:
    """
    Détecte et extrait les sections selon le pattern configuré.
    """
    if config is None:
        config = XMLParseConfig()

    pattern = get_pattern_regex(config)

    if pattern is None:
        # Pas de pattern, retourner tout comme une seule section
        return [Section(
            code="DOCUMENT",
            title="Contenu complet",
            content=text.strip(),
            start_pos=0
        )]

    matches = list(pattern.finditer(text))

    if not matches:
        return [Section(
            code="DOCUMENT",
            title="Contenu complet (aucune section détectée)",
            content=text.strip(),
            start_pos=0
        )]

    sections = []
    for i, match in enumerate(matches):
        code = match.group(1).strip()
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start_pos:end_pos].strip()

        # Extraire le titre
        lines = section_text.split('\n')
        title = ""
        if lines:
            first_line = lines[0].replace(code, "").strip()
            if first_line:
                title = first_line[:100]
            elif len(lines) > 1:
                title = lines[1].strip()[:100]

        sections.append(Section(
            code=code,
            title=title,
            content=section_text,
            start_pos=start_pos
        ))

    return sections


def analyze_xml(xml_path: str, config: Optional[XMLParseConfig] = None) -> Dict[str, Any]:
    """
    Analyse un fichier XML et détecte les sections.
    """
    if config is None:
        config = XMLParseConfig()

    result = {
        "file": os.path.basename(xml_path),
        "total_chars": 0,
        "sections_count": 0,
        "sections": [],
        "pattern_used": config.pattern_type.value,
        "error": None
    }

    try:
        text = extract_text_from_xml(xml_path, config)
        result["total_chars"] = len(text)

        sections = detect_sections(text, config)
        result["sections_count"] = len(sections)

        for sec in sections:
            result["sections"].append({
                "code": sec.code,
                "title": sec.title[:50] + "..." if len(sec.title) > 50 else sec.title,
                "length": len(sec.content),
                "preview": sec.content[:150].replace('\n', ' ') + "..." if len(sec.content) > 150 else sec.content.replace('\n', ' ')
            })

    except Exception as e:
        result["error"] = str(e)

    return result


def preview_sections(xml_path: str, config: Optional[XMLParseConfig] = None, max_sections: int = 10) -> Tuple[str, Dict[str, Any]]:
    """
    Génère une prévisualisation des sections trouvées.
    """
    if config is None:
        config = XMLParseConfig()

    analysis = analyze_xml(xml_path, config)

    if analysis["error"]:
        return f"Erreur: {analysis['error']}", analysis

    lines = []
    lines.append(f"📄 Fichier: {analysis['file']}")
    lines.append(f"📊 {analysis['total_chars']:,} caractères")
    lines.append(f"🔍 Pattern: {PATTERN_DESCRIPTIONS.get(config.pattern_type, config.pattern_type.value)}")
    lines.append(f"📑 {analysis['sections_count']} section(s) détectée(s)")
    lines.append("")
    lines.append("=" * 50)

    for i, sec in enumerate(analysis["sections"][:max_sections]):
        lines.append("")
        lines.append(f"[{i+1}] {sec['code']}")
        if sec['title']:
            lines.append(f"    Titre: {sec['title']}")
        lines.append(f"    Taille: {sec['length']:,} caractères")
        lines.append(f"    Aperçu: {sec['preview'][:80]}...")

    if analysis["sections_count"] > max_sections:
        lines.append("")
        lines.append(f"... et {analysis['sections_count'] - max_sections} autres sections")

    return "\n".join(lines), analysis


def get_sections_for_chunking(xml_path: str, config: Optional[XMLParseConfig] = None) -> List[Dict[str, str]]:
    """
    Retourne les sections prêtes pour le chunking.
    """
    if config is None:
        config = XMLParseConfig()

    text = extract_text_from_xml(xml_path, config)
    sections = detect_sections(text, config)

    chunks = []
    for sec in sections:
        if len(sec.content) >= config.min_section_length:
            chunk_text = sec.content
            if config.include_section_title and sec.title:
                chunk_text = f"{sec.code} - {sec.title}\n\n{sec.content}"

            chunks.append({
                "text": chunk_text,
                "code": sec.code,
                "title": sec.title
            })

    return chunks


# Compatibilité avec l'ancien code
def detect_xml_structure(xml_path: str) -> Dict[str, Any]:
    """Compatibilité: alias pour analyze_xml"""
    return analyze_xml(xml_path)

def analyze_xml_for_easa(xml_path: str) -> Dict[str, Any]:
    """Compatibilité: alias pour analyze_xml avec pattern ALL_EASA"""
    return analyze_xml(xml_path, XMLParseConfig(pattern_type=SectionPattern.ALL_EASA))

def preview_xml_sections(xml_path: str, max_sections: int = 10) -> Tuple[str, Dict[str, Any]]:
    """Compatibilité: alias pour preview_sections"""
    return preview_sections(xml_path, None, max_sections)

def get_recommended_config(structure_info: Dict[str, Any]) -> XMLParseConfig:
    """Compatibilité: retourne config par défaut"""
    return XMLParseConfig()
