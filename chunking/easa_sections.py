# easa_sections.py
"""
Utilities to detect and split EASA-style sections in regulatory documents.

Typical headers:
  - CS 25.613  Fatigue evaluation of structure
  - AMC 25.613  Fatigue evaluation of structure
  - GM 25.613  Fatigue evaluation of structure
  - CS-E 510  Engine controls
  - CS-APU 25.1309  Equipment, systems and installations
  - AMC1 25.631  Numbered AMC variant
  - GM2 25.631  Numbered GM variant
  - CS 25A.631  Lettered chapter variant

The main function split_easa_sections(text) returns a list of dicts:

  {
    "id": "CS 25.613",
    "kind": "CS" ou "CS-E" ou "CS-APU" etc.,
    "number": "25.613",
    "title": "Fatigue evaluation of structure",
    "full_text": "...."
  }

Supported examples:
  - "CS 25.631"
  - "AMC 25.631 Emergency provisions"
  - "GM 25.631"
  - "CS-E 510 Engine controls"
  - "CS-APU 25.1309 Equipment, systems and installations"
  - "BOOK 1 – CS 25.631 Emergency provisions"
  - "AMC1 25.631"    (numbered variant)
  - "GM2 25.1309"    (numbered variant)
  - "CS 25A.631"     (lettered chapter)
  - "Appendix A to CS 25.631"
"""

import re
from typing import List, Dict

# ----------------------------------------------------------------------
# HEADER_RE : détection des lignes de type "CS 25.613", "AMC 25.613", etc.
#
# On tolère :
#   - Des espaces / tabulations autour
#   - Préfixes optionnels: "BOOK 1 –", "Appendix A to", "Appendix A -", etc.
#   - Variantes de type "CS-E", "CS-APU", "CS-P", etc.
#   - Variantes numérotées: AMC1, AMC2, GM1, GM2, etc.
#   - Un titre optionnel après le numéro
#   - Des numéros comme "25.613", "25A.613", "25.613A", "25.613(a)", etc.
# ----------------------------------------------------------------------
HEADER_RE = re.compile(
    r"""^
        \s*
        (?:                                               # Préfixes optionnels
            BOOK\s+\d+\s*[-–—]\s*                         |  # "BOOK 1 –"
            (?:Appendix|APPENDIX)\s+[A-Z0-9]+\s*(?:to|TO|[-–—])\s*  # "Appendix A to" ou "Appendix A -"
        )?
        (                                                 # groupe 1: kind complet
            (CS|AMC|GM)                                   # groupe 2: base (CS, AMC, GM)
            (?:                                           # extensions optionnelles
                -[A-Z0-9]+                                |  # suffixe: CS-E, CS-APU, CS-25 (priorité)
                \d{1,2}                                      # numéro: AMC1, GM2 (1-2 chiffres)
            )?
        )
        [\s\-]+                                           # séparateur: espace(s) ou tiret(s)
        (                                                 # groupe 3: numéro de section
            [0-9]+[A-Z]?                                  # numéro principal avec lettre optionnelle (25, 25A)
            (?:[.\-][0-9A-Za-z]+)*                        # sous-numéros: .613, .613A, -613
            (?:\([0-9A-Za-z]+\))*                         # sous-paragraphes: (a), (1), (a)(1)
        )
        (?:\s+(.*))?                                      # groupe 4: titre optionnel
        \s*
        $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def split_easa_sections(text: str) -> List[Dict]:
    """
    Split a large EASA-like document text into sections based on CS/AMC/GM-style headers.

    Returns a list of sections, each being a dict:
      {
        "id": "CS 25.613",
        "kind": "CS" ou "CS-E" ou "CS-APU" etc.,
        "number": "25.613",
        "title": "Fatigue evaluation of structure",
        "full_text": "...."
      }
    """
    # Normalisation des fins de lignes
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: List[Dict] = []

    current_kind = None
    current_number = None
    current_title = None
    current_body_lines: List[str] = []

    def flush_section():
        nonlocal current_kind, current_number, current_title, current_body_lines
        if current_kind and current_number:
            section_id = f"{current_kind} {current_number}"
            sections.append(
                {
                    "id": section_id,
                    "kind": current_kind,          # ex: "CS", "CS-E", "CS-APU"
                    "number": current_number,      # ex: "25.613"
                    "title": current_title or "",
                    "full_text": "\n".join(current_body_lines).strip(),
                }
            )
        # reset
        current_kind = None
        current_number = None
        current_title = None
        current_body_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()

        m = HEADER_RE.match(line)
        if m:
            # On a trouvé un nouveau header de section
            flush_section()

            kind_full = m.group(1).strip()   # ex: "CS", "AMC", "GM", "CS-E", "CS-APU"
            number = m.group(3).strip()      # ex: "25.613", "25.613(a)(1)"
            title = (m.group(4) or "").strip()

            current_kind = kind_full
            current_number = number
            current_title = title
            current_body_lines = []

        else:
            # Ligne de corps : ajout à la section courante si on en a une
            if current_kind and current_number:
                current_body_lines.append(line)

    # Dernière section en fin de fichier
    flush_section()

    return sections
