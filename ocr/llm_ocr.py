"""
LLM OCR Module - OCR par Vision LLM

Utilise un LLM vision (GPT-4V, Claude Vision, DALLEM, etc.) pour extraire
le texte des pages PDF scannées ou de mauvaise qualité.

Fonctionnalités:
- Conversion PDF vers images via PyMuPDF
- Envoi des images au LLM Vision
- Extraction de texte structuré
- Mode fallback automatique si OCR classique échoue
- Support natif de l'API DALLEM
- Mode OFFLINE avec Donut-base local
"""

import os
import io
import base64
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import du gestionnaire de configuration pour le mode offline
try:
    from core.config_manager import is_offline_mode
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    def is_offline_mode():
        return False

# Import de l'OCR offline (Donut-base)
OFFLINE_OCR_AVAILABLE = False
try:
    from core.offline_models import get_offline_ocr, OfflineOCR
    OFFLINE_OCR_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
#  IMPORT CONFIGURATION DALLEM (optionnel)
# =============================================================================

try:
    from core.models_utils import (
        DALLEM_API_KEY,
        DALLEM_API_BASE,
        LLM_MODEL,
        create_http_client,
    )
    DALLEM_CONFIG_AVAILABLE = True
except ImportError:
    DALLEM_CONFIG_AVAILABLE = False
    DALLEM_API_KEY = None
    DALLEM_API_BASE = None
    LLM_MODEL = None

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Résolution pour la conversion PDF -> Image (DPI)
DEFAULT_DPI = 200  # 200 DPI pour meilleure qualité OCR

# Taille maximale d'image en pixels (redimensionner si plus grand)
MAX_IMAGE_SIZE = 2048

# Nombre maximum de pages à traiter par appel
MAX_PAGES_PER_BATCH = 5

# Format d'image pour l'encodage
IMAGE_FORMAT = "PNG"

# Prompt système pour l'extraction OCR
OCR_SYSTEM_PROMPT = """Tu es un expert en OCR (reconnaissance optique de caractères).
Ta tâche est d'extraire le texte EXACT de l'image fournie.

Instructions:
1. Extrais TOUT le texte visible, sans rien ajouter ni modifier
2. Préserve la structure (paragraphes, listes, tableaux)
3. Pour les tableaux, utilise des espaces ou | pour séparer les colonnes
4. Indique [illisible] pour les parties impossibles à lire
5. Conserve les numéros, références et codes exactement comme ils apparaissent
6. Si c'est un document réglementaire (EASA, FAA), fais très attention aux références (CS xx.xxx, AMC, GM)

Retourne UNIQUEMENT le texte extrait, sans commentaires ni explications."""

# Prompt utilisateur pour l'OCR
OCR_USER_PROMPT = """Extrais le texte de cette image de document.
Préserve la mise en page et la structure autant que possible.
Texte extrait:"""


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class OCRResult:
    """Résultat de l'OCR pour une page."""
    page_number: int
    text: str
    confidence: float  # 0-1 estimation de confiance
    processing_time: float  # secondes
    error: Optional[str] = None


@dataclass
class DocumentOCRResult:
    """Résultat de l'OCR pour un document complet."""
    total_pages: int
    processed_pages: int
    full_text: str
    page_results: List[OCRResult]
    total_time: float
    avg_confidence: float
    errors: List[str]


# =============================================================================
#  IMAGE UTILITIES
# =============================================================================

def detect_page_orientation(
    pdf_path: str,
    page_number: int,
    log=None
) -> int:
    """
    Détecte l'orientation d'une page PDF et retourne l'angle de correction.

    Méthodes de détection:
    1. Rotation metadata du PDF
    2. Analyse du texte existant (direction)
    3. Heuristique basée sur l'aspect ratio

    Args:
        pdf_path: Chemin du fichier PDF
        page_number: Numéro de page (0-indexed)
        log: Logger

    Returns:
        Angle de rotation à appliquer (0, 90, 180, 270) pour remettre droit
    """
    _log = log or logger

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return 0

    doc = None
    try:
        doc = fitz.open(pdf_path)
        if page_number >= len(doc):
            return 0

        page = doc[page_number]

        # 1. Vérifier la rotation metadata de la page
        page_rotation = page.rotation
        if page_rotation != 0:
            _log.debug(f"[LLM-OCR] Page {page_number}: rotation metadata = {page_rotation}°")
            # La rotation est déjà gérée par PyMuPDF lors du rendu
            return 0

        # 2. Analyser le texte existant pour détecter l'orientation
        # Extraire les blocs de texte avec leurs coordonnées
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = text_dict.get("blocks", [])

        text_blocks = [b for b in blocks if b.get("type") == 0]  # Type 0 = texte

        if text_blocks:
            # Analyser la direction dominante du texte
            orientation = _analyze_text_direction(text_blocks, page.rect, _log)
            if orientation != 0:
                _log.info(f"[LLM-OCR] Page {page_number}: orientation texte détectée = {orientation}°")
                return orientation

        # 3. Heuristique basée sur l'aspect ratio (pour pages scannées sans texte détectable)
        # Les documents A4/Letter sont généralement en portrait
        width = page.rect.width
        height = page.rect.height

        if width > height * 1.2:
            # Page en paysage - probablement tournée de 90°
            _log.debug(f"[LLM-OCR] Page {page_number}: aspect ratio suggère rotation 90°")
            # On ne corrige pas automatiquement car ça peut être intentionnel
            # Retourner 0 et laisser l'utilisateur décider

        return 0

    except Exception as e:
        _log.warning(f"[LLM-OCR] Erreur détection orientation page {page_number}: {e}")
        return 0
    finally:
        if doc:
            doc.close()


def _analyze_text_direction(blocks: List[Dict], page_rect, log) -> int:
    """
    Analyse la direction du texte dans les blocs pour détecter la rotation.

    Retourne l'angle de correction (0, 90, 180, 270).
    """
    if not blocks:
        return 0

    # Analyser les lignes de texte
    horizontal_count = 0
    vertical_count = 0
    upside_down_indicators = 0

    for block in blocks:
        for line in block.get("lines", []):
            # Calculer la direction de la ligne
            spans = line.get("spans", [])
            if not spans:
                continue

            # Bounding box de la ligne
            bbox = line.get("bbox", [0, 0, 0, 0])
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]

            if line_width > line_height * 2:
                horizontal_count += 1
            elif line_height > line_width * 2:
                vertical_count += 1

            # Vérifier si le texte est en haut ou en bas de la page
            # Le texte normal commence généralement en haut
            line_center_y = (bbox[1] + bbox[3]) / 2
            page_center_y = page_rect.height / 2

            # Si beaucoup de texte est en bas, possible rotation 180°
            if line_center_y > page_center_y:
                upside_down_indicators += 1

    total_lines = horizontal_count + vertical_count

    if total_lines == 0:
        return 0

    # Si majorité de lignes verticales -> rotation 90° ou 270°
    if vertical_count > horizontal_count * 2:
        log.debug(f"[LLM-OCR] Texte vertical détecté ({vertical_count} vs {horizontal_count} horizontal)")
        return 90  # Ou 270, difficile à déterminer sans plus d'analyse

    return 0


def pdf_page_to_image(
    pdf_path: str,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    max_size: int = MAX_IMAGE_SIZE,
    auto_rotate: bool = True,
    force_rotation: Optional[int] = None,
    log=None
) -> Optional[bytes]:
    """
    Convertit une page PDF en image PNG avec correction d'orientation.

    Args:
        pdf_path: Chemin du fichier PDF
        page_number: Numéro de page (0-indexed)
        dpi: Résolution en DPI
        max_size: Taille maximale en pixels
        auto_rotate: Si True, détecte et corrige l'orientation automatiquement
        force_rotation: Force une rotation spécifique (0, 90, 180, 270)

    Returns:
        Image en bytes (PNG) ou None si erreur
    """
    _log = log or logger

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            _log.error("[LLM-OCR] PyMuPDF non installé")
            return None

    doc = None
    try:
        doc = fitz.open(pdf_path)

        if page_number >= len(doc):
            _log.warning(f"[LLM-OCR] Page {page_number} n'existe pas (max={len(doc)-1})")
            return None

        page = doc[page_number]

        # Déterminer la rotation à appliquer
        rotation = 0
        if force_rotation is not None:
            rotation = force_rotation % 360
            _log.debug(f"[LLM-OCR] Page {page_number}: rotation forcée = {rotation}°")
        elif auto_rotate:
            rotation = detect_page_orientation(pdf_path, page_number, log=_log)

        # Calculer le zoom pour la résolution souhaitée
        zoom = dpi / 72.0  # 72 DPI est la résolution par défaut des PDF

        # Matrice de transformation avec rotation
        mat = fitz.Matrix(zoom, zoom)

        if rotation != 0:
            # Appliquer la rotation autour du centre de la page
            # PyMuPDF: rotation positive = sens horaire
            mat = mat.prerotate(rotation)
            _log.info(f"[LLM-OCR] Page {page_number}: rotation appliquée = {rotation}°")

        # Rendre la page en pixmap
        pix = page.get_pixmap(matrix=mat)

        # Redimensionner si nécessaire
        if pix.width > max_size or pix.height > max_size:
            scale = min(max_size / pix.width, max_size / pix.height)

            # Recréer avec le bon scale
            mat_scaled = fitz.Matrix(zoom * scale, zoom * scale)
            if rotation != 0:
                mat_scaled = mat_scaled.prerotate(rotation)
            pix = page.get_pixmap(matrix=mat_scaled)

        # Convertir en PNG bytes
        png_bytes = pix.tobytes("png")

        _log.debug(f"[LLM-OCR] Page {page_number}: {pix.width}x{pix.height} pixels")

        return png_bytes

    except Exception as e:
        _log.error(f"[LLM-OCR] Erreur conversion page {page_number}: {e}")
        return None
    finally:
        if doc:
            doc.close()


def detect_orientation_with_llm(
    image_bytes: bytes,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    log=None
) -> int:
    """
    Utilise le LLM Vision pour détecter l'orientation d'une image.

    Fait un appel réseau supplémentaire mais très fiable.

    Args:
        image_bytes: Image en bytes
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL de base
        model: Modèle vision

    Returns:
        Angle de rotation pour corriger (0, 90, 180, 270)
    """
    _log = log or logger

    orientation_prompt = """Analyse cette image de document.
Détermine si le document est correctement orienté pour la lecture.

Réponds UNIQUEMENT par un nombre:
- 0 si le document est droit (texte lisible normalement)
- 90 si le document est tourné de 90° dans le sens horaire (texte vertical, tête à droite)
- 180 si le document est à l'envers (texte inversé)
- 270 si le document est tourné de 90° dans le sens anti-horaire (texte vertical, tête à gauche)

Réponds uniquement par le nombre, sans explication."""

    image_b64 = image_to_base64(image_bytes)

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": orientation_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "low"  # Basse résolution suffit pour orientation
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 10,
    }

    try:
        resp = http_client.post(url, headers=headers, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Parser la réponse
        for angle in [0, 90, 180, 270]:
            if str(angle) in content:
                _log.info(f"[LLM-OCR] Orientation détectée par LLM: {angle}°")
                return angle

        _log.warning(f"[LLM-OCR] Réponse orientation non parsable: {content}")
        return 0

    except Exception as e:
        _log.warning(f"[LLM-OCR] Erreur détection orientation LLM: {e}")
        return 0


def image_to_base64(image_bytes: bytes) -> str:
    """Encode une image en base64."""
    return base64.b64encode(image_bytes).decode('utf-8')


def get_pdf_page_count(pdf_path: str, log=None) -> int:
    """Retourne le nombre de pages d'un PDF."""
    _log = log or logger

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            _log.error("[LLM-OCR] PyMuPDF non installé")
            return 0

    doc = None
    try:
        doc = fitz.open(pdf_path)
        return len(doc)
    except Exception as e:
        _log.error(f"[LLM-OCR] Erreur lecture PDF: {e}")
        return 0
    finally:
        if doc:
            doc.close()


# =============================================================================
#  LLM VISION OCR
# =============================================================================

def ocr_image_with_llm(
    image_bytes: bytes,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str = OCR_SYSTEM_PROMPT,
    user_prompt: str = OCR_USER_PROMPT,
    log=None
) -> Tuple[str, float]:
    """
    Extrait le texte d'une image via LLM Vision.

    Args:
        image_bytes: Image en bytes
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL de base de l'API
        model: Nom du modèle vision
        system_prompt: Prompt système
        user_prompt: Prompt utilisateur
        log: Logger

    Returns:
        (texte extrait, estimation de confiance)
    """
    _log = log or logger

    # Encoder l'image en base64
    image_b64 = image_to_base64(image_bytes)

    # Construire la requête pour l'API Vision
    # Format compatible OpenAI Vision API
    url = api_base.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Message avec image (format OpenAI Vision)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"  # Haute résolution pour OCR
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,  # Basse température pour extraction précise
        "max_tokens": 4096,  # Assez pour une page complète
    }

    try:
        resp = http_client.post(url, headers=headers, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not content:
            _log.warning("[LLM-OCR] Réponse vide du LLM")
            return "", 0.0

        # Estimation de confiance basée sur la longueur et le contenu
        confidence = _estimate_ocr_confidence(content)

        _log.debug(f"[LLM-OCR] Texte extrait: {len(content)} chars, confiance: {confidence:.0%}")

        return content, confidence

    except Exception as e:
        _log.error(f"[LLM-OCR] Erreur appel LLM Vision: {e}")
        return "", 0.0


def _estimate_ocr_confidence(text: str) -> float:
    """
    Estime la confiance de l'OCR basée sur le texte extrait.

    Heuristiques:
    - Texte trop court = faible confiance
    - Beaucoup de [illisible] = faible confiance
    - Ratio de mots reconnaissables
    """
    if not text:
        return 0.0

    confidence = 1.0

    # Pénalité pour texte court
    if len(text) < 100:
        confidence -= 0.3
    elif len(text) < 500:
        confidence -= 0.1

    # Pénalité pour sections illisibles
    illisible_count = text.lower().count("[illisible]")
    if illisible_count > 0:
        confidence -= min(0.4, illisible_count * 0.1)

    # Vérifier le ratio de caractères alphabétiques
    alpha_count = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_count / len(text) if text else 0

    if alpha_ratio < 0.3:
        confidence -= 0.2

    return max(0.0, min(1.0, confidence))


# =============================================================================
#  DOCUMENT OCR
# =============================================================================

def _format_duration(seconds: float) -> str:
    """Formate une durée en secondes en format lisible."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes:02d}m"


def _print_progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> None:
    """Affiche une barre de progression textuelle."""
    if total == 0:
        return
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix}[{bar}] {current}/{total} ({percent*100:.0f}%)", end="", flush=True)


def ocr_pdf_with_llm(
    pdf_path: str,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    pages: Optional[List[int]] = None,
    dpi: int = DEFAULT_DPI,
    auto_rotate: bool = True,
    smart_rotate: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log=None
) -> DocumentOCRResult:
    """
    Effectue l'OCR d'un PDF complet via LLM Vision.

    Args:
        pdf_path: Chemin du fichier PDF
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL de base de l'API
        model: Nom du modèle vision
        pages: Liste des pages à traiter (None = toutes)
        dpi: Résolution pour la conversion
        auto_rotate: Si True, corrige l'orientation automatiquement (local)
        smart_rotate: Si True et auto_rotate échoue, utilise LLM pour détecter
                      l'orientation (ajoute 1 appel réseau par page scannée)
        progress_cb: Callback de progression (ratio, message)
        log: Logger

    Returns:
        DocumentOCRResult avec le texte extrait
    """
    _log = log or logger
    start_time = time.time()

    # Obtenir le nombre de pages
    total_pages = get_pdf_page_count(pdf_path, log=_log)

    if total_pages == 0:
        return DocumentOCRResult(
            total_pages=0,
            processed_pages=0,
            full_text="",
            page_results=[],
            total_time=0,
            avg_confidence=0,
            errors=["Impossible de lire le PDF"]
        )

    # Déterminer les pages à traiter
    if pages is None:
        pages_to_process = list(range(total_pages))
    else:
        pages_to_process = [p for p in pages if 0 <= p < total_pages]

    # Affichage du démarrage OCR LLM
    import os
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"🔍 OCR LLM VISION - DÉMARRAGE")
    print(f"{'='*60}")
    print(f"  📄 Fichier: {filename}")
    print(f"  📖 Pages à traiter: {len(pages_to_process)}/{total_pages}")
    print(f"  🎯 Résolution: {dpi} DPI")
    print(f"  🔄 Auto-rotation: {'✅ Oui' if auto_rotate else '❌ Non'}")
    print(f"  🤖 Smart rotation (LLM): {'✅ Oui' if smart_rotate else '❌ Non'}")
    print(f"{'─'*60}")

    _log.info(f"[LLM-OCR] Démarrage OCR: {len(pages_to_process)}/{total_pages} pages")
    _log.info(f"[LLM-OCR] Options: auto_rotate={auto_rotate}, smart_rotate={smart_rotate}")

    page_results: List[OCRResult] = []
    errors: List[str] = []
    all_texts: List[str] = []

    for i, page_num in enumerate(pages_to_process):
        page_start = time.time()

        # Callback de progression
        if progress_cb:
            progress = (i + 1) / len(pages_to_process)
            progress_cb(progress, f"OCR page {page_num + 1}/{total_pages}")

        # Afficher la barre de progression
        _print_progress_bar(i, len(pages_to_process), prefix="📄 ")

        _log.info(f"[LLM-OCR] Traitement page {page_num + 1}/{total_pages}...")

        # Convertir la page en image (avec auto-rotation locale si activée)
        image_bytes = pdf_page_to_image(
            pdf_path, page_num,
            dpi=dpi,
            auto_rotate=auto_rotate,
            log=_log
        )

        if not image_bytes:
            error_msg = f"Erreur conversion page {page_num + 1}"
            errors.append(error_msg)
            page_results.append(OCRResult(
                page_number=page_num + 1,
                text="",
                confidence=0.0,
                processing_time=time.time() - page_start,
                error=error_msg
            ))
            continue

        # Smart rotation: détecter l'orientation par LLM si nécessaire
        # (pour les PDF scannés où la détection locale ne fonctionne pas)
        if smart_rotate:
            # Vérifier si la page semble être une image scannée (pas de texte détecté)
            local_rotation = detect_page_orientation(pdf_path, page_num, log=_log)

            if local_rotation == 0:
                # La détection locale n'a rien trouvé, utiliser LLM
                _log.debug(f"[LLM-OCR] Page {page_num + 1}: détection orientation par LLM...")
                llm_rotation = detect_orientation_with_llm(
                    image_bytes=image_bytes,
                    http_client=http_client,
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    log=_log
                )

                if llm_rotation != 0:
                    # Re-générer l'image avec la bonne rotation
                    _log.info(f"[LLM-OCR] Page {page_num + 1}: correction rotation {llm_rotation}°")
                    image_bytes = pdf_page_to_image(
                        pdf_path, page_num,
                        dpi=dpi,
                        auto_rotate=False,
                        force_rotation=llm_rotation,
                        log=_log
                    )

                    if not image_bytes:
                        error_msg = f"Erreur rotation page {page_num + 1}"
                        errors.append(error_msg)
                        page_results.append(OCRResult(
                            page_number=page_num + 1,
                            text="",
                            confidence=0.0,
                            processing_time=time.time() - page_start,
                            error=error_msg
                        ))
                        continue

        # OCR via LLM
        text, confidence = ocr_image_with_llm(
            image_bytes=image_bytes,
            http_client=http_client,
            api_key=api_key,
            api_base=api_base,
            model=model,
            log=_log
        )

        processing_time = time.time() - page_start

        if text:
            all_texts.append(f"--- Page {page_num + 1} ---\n{text}")
            page_results.append(OCRResult(
                page_number=page_num + 1,
                text=text,
                confidence=confidence,
                processing_time=processing_time
            ))
            _log.info(
                f"[LLM-OCR] Page {page_num + 1}: {len(text)} chars, "
                f"confiance={confidence:.0%}, temps={processing_time:.1f}s"
            )
        else:
            error_msg = f"OCR échoué page {page_num + 1}"
            errors.append(error_msg)
            page_results.append(OCRResult(
                page_number=page_num + 1,
                text="",
                confidence=0.0,
                processing_time=processing_time,
                error=error_msg
            ))

    # Barre de progression finale
    _print_progress_bar(len(pages_to_process), len(pages_to_process), prefix="📄 ")
    print()  # Nouvelle ligne après la barre

    # Calculer les statistiques
    total_time = time.time() - start_time
    processed_pages = sum(1 for r in page_results if r.text)
    confidences = [r.confidence for r in page_results if r.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Assembler le texte complet
    full_text = "\n\n".join(all_texts)

    # Affichage du résumé final
    print(f"{'─'*60}")
    print(f"✅ OCR LLM TERMINÉ")
    print(f"{'─'*60}")
    print(f"  📖 Pages traitées: {processed_pages}/{len(pages_to_process)}")
    if errors:
        print(f"  ❌ Erreurs: {len(errors)}")
    print(f"  📊 Confiance moyenne: {avg_confidence:.0%}")
    print(f"  📝 Caractères extraits: {len(full_text):,}")
    print(f"  ⏱️  Temps total: {_format_duration(total_time)}")
    if processed_pages > 0:
        print(f"  ⚡ Temps moyen/page: {_format_duration(total_time / processed_pages)}")
    print(f"{'='*60}\n")

    _log.info(
        f"[LLM-OCR] Terminé: {processed_pages}/{len(pages_to_process)} pages, "
        f"confiance={avg_confidence:.0%}, temps={total_time:.1f}s"
    )

    return DocumentOCRResult(
        total_pages=total_pages,
        processed_pages=processed_pages,
        full_text=full_text,
        page_results=page_results,
        total_time=total_time,
        avg_confidence=avg_confidence,
        errors=errors
    )


# =============================================================================
#  SMART OCR (FALLBACK AUTOMATIQUE)
# =============================================================================

def smart_ocr_pdf(
    pdf_path: str,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    quality_threshold: float = 0.5,
    fallback_pages_only: bool = True,
    auto_rotate: bool = True,
    smart_rotate: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log=None
) -> Dict[str, Any]:
    """
    OCR intelligent avec fallback automatique.

    1. Tente l'extraction classique (PyMuPDF/pdfminer)
    2. Évalue la qualité de chaque page
    3. Utilise LLM OCR uniquement pour les pages de mauvaise qualité
    4. Corrige l'orientation des pages si nécessaire

    Args:
        pdf_path: Chemin du PDF
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL de base de l'API
        model: Modèle vision
        quality_threshold: Seuil en dessous duquel on utilise LLM OCR
        fallback_pages_only: Si True, ne traite que les pages de mauvaise qualité
        auto_rotate: Si True, détection/correction locale de l'orientation
        smart_rotate: Si True, utilise LLM pour détecter l'orientation si local échoue
        progress_cb: Callback de progression
        log: Logger

    Returns:
        Dict avec texte final et métadonnées
    """
    _log = log or logger

    # Import des fonctions de qualité
    try:
        from processing.pdf_processing import (
            extract_text_from_pdf,
            assess_extraction_quality
        )
    except ImportError:
        _log.error("[LLM-OCR] Impossible d'importer pdf_processing")
        return {"error": "Module pdf_processing non disponible"}

    import os
    filename = os.path.basename(pdf_path)
    total_pages = get_pdf_page_count(pdf_path, log=_log)

    print(f"\n{'='*60}")
    print(f"🔬 SMART OCR - ANALYSE DE QUALITÉ")
    print(f"{'='*60}")
    print(f"  📄 Fichier: {filename}")
    print(f"  📖 Pages: {total_pages}")
    print(f"  🎯 Seuil qualité: {quality_threshold:.0%}")
    print(f"{'─'*60}")

    _log.info(f"[SMART-OCR] Analyse du PDF: {pdf_path}")

    # 1. Extraction classique
    print(f"  ⏳ Extraction classique en cours...")
    if progress_cb:
        progress_cb(0.1, "Extraction classique du texte...")

    try:
        classic_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        _log.warning(f"[SMART-OCR] Extraction classique échouée: {e}")
        classic_text = ""

    # 2. Évaluer la qualité globale
    print(f"  ⏳ Évaluation de la qualité...")
    if progress_cb:
        progress_cb(0.2, "Évaluation de la qualité...")

    quality = assess_extraction_quality(classic_text)
    quality_score = quality.get("quality_score", 0) if quality else 0

    # Afficher le résultat de l'évaluation
    quality_icon = "✅" if quality_score >= quality_threshold else "⚠️"
    print(f"  {quality_icon} Qualité extraction: {quality_score:.0%} (seuil: {quality_threshold:.0%})")
    print(f"  📝 Caractères extraits: {len(classic_text):,}")

    _log.info(f"[SMART-OCR] Qualité extraction classique: {quality_score:.0%}")

    # Si la qualité est bonne, retourner le texte classique
    if quality_score >= quality_threshold:
        print(f"{'─'*60}")
        print(f"  ✅ Qualité suffisante - pas besoin d'OCR LLM")
        print(f"{'='*60}\n")
        _log.info("[SMART-OCR] Qualité suffisante, pas besoin de LLM OCR")
        return {
            "text": classic_text,
            "method": "classic",
            "quality_score": quality_score,
            "llm_ocr_used": False,
            "pages_ocr": 0,
        }

    # 3. Utiliser LLM OCR
    print(f"{'─'*60}")
    print(f"  ⚠️ Qualité insuffisante → lancement OCR LLM Vision")
    print(f"{'─'*60}")
    _log.info(f"[SMART-OCR] Qualité insuffisante ({quality_score:.0%}), utilisation LLM OCR...")

    if progress_cb:
        progress_cb(0.3, "Démarrage OCR LLM Vision...")

    # Déterminer les pages à traiter
    if fallback_pages_only:
        # TODO: Évaluer la qualité page par page et ne traiter que les mauvaises
        # Pour l'instant, on traite tout si la qualité globale est mauvaise
        pages = None
    else:
        pages = None

    # OCR avec progression ajustée
    def ocr_progress(ratio, msg):
        if progress_cb:
            # Mapper 0-1 sur 0.3-0.95
            adjusted = 0.3 + ratio * 0.65
            progress_cb(adjusted, msg)

    ocr_result = ocr_pdf_with_llm(
        pdf_path=pdf_path,
        http_client=http_client,
        api_key=api_key,
        api_base=api_base,
        model=model,
        pages=pages,
        auto_rotate=auto_rotate,
        smart_rotate=smart_rotate,
        progress_cb=ocr_progress,
        log=_log
    )

    if progress_cb:
        progress_cb(1.0, "OCR terminé")

    # Choisir le meilleur résultat
    if ocr_result.avg_confidence > quality_score:
        final_text = ocr_result.full_text
        method = "llm_ocr"
    else:
        # Si l'OCR LLM n'est pas meilleur, garder le classique
        final_text = classic_text
        method = "classic_preferred"

    return {
        "text": final_text,
        "method": method,
        "quality_score": quality_score,
        "llm_ocr_used": True,
        "llm_ocr_confidence": ocr_result.avg_confidence,
        "pages_ocr": ocr_result.processed_pages,
        "total_pages": ocr_result.total_pages,
        "ocr_time": ocr_result.total_time,
        "ocr_errors": ocr_result.errors,
    }


# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def ocr_single_page(
    pdf_path: str,
    page_number: int,
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    dpi: int = DEFAULT_DPI,
    log=None
) -> OCRResult:
    """
    OCR d'une seule page via LLM Vision.

    Utile pour tester ou traiter des pages spécifiques.

    Args:
        pdf_path: Chemin du PDF
        page_number: Numéro de page (1-indexed pour l'utilisateur)
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL API
        model: Modèle vision
        dpi: Résolution
        log: Logger

    Returns:
        OCRResult pour cette page
    """
    _log = log or logger
    start_time = time.time()

    # Convertir en 0-indexed
    page_idx = page_number - 1

    # Convertir la page en image
    image_bytes = pdf_page_to_image(pdf_path, page_idx, dpi=dpi, log=_log)

    if not image_bytes:
        return OCRResult(
            page_number=page_number,
            text="",
            confidence=0.0,
            processing_time=time.time() - start_time,
            error="Erreur conversion page en image"
        )

    # OCR via LLM
    text, confidence = ocr_image_with_llm(
        image_bytes=image_bytes,
        http_client=http_client,
        api_key=api_key,
        api_base=api_base,
        model=model,
        log=_log
    )

    return OCRResult(
        page_number=page_number,
        text=text,
        confidence=confidence,
        processing_time=time.time() - start_time,
        error=None if text else "OCR n'a retourné aucun texte"
    )


def is_vision_model_available(
    http_client,
    api_key: str,
    api_base: str,
    model: str,
    log=None
) -> bool:
    """
    Vérifie si le modèle vision est disponible et fonctionnel.

    Args:
        http_client: Client HTTP
        api_key: Clé API
        api_base: URL API
        model: Nom du modèle

    Returns:
        True si le modèle vision est disponible
    """
    _log = log or logger

    # Créer une petite image de test (1x1 pixel blanc)
    test_image = base64.b64encode(
        # PNG 1x1 pixel blanc (plus petit PNG valide)
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff'
        b'\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82'
    ).decode('utf-8')

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Test"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{test_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 10,
    }

    try:
        resp = http_client.post(url, headers=headers, json=payload, timeout=30.0)

        if resp.status_code == 200:
            _log.info(f"[LLM-OCR] Modèle vision {model} disponible")
            return True
        else:
            _log.warning(f"[LLM-OCR] Modèle vision {model} non disponible: {resp.status_code}")
            return False

    except Exception as e:
        _log.warning(f"[LLM-OCR] Test modèle vision échoué: {e}")
        return False


# =============================================================================
#  FONCTIONS DE COMMODITÉ DALLEM
# =============================================================================

def ocr_pdf_with_dallem(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    dpi: int = DEFAULT_DPI,
    auto_rotate: bool = True,
    smart_rotate: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log=None
) -> DocumentOCRResult:
    """
    OCR d'un PDF en utilisant l'API DALLEM configurée.

    Utilise automatiquement les credentials DALLEM depuis models_utils.py.
    Détecte et corrige automatiquement l'orientation des pages scannées.

    Args:
        pdf_path: Chemin du fichier PDF
        pages: Liste des pages à traiter (None = toutes)
        dpi: Résolution pour la conversion
        auto_rotate: Si True, corrige l'orientation avec détection locale (gratuit)
        smart_rotate: Si True, utilise DALLEM pour détecter l'orientation des pages
                      scannées quand la détection locale échoue (+1 appel/page)
        progress_cb: Callback de progression
        log: Logger

    Returns:
        DocumentOCRResult avec le texte extrait

    Raises:
        RuntimeError: Si la configuration DALLEM n'est pas disponible

    Note:
        Appels réseau par page:
        - smart_rotate=False: 1 appel (OCR uniquement)
        - smart_rotate=True: 1-2 appels (détection orientation + OCR)
    """
    _log = log or logger

    if not DALLEM_CONFIG_AVAILABLE:
        raise RuntimeError(
            "Configuration DALLEM non disponible. "
            "Vérifiez que models_utils.py est accessible et configuré."
        )

    _log.info(f"[LLM-OCR] OCR avec DALLEM: {pdf_path}")
    _log.info(f"[LLM-OCR] API: {DALLEM_API_BASE}, Model: {LLM_MODEL}")
    _log.info(f"[LLM-OCR] Rotation: auto={auto_rotate}, smart={smart_rotate}")

    http_client = create_http_client()

    return ocr_pdf_with_llm(
        pdf_path=pdf_path,
        http_client=http_client,
        api_key=DALLEM_API_KEY,
        api_base=DALLEM_API_BASE,
        model=LLM_MODEL,
        pages=pages,
        dpi=dpi,
        auto_rotate=auto_rotate,
        smart_rotate=smart_rotate,
        progress_cb=progress_cb,
        log=_log
    )


def smart_ocr_with_dallem(
    pdf_path: str,
    quality_threshold: float = 0.5,
    fallback_pages_only: bool = True,
    auto_rotate: bool = True,
    smart_rotate: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log=None
) -> Dict[str, Any]:
    """
    OCR intelligent avec DALLEM et fallback automatique.

    1. Tente l'extraction classique (PyMuPDF/pdfminer)
    2. Évalue la qualité
    3. Utilise DALLEM OCR si qualité < threshold
    4. Corrige automatiquement l'orientation des pages

    Args:
        pdf_path: Chemin du PDF
        quality_threshold: Seuil de qualité (0-1)
        fallback_pages_only: Si True, ne traite que les pages de mauvaise qualité
        auto_rotate: Si True, détection locale de l'orientation (gratuit)
        smart_rotate: Si True, détection par LLM si locale échoue (+1 appel/page)
        progress_cb: Callback de progression
        log: Logger

    Returns:
        Dict avec texte et métadonnées

    Example:
        >>> result = smart_ocr_with_dallem("document_scanne.pdf", quality_threshold=0.5)
        >>> print(f"Méthode: {result['method']}, LLM utilisé: {result['llm_ocr_used']}")
    """
    _log = log or logger

    if not DALLEM_CONFIG_AVAILABLE:
        raise RuntimeError(
            "Configuration DALLEM non disponible. "
            "Vérifiez que models_utils.py est accessible et configuré."
        )

    _log.info(f"[LLM-OCR] Smart OCR avec DALLEM: {pdf_path}")

    http_client = create_http_client()

    return smart_ocr_pdf(
        pdf_path=pdf_path,
        http_client=http_client,
        api_key=DALLEM_API_KEY,
        api_base=DALLEM_API_BASE,
        model=LLM_MODEL,
        quality_threshold=quality_threshold,
        fallback_pages_only=fallback_pages_only,
        auto_rotate=auto_rotate,
        smart_rotate=smart_rotate,
        progress_cb=progress_cb,
        log=_log
    )


def check_dallem_vision_available(log=None) -> bool:
    """
    Vérifie si DALLEM est disponible et supporte les images.

    Returns:
        True si DALLEM Vision est disponible

    Example:
        >>> if check_dallem_vision_available():
        ...     result = smart_ocr_with_dallem("scan.pdf")
    """
    _log = log or logger

    if not DALLEM_CONFIG_AVAILABLE:
        _log.warning("[LLM-OCR] Configuration DALLEM non disponible")
        return False

    _log.info(f"[LLM-OCR] Test DALLEM Vision: {DALLEM_API_BASE}")

    http_client = create_http_client()

    return is_vision_model_available(
        http_client=http_client,
        api_key=DALLEM_API_KEY,
        api_base=DALLEM_API_BASE,
        model=LLM_MODEL,
        log=_log
    )


# =============================================================================
#  MODE OFFLINE - OCR AVEC DONUT-BASE LOCAL
# =============================================================================

def ocr_image_offline(
    image_bytes: bytes,
    log=None
) -> Tuple[str, float]:
    """
    Extrait le texte d'une image en mode offline avec Donut-base.

    Args:
        image_bytes: Image en bytes (PNG)
        log: Logger

    Returns:
        (texte extrait, estimation de confiance)
    """
    _log = log or logger

    if not OFFLINE_OCR_AVAILABLE:
        _log.error("[OFFLINE-OCR] Module offline non disponible")
        return "", 0.0

    try:
        from PIL import Image
        import io

        # Convertir les bytes en image PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Obtenir le client OCR offline
        ocr_client = get_offline_ocr(log=_log)

        # Extraire le texte
        text = ocr_client.extract_text(image)

        # Estimer la confiance
        confidence = _estimate_ocr_confidence(text)

        _log.debug(f"[OFFLINE-OCR] Texte extrait: {len(text)} chars, confiance: {confidence:.0%}")

        return text, confidence

    except Exception as e:
        _log.error(f"[OFFLINE-OCR] Erreur: {e}")
        return "", 0.0


def ocr_pdf_offline(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    dpi: int = DEFAULT_DPI,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log=None
) -> DocumentOCRResult:
    """
    OCR d'un PDF complet en mode offline avec Donut-base.

    Args:
        pdf_path: Chemin du fichier PDF
        pages: Liste des pages a traiter (None = toutes)
        dpi: Resolution pour la conversion
        progress_cb: Callback de progression
        log: Logger

    Returns:
        DocumentOCRResult avec le texte extrait
    """
    _log = log or logger
    start_time = time.time()

    if not OFFLINE_OCR_AVAILABLE:
        return DocumentOCRResult(
            total_pages=0,
            processed_pages=0,
            full_text="",
            page_results=[],
            total_time=0,
            avg_confidence=0,
            errors=["Module OCR offline non disponible"]
        )

    # Obtenir le nombre de pages
    total_pages = get_pdf_page_count(pdf_path, log=_log)

    if total_pages == 0:
        return DocumentOCRResult(
            total_pages=0,
            processed_pages=0,
            full_text="",
            page_results=[],
            total_time=0,
            avg_confidence=0,
            errors=["Impossible de lire le PDF"]
        )

    # Determiner les pages a traiter
    if pages is None:
        pages_to_process = list(range(total_pages))
    else:
        pages_to_process = [p for p in pages if 0 <= p < total_pages]

    # Affichage
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"OCR OFFLINE (Donut-base) - DEMARRAGE")
    print(f"{'='*60}")
    print(f"  Fichier: {filename}")
    print(f"  Pages a traiter: {len(pages_to_process)}/{total_pages}")
    print(f"{'─'*60}")

    _log.info(f"[OFFLINE-OCR] Demarrage: {len(pages_to_process)}/{total_pages} pages")

    page_results: List[OCRResult] = []
    errors: List[str] = []
    all_texts: List[str] = []

    for i, page_num in enumerate(pages_to_process):
        page_start = time.time()

        # Callback de progression
        if progress_cb:
            progress = (i + 1) / len(pages_to_process)
            progress_cb(progress, f"OCR page {page_num + 1}/{total_pages}")

        _log.info(f"[OFFLINE-OCR] Page {page_num + 1}/{total_pages}...")

        # Convertir la page en image
        image_bytes = pdf_page_to_image(pdf_path, page_num, dpi=dpi, log=_log)

        if not image_bytes:
            error_msg = f"Erreur conversion page {page_num + 1}"
            errors.append(error_msg)
            page_results.append(OCRResult(
                page_number=page_num + 1,
                text="",
                confidence=0.0,
                processing_time=time.time() - page_start,
                error=error_msg
            ))
            continue

        # OCR offline
        text, confidence = ocr_image_offline(image_bytes, log=_log)

        processing_time = time.time() - page_start

        if text:
            all_texts.append(f"--- Page {page_num + 1} ---\n{text}")
            page_results.append(OCRResult(
                page_number=page_num + 1,
                text=text,
                confidence=confidence,
                processing_time=processing_time
            ))
            _log.info(
                f"[OFFLINE-OCR] Page {page_num + 1}: {len(text)} chars, "
                f"confiance={confidence:.0%}, temps={processing_time:.1f}s"
            )
        else:
            error_msg = f"OCR echoue page {page_num + 1}"
            errors.append(error_msg)
            page_results.append(OCRResult(
                page_number=page_num + 1,
                text="",
                confidence=0.0,
                processing_time=processing_time,
                error=error_msg
            ))

    # Statistiques
    total_time = time.time() - start_time
    processed_pages = sum(1 for r in page_results if r.text)
    confidences = [r.confidence for r in page_results if r.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Assembler le texte complet
    full_text = "\n\n".join(all_texts)

    # Affichage final
    print(f"{'─'*60}")
    print(f"OCR OFFLINE TERMINE")
    print(f"  Pages traitees: {processed_pages}/{len(pages_to_process)}")
    print(f"  Confiance moyenne: {avg_confidence:.0%}")
    print(f"  Temps total: {total_time:.1f}s")
    print(f"{'='*60}\n")

    return DocumentOCRResult(
        total_pages=total_pages,
        processed_pages=processed_pages,
        full_text=full_text,
        page_results=page_results,
        total_time=total_time,
        avg_confidence=avg_confidence,
        errors=errors
    )


def smart_ocr_auto(
    pdf_path: str,
    quality_threshold: float = 0.5,
    fallback_pages_only: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    force_offline: bool = False,
    log=None
) -> Dict[str, Any]:
    """
    OCR intelligent avec selection automatique online/offline.

    Detecte le mode actuel et utilise:
    - Mode OFFLINE: Donut-base local
    - Mode ONLINE: DALLEM Vision API

    Args:
        pdf_path: Chemin du PDF
        quality_threshold: Seuil de qualite (0-1)
        fallback_pages_only: Si True, ne traite que les pages de mauvaise qualite
        progress_cb: Callback de progression
        force_offline: Si True, force le mode offline
        log: Logger

    Returns:
        Dict avec texte et metadonnees
    """
    _log = log or logger

    # Determiner le mode
    offline_mode = force_offline or (CONFIG_MANAGER_AVAILABLE and is_offline_mode())

    if offline_mode and OFFLINE_OCR_AVAILABLE:
        _log.info("[OCR] Mode OFFLINE actif - utilisation Donut-base local")

        # Utiliser l'OCR offline
        result = ocr_pdf_offline(
            pdf_path=pdf_path,
            progress_cb=progress_cb,
            log=_log
        )

        return {
            "text": result.full_text,
            "method": "offline_donut",
            "quality_score": result.avg_confidence,
            "llm_ocr_used": True,
            "llm_ocr_confidence": result.avg_confidence,
            "pages_ocr": result.processed_pages,
            "total_pages": result.total_pages,
            "ocr_time": result.total_time,
            "ocr_errors": result.errors,
            "offline_mode": True,
        }

    # Mode online
    if DALLEM_CONFIG_AVAILABLE:
        _log.info("[OCR] Mode ONLINE - utilisation DALLEM Vision")
        return smart_ocr_with_dallem(
            pdf_path=pdf_path,
            quality_threshold=quality_threshold,
            fallback_pages_only=fallback_pages_only,
            progress_cb=progress_cb,
            log=_log
        )

    # Aucun mode disponible
    _log.error("[OCR] Aucun mode OCR disponible (ni offline ni online)")
    return {
        "text": "",
        "method": "none",
        "error": "Aucun mode OCR disponible",
    }


def check_ocr_available() -> Dict[str, bool]:
    """
    Verifie la disponibilite des modes OCR.

    Returns:
        Dict avec statut de chaque mode
    """
    return {
        "online": DALLEM_CONFIG_AVAILABLE,
        "offline": OFFLINE_OCR_AVAILABLE,
        "current_mode": "offline" if (CONFIG_MANAGER_AVAILABLE and is_offline_mode()) else "online",
    }
