"""
OCR module - Reconnaissance optique de caractères via LLM

Ce module gère l'OCR par LLM (DALLEM):
- llm_ocr: OCR via modèle de vision
"""

try:
    from .llm_ocr import (
        ocr_pdf_with_dallem,
        smart_ocr_with_dallem,
        ocr_pdf_with_llm,
        smart_ocr_pdf,
        ocr_single_page,
        check_dallem_vision_available,
    )
    LLM_OCR_AVAILABLE = True
except ImportError:
    LLM_OCR_AVAILABLE = False

__all__ = [
    "ocr_pdf_with_dallem",
    "smart_ocr_with_dallem",
    "ocr_pdf_with_llm",
    "smart_ocr_pdf",
    "ocr_single_page",
    "check_dallem_vision_available",
    "LLM_OCR_AVAILABLE",
]
