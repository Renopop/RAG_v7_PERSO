"""
Processing module - Extraction de texte des différents formats

Ce module gère l'extraction de texte depuis:
- PDF (avec OCR si nécessaire)
- DOCX (Word)
- PPTX (PowerPoint avec pièces jointes)
- XML
- CSV
- Confluence
"""

from .pdf_processing import (
    extract_text_from_pdf,
    extract_attachments_from_pdf,
    extract_text_with_dallem,
    assess_extraction_quality,
    log_extraction_quality,
    LLM_OCR_AVAILABLE,
)

from .docx_processing import (
    extract_text_from_docx,
)

from .xml_processing import (
    extract_text_from_xml,
    XMLParseConfig,
)

# PPTX - optionnel car dépend de python-pptx
try:
    from .pptx_processing import (
        PPTXExtractor,
        PPTXExtractionResult,
        extract_pptx,
        process_pptx_with_attachments,
        is_pptx_processing_available,
        pptx_to_text,
    )
    PPTX_AVAILABLE = is_pptx_processing_available()
except ImportError:
    PPTX_AVAILABLE = False

# Confluence - optionnel
try:
    from .confluence_processing import (
        test_confluence_connection,
        list_spaces,
        extract_text_from_confluence_space,
        group_pages_by_section,
        get_space_info,
    )
    CONFLUENCE_AVAILABLE = True
except ImportError:
    CONFLUENCE_AVAILABLE = False

__all__ = [
    # pdf_processing
    "extract_text_from_pdf",
    "extract_attachments_from_pdf",
    "extract_text_with_dallem",
    "assess_extraction_quality",
    "log_extraction_quality",
    "LLM_OCR_AVAILABLE",
    # docx_processing
    "extract_text_from_docx",
    # xml_processing
    "extract_text_from_xml",
    "XMLParseConfig",
    # pptx_processing
    "PPTXExtractor",
    "PPTXExtractionResult",
    "extract_pptx",
    "process_pptx_with_attachments",
    "is_pptx_processing_available",
    "pptx_to_text",
    "PPTX_AVAILABLE",
    # confluence_processing
    "test_confluence_connection",
    "list_spaces",
    "extract_text_from_confluence_space",
    "group_pages_by_section",
    "get_space_info",
    "CONFLUENCE_AVAILABLE",
]
