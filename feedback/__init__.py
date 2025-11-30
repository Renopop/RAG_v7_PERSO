"""
Feedback module - Gestion des retours utilisateurs

Ce module gère le feedback pour améliorer le reranking:
- feedback_store: Stockage et agrégation des feedbacks
"""

from .feedback_store import (
    FeedbackStore,
)

__all__ = [
    "FeedbackStore",
]
