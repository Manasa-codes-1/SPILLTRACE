"""
Module 5 - Vessel Ranking

Public interface for other SPILLTRACE modules (dashboard, alerts, final
report) to use. Import from here, not from scorer.py directly.
"""

from .models import RankedVessel
from .scorer import rank_candidates

__all__ = [
    "RankedVessel",
    "rank_candidates",
]
