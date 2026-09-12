"""
Module 5 - Vessel Ranking
Data model for a scored candidate vessel.

This is the ONLY interface other parts of SPILLTRACE (dashboard, alerts,
final report) should rely on. Import from the package's __init__.py,
not from this file directly.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RankedVessel:
    """A candidate vessel after evidence-based scoring."""
    vessel_id: str
    evidence_level: str          # "HIGH", "MEDIUM", or "LOW"
    score: float                 # 0.0 to 1.0, higher = stronger evidence
    min_distance_km: float
    pings_in_window: int
    data_quality: str
    evidence_notes: List[str] = field(default_factory=list)  # e.g. "Present near probable origin"

    def summary(self) -> dict:
        return {
            "vessel_id": self.vessel_id,
            "evidence_level": self.evidence_level,
            "score": round(self.score, 2),
            "min_distance_km": round(self.min_distance_km, 2),
            "pings_in_window": self.pings_in_window,
            "data_quality": self.data_quality,
            "evidence_notes": self.evidence_notes,
        }

    def print_report(self):
        """Print this vessel's evidence in the checklist style from the project doc."""
        print(f"\n{self.vessel_id}: {self.evidence_level} EVIDENCE (score: {self.score:.2f})")
        for note in self.evidence_notes:
            print(f"  {note}")
