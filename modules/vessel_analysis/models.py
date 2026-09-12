"""
Module 4 - Vessel Investigation
Data models shared by this module.

These classes define the ONLY interface other modules (dashboard,
Module 5 ranking, alerts) should use to talk to this module.
Keeping the interface small means we can change the internals of
Module 4 later without breaking anything else in SPILLTRACE.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class OriginEstimate:
    """Output of Module 3 (Drift Intelligence) - what Module 4 needs as input."""
    latitude: float
    longitude: float
    time_start: datetime
    time_end: datetime
    radius_km: float = 15.0           # how far around the point we search
    time_buffer_hours: float = 2.0    # extra time margin before/after the window


@dataclass
class VesselPing:
    """One AIS record for one vessel at one point in time."""
    vessel_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed_knots: Optional[float] = None
    heading_deg: Optional[float] = None


@dataclass
class CandidateVessel:
    """A vessel identified as being near the probable origin during the time window."""
    vessel_id: str
    pings_in_window: List[VesselPing] = field(default_factory=list)
    min_distance_km: Optional[float] = None
    closest_ping_time: Optional[datetime] = None
    data_quality: str = "unknown"     # "good", "sparse", "very_sparse"

    def summary(self) -> dict:
        """A plain-dict view, handy for printing or feeding into a dashboard table."""
        return {
            "vessel_id": self.vessel_id,
            "num_pings_in_window": len(self.pings_in_window),
            "min_distance_km": round(self.min_distance_km, 2) if self.min_distance_km is not None else None,
            "closest_ping_time": self.closest_ping_time.isoformat() if self.closest_ping_time else None,
            "data_quality": self.data_quality,
        }
