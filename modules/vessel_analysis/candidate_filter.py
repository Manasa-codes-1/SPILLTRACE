"""
Module 4 - Vessel Investigation
Core logic: given a probable origin area + time window (from Module 3)
and historical AIS data, find candidate vessels that were nearby.

Public function other modules should call:
    find_candidate_vessels(origin_estimate, pings) -> List[CandidateVessel]

Everything else in this file is an internal helper - don't import them
directly from other modules, import from the package's __init__.py instead.
"""

import math
from datetime import timedelta
from typing import List

from .models import OriginEstimate, VesselPing, CandidateVessel


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _within_time_window(ping_time, origin: OriginEstimate) -> bool:
    buffer = timedelta(hours=origin.time_buffer_hours)
    return (origin.time_start - buffer) <= ping_time <= (origin.time_end + buffer)


def find_candidate_vessels(
    origin: OriginEstimate,
    pings: List[VesselPing],
) -> List[CandidateVessel]:
    """
    Filter AIS pings down to vessels that were spatially AND temporally
    close to the probable spill origin.

    This does NOT rank or score vessels with evidence weights - it only
    identifies candidates. Ranking (Module 5) is a separate step that
    will consume this function's output.
    """
    matches_by_vessel = {}

    for ping in pings:
        if not _within_time_window(ping.timestamp, origin):
            continue

        distance_km = _haversine_km(origin.latitude, origin.longitude, ping.latitude, ping.longitude)
        if distance_km > origin.radius_km:
            continue

        candidate = matches_by_vessel.setdefault(
            ping.vessel_id, CandidateVessel(vessel_id=ping.vessel_id)
        )
        candidate.pings_in_window.append(ping)

        if candidate.min_distance_km is None or distance_km < candidate.min_distance_km:
            candidate.min_distance_km = distance_km
            candidate.closest_ping_time = ping.timestamp

    candidates = list(matches_by_vessel.values())

    for c in candidates:
        c.data_quality = _assess_data_quality(c, origin)

    candidates.sort(key=lambda c: c.min_distance_km)

    return candidates


def _assess_data_quality(candidate: CandidateVessel, origin: OriginEstimate) -> str:
    """
    Simple heuristic: more AIS pings inside the window relative to the
    window length = better data quality. This can be improved later
    without changing the function signature other modules depend on.
    """
    n = len(candidate.pings_in_window)
    window_hours = (origin.time_end - origin.time_start).total_seconds() / 3600.0
    window_hours = max(window_hours, 1.0)

    pings_per_hour = n / window_hours

    if pings_per_hour >= 2:
        return "good"
    elif pings_per_hour >= 0.5:
        return "sparse"
    else:
        return "very_sparse"
