"""
Module 5 - Vessel Ranking
Core logic: takes Module 4's candidate vessels and the origin estimate,
and produces evidence-based scores and HIGH/MEDIUM/LOW labels.

Public function other modules should call:
    rank_candidates(origin, candidates) -> List[RankedVessel]

Important principle (from the project doc): this NEVER claims a vessel
is guilty. It only ranks candidates by how strong the available evidence
is, for further human investigation.
"""

from typing import List

from modules.vessel_analysis import OriginEstimate, CandidateVessel
from .models import RankedVessel

# Score thresholds for the evidence level labels
HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4

# How much each factor contributes to the overall score
WEIGHT_DISTANCE = 0.5
WEIGHT_QUALITY = 0.3
WEIGHT_TIMING = 0.2

QUALITY_SCORES = {
    "good": 1.0,
    "sparse": 0.6,
    "very_sparse": 0.3,
    "unknown": 0.3,
}


def _distance_score(min_distance_km: float, radius_km: float) -> float:
    """1.0 if right at the origin, 0.0 if right at the edge of the search radius."""
    if radius_km <= 0:
        return 0.0
    score = 1.0 - (min_distance_km / radius_km)
    return max(0.0, min(1.0, score))


def _timing_score(closest_ping_time, origin: OriginEstimate) -> float:
    """
    1.0 if the closest ping landed right in the middle of the estimated
    release window, fading toward 0.0 near the edges (including buffer).
    """
    window_center = origin.time_start + (origin.time_end - origin.time_start) / 2
    half_width = (origin.time_end - origin.time_start).total_seconds() / 2
    half_width = max(half_width, 1.0)  # avoid divide-by-zero on a zero-length window

    offset_seconds = abs((closest_ping_time - window_center).total_seconds())
    score = 1.0 - (offset_seconds / half_width)
    return max(0.0, min(1.0, score))


def _evidence_level(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def _build_evidence_notes(candidate: CandidateVessel, distance_score, quality_score, timing_score) -> list:
    notes = []

    if distance_score >= 0.7:
        notes.append(f"[+] Very close to the probable origin ({candidate.min_distance_km:.2f} km)")
    elif distance_score >= 0.4:
        notes.append(f"[~] Within the search area, moderate distance from origin ({candidate.min_distance_km:.2f} km)")
    else:
        notes.append(f"[-] Near the edge of the search area ({candidate.min_distance_km:.2f} km)")

    if quality_score >= 0.9:
        notes.append(f"[+] Good AIS data coverage ({len(candidate.pings_in_window)} pings during the window)")
    elif quality_score >= 0.5:
        notes.append(f"[~] Sparse AIS data ({len(candidate.pings_in_window)} pings during the window)")
    else:
        notes.append(f"[-] Very limited AIS data ({len(candidate.pings_in_window)} ping(s)) - treat as uncertain, not exculpatory")

    if timing_score >= 0.7:
        notes.append("[+] Closest AIS ping is well-centered within the estimated release time")
    elif timing_score >= 0.4:
        notes.append("[~] Closest AIS ping is near the edge of the estimated release time")
    else:
        notes.append("[-] Closest AIS ping is at the far edge of the time buffer")

    return notes


def rank_candidates(
    origin: OriginEstimate,
    candidates: List[CandidateVessel],
) -> List[RankedVessel]:
    """
    Score and label each candidate vessel from Module 4 by evidence strength.
    Does NOT change or re-filter the candidate list - Module 4 already
    decided who's a candidate. This only adds scoring on top.
    """
    ranked = []

    for c in candidates:
        distance_score = _distance_score(c.min_distance_km, origin.radius_km)
        quality_score = QUALITY_SCORES.get(c.data_quality, 0.3)
        timing_score = _timing_score(c.closest_ping_time, origin)

        overall_score = (
            WEIGHT_DISTANCE * distance_score
            + WEIGHT_QUALITY * quality_score
            + WEIGHT_TIMING * timing_score
        )

        notes = _build_evidence_notes(c, distance_score, quality_score, timing_score)

        ranked.append(
            RankedVessel(
                vessel_id=c.vessel_id,
                evidence_level=_evidence_level(overall_score),
                score=overall_score,
                min_distance_km=c.min_distance_km,
                pings_in_window=len(c.pings_in_window),
                data_quality=c.data_quality,
                evidence_notes=notes,
            )
        )

    # Strongest evidence first
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
