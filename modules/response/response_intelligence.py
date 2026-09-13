"""
SPILLTRACE - Module 5: Response Intelligence

Uses REAL data throughout:
  - Real coastline geometry (Natural Earth 10m coastline, public domain,
    the same dataset used in professional GIS work)
  - Real oil-layer-thickness reference values (Bonn Agreement Oil
    Appearance Code - the real international table used by aerial
    oil-spill surveillance to estimate spill volume from its visible
    appearance/area)
  - Real Indian government tier classification (NOS-DCP - National Oil
    Spill Disaster Contingency Plan, Indian Coast Guard): Tier 1 (local,
    up to 700 tonnes), Tier 2 (regional, 700-10,000 tonnes), Tier 3
    (national emergency, over 10,000 tonnes)

IMPORTANT HONEST ASSUMPTION (state this in your presentation):
Converting a satellite-detected AREA into a TONNAGE requires assuming
an oil layer thickness, since area alone doesn't tell you volume. We
use the Bonn Agreement's "Discontinuous True Oil Colour" category
(~100 microns average thickness) as a reasonable middle estimate for
a visibly-detected slick. A production system would refine this using
the SAR backscatter intensity itself, not a fixed assumption.
"""

import math
import os
from datetime import datetime, timezone

import numpy as np
import geopandas as gpd

COASTLINE_PATH = "data/coastline/ne_10m_coastline.shp"

# Bounding box around the Arabian Sea / west coast of India, used to
# clip the global coastline file down to just the relevant region
# (keeps distance calculations fast without losing real coastline detail)
REGION_BBOX = (60, 5, 80, 25)  # (min_lon, min_lat, max_lon, max_lat)

# Bonn Agreement Oil Appearance Code (real, official reference table)
# category used: "Discontinuous True Oil Colour" - a reasonable midpoint
# assumption for a visibly detected satellite slick
ASSUMED_OIL_THICKNESS_MICRONS = 100.0
CRUDE_OIL_DENSITY_TONNES_PER_M3 = 0.9  # standard reference density (ITOPF)

# NOS-DCP real official tier thresholds (Indian Coast Guard)
NOSDCP_TIER1_MAX_TONNES = 700
NOSDCP_TIER3_MIN_TONNES = 10000

_coastline_points_cache = None


def load_coastline_points():
    """
    Loads REAL coastline vertex points from the Natural Earth 10m
    coastline shapefile, clipped to the Arabian Sea / India region.
    Cached after first load so repeated calls are fast.
    """
    global _coastline_points_cache
    if _coastline_points_cache is not None:
        return _coastline_points_cache

    if not os.path.exists(COASTLINE_PATH):
        raise FileNotFoundError(
            f"Real coastline file not found at {COASTLINE_PATH}. "
            f"Download it from naturalearthdata.com (10m coastline)."
        )

    gdf = gpd.read_file(COASTLINE_PATH)
    minx, miny, maxx, maxy = REGION_BBOX
    clipped = gdf.cx[minx:maxx, miny:maxy]

    points = []
    for geom in clipped.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            points.extend(list(geom.coords))
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                points.extend(list(line.coords))

    points = np.array(points)  # columns: [lon, lat]
    _coastline_points_cache = points
    print(f"Loaded {len(points)} real coastline points (Natural Earth, "
          f"clipped to Arabian Sea / India region)")
    return points


def nearest_real_coast(lat, lon):
    """
    Returns (distance_km, coast_lat, coast_lon) to the closest point on
    the REAL coastline, using vectorized haversine distance.
    """
    points = load_coastline_points()
    lons = points[:, 0]
    lats = points[:, 1]

    R = 6371.0
    phi1 = math.radians(lat)
    phi2 = np.radians(lats)
    dphi = np.radians(lats - lat)
    dlambda = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(a))

    idx = int(np.argmin(dist_km))
    return float(dist_km[idx]), float(lats[idx]), float(lons[idx])


def estimate_tonnes(spill_area_km2):
    """
    Converts a real detected spill area into an estimated tonnage,
    using the Bonn Agreement's real oil-appearance thickness reference.
    """
    thickness_m = ASSUMED_OIL_THICKNESS_MICRONS * 1e-6
    area_m2 = spill_area_km2 * 1_000_000
    volume_m3 = area_m2 * thickness_m
    tonnes = volume_m3 * CRUDE_OIL_DENSITY_TONNES_PER_M3
    return round(tonnes, 2)


def classify_nosdcp_tier(estimated_tonnes):
    """
    Classifies against India's REAL, official NOS-DCP tier system
    (Indian Coast Guard - National Oil Spill Disaster Contingency Plan).
    """
    if estimated_tonnes <= NOSDCP_TIER1_MAX_TONNES:
        return "Tier 1", "Local spill - handled by port/vessel/terminal operator"
    elif estimated_tonnes < NOSDCP_TIER3_MIN_TONNES:
        return "Tier 2", "Regional spill - state/regional response resources required"
    else:
        return "Tier 3", "National emergency - Indian Coast Guard central coordination required"


def estimate_time_to_impact(forward_path, impact_threshold_km=5.0):
    """
    Walks Module 3's forward_path (list of [lat, lon, hour]) and finds
    the first hour where the spill comes within impact_threshold_km of
    the REAL coastline. Returns (hours, distance_km_at_that_point) or
    (None, None) if it never gets close within the simulated window.
    """
    for lat, lon, hour in forward_path:
        dist_km, _, _ = nearest_real_coast(lat, lon)
        if dist_km <= impact_threshold_km:
            return hour, dist_km
    return None, None


def compute_response_priority(nosdcp_tier, time_to_impact_hours):
    """
    Combines the real NOS-DCP tier with real urgency (time-to-impact)
    into a single recommended response priority.
    """
    if nosdcp_tier == "Tier 3":
        base = "CRITICAL"
    elif nosdcp_tier == "Tier 2":
        base = "HIGH"
    else:
        base = "MEDIUM"

    # urgency can escalate the priority if impact is imminent
    if time_to_impact_hours is not None:
        if time_to_impact_hours <= 6:
            return "CRITICAL"
        elif time_to_impact_hours <= 12 and base in ("MEDIUM",):
            return "HIGH"

    if nosdcp_tier == "Tier 1" and time_to_impact_hours is None:
        return "LOW"

    return base


PRIORITY_ACTIONS = {
    "LOW": "Monitor",
    "MEDIUM": "Verify and prepare response",
    "HIGH": "Dispatch/prepare response resources",
    "CRITICAL": "Immediate escalation recommended",
}


def build_response_intelligence(detection, drift_result, forward_path):
    """
    Main entry point. Combines Module 2 + Module 3 outputs into a full
    Response Intelligence report, using real coastline data and the
    real NOS-DCP tier framework throughout.
    """
    spill_area_km2 = detection.get("spill_area_km2", 0.0)
    confidence = detection.get("detection_confidence", detection.get("confidence", 0.0))

    impact_point = drift_result["predicted_impact_point"]
    dist_now_km, coast_lat, coast_lon = nearest_real_coast(impact_point["lat"], impact_point["lon"])

    estimated_tonnes = estimate_tonnes(spill_area_km2)
    tier, tier_description = classify_nosdcp_tier(estimated_tonnes)

    time_to_impact_hours, dist_at_impact = estimate_time_to_impact(forward_path)

    priority = compute_response_priority(tier, time_to_impact_hours)

    report = {
        "spill_area_km2": spill_area_km2,
        "detection_confidence": confidence,
        "estimated_tonnes": estimated_tonnes,
        "nosdcp_tier": tier,
        "nosdcp_tier_description": tier_description,
        "nearest_real_coast_km": round(dist_now_km, 1),
        "nearest_real_coast_coords": (round(coast_lat, 4), round(coast_lon, 4)),
        "time_to_impact_hours": time_to_impact_hours,
        "response_priority": priority,
        "recommended_action": PRIORITY_ACTIONS[priority],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return report


def format_alert_card(source_image, spill_lat, spill_lon, report):
    """
    Formats the report into a readable incident alert card.
    """
    time_str = (
        f"{report['time_to_impact_hours']}h"
        if report["time_to_impact_hours"] is not None
        else f"Not predicted to reach real coastline within simulation window "
             f"(currently {report['nearest_real_coast_km']} km from nearest "
             f"real coastline point {report['nearest_real_coast_coords']})"
    )

    card = f"""
{'=' * 65}
MARINE OIL SPILL INCIDENT ALERT
{'=' * 65}
Source Image:            {source_image}
Location:                 {spill_lat:.4f}, {spill_lon:.4f}
Detection Confidence:     {report['detection_confidence']:.1%}
Estimated Spill Area:     {report['spill_area_km2']} km2
Estimated Volume:         {report['estimated_tonnes']} tonnes
                          (Bonn Agreement thickness assumption: 100 microns)

NOS-DCP Classification:   {report['nosdcp_tier']} - {report['nosdcp_tier_description']}
                          (Indian Coast Guard official tier framework)

Nearest Real Coastline:   {report['nearest_real_coast_km']} km away
                          at {report['nearest_real_coast_coords']}
Estimated Time-to-Impact: {time_str}

RESPONSE PRIORITY:        {report['response_priority']}
Recommended Action:       {report['recommended_action']}

Note: This is a decision-support alert for an authorized maritime
response/control centre. It is not a direct alert to Navy or
government emergency systems. Tonnage is estimated from spill area
using a standard thickness assumption; coastline data is real
(Natural Earth); NOS-DCP classification follows India's real,
official tiered response framework.
{'=' * 65}
"""
    return card


if __name__ == "__main__":
    sample_detection = {"spill_area_km2": 0.0504, "detection_confidence": 0.873}
    sample_drift_result = {
        "predicted_impact_point": {"lat": 14.9924, "lon": 69.9574}
    }
    sample_forward_path = [(14.9924 - i * 0.05, 69.9574, i) for i in range(13)]

    report = build_response_intelligence(sample_detection, sample_drift_result, sample_forward_path)
    print(format_alert_card("img_0003.jpg", 15.00837, 69.99302, report))