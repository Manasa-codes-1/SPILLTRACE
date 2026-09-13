"""
SPILLTRACE - Full Pipeline Runner
Runs Module 3 (Drift Intelligence) output through Module 4 (Vessel
Investigation) and Module 5 (Vessel Ranking), end to end, on one laptop.

Uses REAL historical AIS data from Global Fishing Watch when available;
falls back to clearly-labeled sample data if the real API call fails
or returns no vessels for this specific area/time.

Run from inside SPILLTRACE (with venv active):
    python run_full_pipeline.py
"""

import random
from datetime import timedelta

from modules.vessel_analysis.module3_json_loader import load_module3_output
from modules.vessel_analysis.module3_connector import build_origin_estimate_from_drift_results
from modules.vessel_analysis.models import VesselPing
from modules.vessel_analysis.candidate_filter import find_candidate_vessels
from modules.vessel_analysis.gfw_ais_loader import fetch_real_ais_pings
from modules.vessel_ranking.scorer import rank_candidates


def generate_sample_ais_pings(origin, num_close=2, num_far=2, num_outside=1):
    """
    SAMPLE/SYNTHETIC AIS DATA - fallback only, used when real GFW data
    is unavailable or empty for this area/time.
    """
    pings = []
    window_center = origin.time_start + (origin.time_end - origin.time_start) / 2

    for i in range(num_close):
        vessel_id = f"VESSEL-CLOSE-{i+1}"
        for j in range(3):
            pings.append(VesselPing(
                vessel_id=vessel_id,
                timestamp=window_center + timedelta(minutes=15 * j),
                latitude=origin.latitude + random.uniform(-0.01, 0.01),
                longitude=origin.longitude + random.uniform(-0.01, 0.01),
                speed_knots=round(random.uniform(8, 15), 1),
                heading_deg=round(random.uniform(0, 360), 1),
            ))

    for i in range(num_far):
        vessel_id = f"VESSEL-FAR-{i+1}"
        offset_deg = (origin.radius_km * 0.8) / 111
        pings.append(VesselPing(
            vessel_id=vessel_id,
            timestamp=window_center + timedelta(hours=random.uniform(-1, 1)),
            latitude=origin.latitude + offset_deg,
            longitude=origin.longitude + offset_deg,
            speed_knots=round(random.uniform(8, 15), 1),
            heading_deg=round(random.uniform(0, 360), 1),
        ))

    for i in range(num_outside):
        pings.append(VesselPing(
            vessel_id=f"VESSEL-OUTSIDE-{i+1}",
            timestamp=window_center,
            latitude=origin.latitude + 2.0,
            longitude=origin.longitude + 2.0,
            speed_knots=10.0,
            heading_deg=90.0,
        ))

    return pings


def get_ais_pings(origin):
    """
    Tries REAL AIS data first (Global Fishing Watch). Falls back to
    clearly-labeled sample data only if the real fetch fails or
    returns nothing.
    """
    try:
        pings = fetch_real_ais_pings(origin)
        if len(pings) > 0:
            print(f"Using REAL AIS data: {len(pings)} pings from Global Fishing Watch")
            return pings, "Global Fishing Watch (real AIS data)"
        else:
            print("[FALLBACK] No real vessels found in this area/time - "
                  "using sample data for demo purposes.")
            return generate_sample_ais_pings(origin), "Sample/synthetic data (no real vessels in range)"
    except Exception as e:
        print(f"[WARNING] Real AIS fetch failed ({e}); using sample data instead.")
        return generate_sample_ais_pings(origin), "Sample/synthetic data (GFW fetch failed)"


def run_pipeline_for_spill(module3_json_path, drift_results_path):
    print(f"\n{'=' * 70}")
    print(f"LOADING MODULE 3 OUTPUT: {module3_json_path}")
    print(f"{'=' * 70}")

    m3_data = load_module3_output(module3_json_path)
    print(f"  Source image: {m3_data['source_image']}")
    print(f"  Probable origin: ({m3_data['origin_lat']}, {m3_data['origin_lon']})")
    print(f"  Hours before detection: {m3_data['hours_before']}")
    print(f"  Backward confidence radius: +/-{m3_data['backward_radius_km']} km")

    origin = build_origin_estimate_from_drift_results(
    drift_results_path,
    m3_data["source_image"],
    time_buffer_hours=1.0,
)

    print(f"\nOriginEstimate built:")
    print(f"  Location: ({origin.latitude}, {origin.longitude})")
    print(f"  Search radius: {origin.radius_km} km")
    print(f"  Time window: {origin.time_start} to {origin.time_end}")

    print(f"\n{'=' * 70}")
    print("FETCHING AIS DATA")
    print(f"{'=' * 70}")
    pings, ais_source = get_ais_pings(origin)
    print(f"  AIS data source used: {ais_source}")
    print(f"  Total pings: {len(pings)} across "
          f"{len(set(p.vessel_id for p in pings))} vessel(s)")

    print(f"\n{'=' * 70}")
    print("MODULE 4: FINDING CANDIDATE VESSELS")
    print(f"{'=' * 70}")
    candidates = find_candidate_vessels(origin, pings)
    print(f"  Found {len(candidates)} candidate vessel(s) near the probable origin")
    for c in candidates:
        print(f"    {c.vessel_id}: {c.min_distance_km:.2f} km away, "
              f"{len(c.pings_in_window)} ping(s), quality={c.data_quality}")

    print(f"\n{'=' * 70}")
    print("MODULE 5: RANKING CANDIDATES BY EVIDENCE")
    print(f"{'=' * 70}")
    ranked = rank_candidates(origin, candidates)
    for r in ranked:
        r.print_report()

    return ranked, ais_source


if __name__ == "__main__":
    run_pipeline_for_spill(
        module3_json_path="modules/drift/module3_output_img_0003.json",
        drift_results_path="modules/drift/drift_results.csv",
    )