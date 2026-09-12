"""
Module 4 - Vessel Investigation
REAL PIPELINE DEMO: uses your friend's actual Module 3 output (backward
drift result) plus the real Module 2 detections CSV, instead of the
hardcoded fake origin in demo_module4.py.

Run this file directly:

    python demo_real_pipeline.py

Note: this still uses SAMPLE AIS data (data/ais/sample_ais.csv) since we
don't have real AIS data yet. Once real AIS data is available, just point
AIS_CSV_PATH below at the new file - nothing else in this script changes.
"""

from modules.vessel_analysis import (
    build_origin_estimate,
    load_module3_output,
    load_ais_csv,
    dataframe_to_pings,
    find_candidate_vessels,
)
from modules.vessel_ranking import rank_candidates

# --- Inputs you'd normally get from teammates ---

DETECTIONS_CSV_PATH = "data/detections/detections_csv.xls"   # from Module 2
AIS_CSV_PATH = "data/ais/sample_ais.csv"                       # placeholder until real AIS data exists
MODULE3_JSON_PATH = "data/module3_output.json"                 # your friend updates THIS file, not this script
TIME_BUFFER_HOURS = 1.0                                        # extra safety margin around the estimated time


def main():
    # Step 1: read your friend's latest Module 3 output straight from her JSON file
    module3_data = load_module3_output(MODULE3_JSON_PATH)

    # Step 2: turn that + the detections CSV into an OriginEstimate
    origin = build_origin_estimate(
        detections_csv_path=DETECTIONS_CSV_PATH,
        source_image=module3_data["source_image"],
        module3_origin={
            "lat": module3_data["origin_lat"],
            "lon": module3_data["origin_lon"],
            "hours_before": module3_data["hours_before"],
        },
        backward_radius_km=module3_data["backward_radius_km"],
        time_buffer_hours=TIME_BUFFER_HOURS,
    )

    print("Origin estimate built from real Module 3 output:")
    print(f"  Location: ({origin.latitude}, {origin.longitude})")
    print(f"  Time window: {origin.time_start} to {origin.time_end}")
    print(f"  Search radius: {origin.radius_km} km\n")

    # Step 3: load AIS data (still sample data for now)
    df = load_ais_csv(AIS_CSV_PATH)
    pings = dataframe_to_pings(df)
    print(f"Loaded {len(pings)} AIS pings from {df['vessel_id'].nunique()} vessels.\n")

    # Step 4: find candidate vessels
    candidates = find_candidate_vessels(origin, pings)

    print(f"Found {len(candidates)} candidate vessel(s) near the real probable origin:\n")
    if not candidates:
        print("  (none - this is expected right now since the sample AIS data")
        print("   was made up for testing and doesn't line up with these real")
        print("   coordinates. Once real AIS data is loaded, this will work.)")
    for c in candidates:
        s = c.summary()
        print(
            f"  {s['vessel_id']}: "
            f"min_distance={s['min_distance_km']} km, "
            f"pings_in_window={s['num_pings_in_window']}, "
            f"data_quality={s['data_quality']}"
        )

    if not candidates:
        return

    # Step 5: rank the candidates by evidence strength (Module 5)
    print("\n--- Module 5: Vessel Ranking ---")
    ranked = rank_candidates(origin, candidates)
    for r in ranked:
        r.print_report()


if __name__ == "__main__":
    main()
