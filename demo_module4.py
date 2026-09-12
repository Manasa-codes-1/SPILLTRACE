"""
Standalone demo for Module 4 - Vessel Investigation.

Run this file directly to test Module 4 completely on its own,
WITHOUT touching app.py or any other module:

    python demo_module4.py

This is exactly how Module 4 should be called later from app.py or
from Module 5 (ranking) - through the public interface only
(modules.vessel_analysis), never by reaching into its internal files.
"""

from datetime import datetime, timezone

from modules.vessel_analysis import (
    OriginEstimate,
    load_ais_csv,
    dataframe_to_pings,
    find_candidate_vessels,
)


def main():
    # In the real system, this OriginEstimate comes from Module 3
    # (Drift Intelligence) after it backtracks the spill's probable
    # source. For this demo we hardcode a fake origin near Goa.
    origin = OriginEstimate(
        latitude=15.30,
        longitude=73.80,
        time_start=datetime(2026, 9, 10, 6, 0, tzinfo=timezone.utc),
        time_end=datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc),
        radius_km=20.0,
        time_buffer_hours=1.0,
    )

    print("Probable origin (from Module 3):")
    print(f"  Location: ({origin.latitude}, {origin.longitude})")
    print(f"  Time window: {origin.time_start} to {origin.time_end}")
    print(f"  Search radius: {origin.radius_km} km\n")

    df = load_ais_csv("data/ais/sample_ais.csv")
    pings = dataframe_to_pings(df)
    print(f"Loaded {len(pings)} AIS pings from {df['vessel_id'].nunique()} vessels.\n")

    candidates = find_candidate_vessels(origin, pings)

    print(f"Found {len(candidates)} candidate vessel(s) near the probable origin:\n")
    for c in candidates:
        s = c.summary()
        print(
            f"  {s['vessel_id']}: "
            f"min_distance={s['min_distance_km']} km, "
            f"pings_in_window={s['num_pings_in_window']}, "
            f"data_quality={s['data_quality']}"
        )


if __name__ == "__main__":
    main()
