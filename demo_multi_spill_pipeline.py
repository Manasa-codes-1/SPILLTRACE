"""
Module 4 + 5 - MULTI-SPILL PIPELINE
Runs the exact same real pipeline as demo_real_pipeline.py, but loops
over EVERY spill in your teammate's drift_results.csv instead of a
single hardcoded module3_output.json.

Nothing in Module 4 (modules/vessel_analysis) or Module 5
(modules/vessel_ranking) changes - this script just calls the same
public functions once per spill, exactly like demo_real_pipeline.py
already does for one.

Run:
    python demo_multi_spill_pipeline.py
"""

import csv

from modules.vessel_analysis import (
    build_origin_estimate,
    load_ais_csv,
    dataframe_to_pings,
    find_candidate_vessels,
)
from modules.vessel_ranking import rank_candidates

# --- Inputs ---
DETECTIONS_CSV_PATH = "data/detections/detections_csv.xls"     # from Module 2 (must contain img_0002/3/4.jpg rows)
AIS_CSV_PATH = "data/ais/sample_ais.csv"                        # placeholder until real AIS data exists
DRIFT_RESULTS_CSV_PATH = "modules/drift/drift_results.csv"      # your teammate's real Module 3 output
TIME_BUFFER_HOURS = 1.0

# --- Column names, matched to the real drift_results.csv header ---
COL_RUN_TIMESTAMP = "run_timestamp_utc"
COL_SOURCE_IMAGE = "source_image"
COL_DEPTH = "selected_depth_m"          # sometimes missing - see note below
COL_ORIGIN_LAT = "probable_origin_lat"
COL_ORIGIN_LON = "probable_origin_lon"
COL_HOURS_BEFORE = "hours_simulated"
COL_BACKWARD_RADIUS_KM = "backward_confidence_radius_km"


def load_all_spills(csv_path):
    """Read drift_results.csv and return the LATEST spill per source
    image, as a dict shaped for build_origin_estimate().

    Handles two real quirks seen in your teammate's file:
      1. The file accumulates one row per run, so the same image can
         appear multiple times - we keep only the newest run per image
         (by run_timestamp_utc).
      2. Some runs omit the "selected_depth_m" field entirely, which
         shifts every column after it by one position in a plain
         positional CSV read. We detect a row that's exactly one field
         short and re-insert a placeholder in the depth slot so every
         other column lines up with its real header name again.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        depth_idx = header.index(COL_DEPTH)

        latest_by_image = {}  # source_image -> row dict
        for line_num, row in enumerate(reader, start=2):
            if len(row) == len(header):
                fixed_row = row
            elif len(row) == len(header) - 1:
                print(
                    f"  [fixup] row {line_num}: missing '{COL_DEPTH}' field - "
                    f"realigning columns"
                )
                fixed_row = row[:depth_idx] + [""] + row[depth_idx:]
            else:
                raise ValueError(
                    f"Row {line_num} has {len(row)} fields, expected "
                    f"{len(header)} (or {len(header) - 1}). Can't safely "
                    f"parse - check drift_results.csv by hand."
                )

            record = dict(zip(header, fixed_row))
            image = record[COL_SOURCE_IMAGE]

            existing = latest_by_image.get(image)
            if existing is None or record[COL_RUN_TIMESTAMP] > existing[COL_RUN_TIMESTAMP]:
                latest_by_image[image] = record

    spills = []
    for image, record in latest_by_image.items():
        spills.append({
            "source_image": image,
            "origin_lat": float(record[COL_ORIGIN_LAT]),
            "origin_lon": float(record[COL_ORIGIN_LON]),
            "hours_before": float(record[COL_HOURS_BEFORE]),
            "backward_radius_km": float(record[COL_BACKWARD_RADIUS_KM]),
        })
    return spills


def run_one_spill(spill, ais_pings):
    origin = build_origin_estimate(
        detections_csv_path=DETECTIONS_CSV_PATH,
        source_image=spill["source_image"],
        module3_origin={
            "lat": spill["origin_lat"],
            "lon": spill["origin_lon"],
            "hours_before": spill["hours_before"],
        },
        backward_radius_km=spill["backward_radius_km"],
        time_buffer_hours=TIME_BUFFER_HOURS,
    )

    print(f"\n{'=' * 60}")
    print(f"SPILL: {spill['source_image']}")
    print(f"{'=' * 60}")
    print(f"  Location: ({origin.latitude}, {origin.longitude})")
    print(f"  Time window: {origin.time_start} to {origin.time_end}")
    print(f"  Search radius: {origin.radius_km} km\n")

    candidates = find_candidate_vessels(origin, ais_pings)
    print(f"Found {len(candidates)} candidate vessel(s):\n")

    if not candidates:
        print("  (none)")
        return

    for c in candidates:
        s = c.summary()
        print(
            f"  {s['vessel_id']}: "
            f"min_distance={s['min_distance_km']} km, "
            f"pings_in_window={s['num_pings_in_window']}, "
            f"data_quality={s['data_quality']}"
        )

    print("\n--- Module 5: Vessel Ranking ---")
    ranked = rank_candidates(origin, candidates)
    for r in ranked:
        r.print_report()


def main():
    spills = load_all_spills(DRIFT_RESULTS_CSV_PATH)
    print(f"Loaded {len(spills)} spill(s) from {DRIFT_RESULTS_CSV_PATH}\n")

    df = load_ais_csv(AIS_CSV_PATH)
    pings = dataframe_to_pings(df)
    print(f"Loaded {len(pings)} AIS pings from {df['vessel_id'].nunique()} vessels.")

    for spill in spills:
        run_one_spill(spill, pings)


if __name__ == "__main__":
    main()
