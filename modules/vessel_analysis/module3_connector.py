"""
Module 4 - Vessel Investigation

Connector between Module 3 (Drift Intelligence) and Module 4
(Vessel Investigation).

Reads Module 3's drift_results.csv and converts one spill result
into the OriginEstimate object required by Module 4.
"""

import csv
from datetime import datetime, timedelta

from .models import OriginEstimate


def build_origin_estimate_from_drift_results(
    drift_results_csv_path: str,
    source_image: str,
    time_buffer_hours: float = 1.0,
) -> OriginEstimate:
    """
    Build a Module 4 OriginEstimate directly from Module 3's
    drift_results.csv.

    Uses the most recent Module 3 result for the requested source image.

    Expected columns:
        run_timestamp_utc
        source_image
        probable_origin_lat
        probable_origin_lon
        hours_simulated
        backward_confidence_radius_km
    """

    matching_rows = []

    with open(drift_results_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "run_timestamp_utc",
            "source_image",
            "probable_origin_lat",
            "probable_origin_lon",
            "hours_simulated",
            "backward_confidence_radius_km",
        }

        if reader.fieldnames is None:
            raise ValueError("drift_results.csv has no header row")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"drift_results.csv is missing required columns: {missing}"
            )

        for row in reader:
            if row["source_image"] == source_image:
                matching_rows.append(row)

    if not matching_rows:
        raise ValueError(
            f"No Module 3 result found for source image: {source_image}"
        )

    # Use the latest run for this spill.
    latest_row = max(
        matching_rows,
        key=lambda row: datetime.fromisoformat(
            row["run_timestamp_utc"].replace("Z", "+00:00")
        ),
    )

    run_time = datetime.fromisoformat(
        latest_row["run_timestamp_utc"].replace("Z", "+00:00")
    )

    hours_before = float(latest_row["hours_simulated"])

    # Module 3 run timestamp is used as the detection/reference time.
    origin_time = run_time - timedelta(hours=hours_before)

    buffer = timedelta(hours=time_buffer_hours)

    return OriginEstimate(
        latitude=float(latest_row["probable_origin_lat"]),
        longitude=float(latest_row["probable_origin_lon"]),
        time_start=origin_time - buffer,
        time_end=origin_time + buffer,
        radius_km=float(latest_row["backward_confidence_radius_km"]),
        time_buffer_hours=time_buffer_hours,
    )