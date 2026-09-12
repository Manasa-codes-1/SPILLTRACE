"""
Module 4 - Vessel Investigation
Connector: builds an OriginEstimate from Module 3's raw output plus the
Module 2 detections CSV (needed because Module 3 only gives "hours_before",
not an absolute timestamp).

Usage:
    from modules.vessel_analysis.module3_connector import build_origin_estimate

    origin = build_origin_estimate(
        detections_csv_path="data/detections/detections_csv.xls",
        source_image="img_0003.jpg",
        module3_origin={"lat": 15.0262, "lon": 70.0237, "hours_before": 12},
        backward_radius_km=1.1,
        time_buffer_hours=1.0,
    )
"""

import csv
from datetime import datetime, timedelta, timezone

from .models import OriginEstimate


def _find_detection_time(detections_csv_path: str, source_image: str) -> datetime:
    """Look up the detection_time_utc for a given source image in the CSV."""
    with open(detections_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image"] == source_image:
                ts = row["detection_time_utc"]
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    raise ValueError(f"No row found for source_image={source_image!r} in {detections_csv_path}")


def build_origin_estimate(
    detections_csv_path: str,
    source_image: str,
    module3_origin: dict,
    backward_radius_km: float,
    time_buffer_hours: float = 1.0,
) -> OriginEstimate:
    """
    Combine Module 2's detection CSV + Module 3's raw drift output into
    the OriginEstimate object Module 4 needs.

    module3_origin is expected to look like:
        {"lat": 15.0262, "lon": 70.0237, "hours_before": 12}
    (plain floats/ints - convert any np.float64 values with float()/int()
    BEFORE calling this function.)
    """
    detection_time = _find_detection_time(detections_csv_path, source_image)

    hours_before = float(module3_origin["hours_before"])
    origin_time = detection_time - timedelta(hours=hours_before)

    buffer = timedelta(hours=time_buffer_hours)

    return OriginEstimate(
        latitude=float(module3_origin["lat"]),
        longitude=float(module3_origin["lon"]),
        time_start=origin_time - buffer,
        time_end=origin_time + buffer,
        radius_km=float(backward_radius_km),
        time_buffer_hours=time_buffer_hours,
    )
