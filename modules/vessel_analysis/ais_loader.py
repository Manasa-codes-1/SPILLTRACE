"""
Module 4 - Vessel Investigation
Loads and cleans historical AIS data from a CSV file.

Expected CSV columns:
    vessel_id, timestamp, latitude, longitude, speed_knots, heading_deg

(speed_knots and heading_deg are optional)

This file only knows how to read AIS data. It does not know anything
about oil spills, drift models, or ranking. That separation means you
can later swap this out for a live AIS feed or a different data source
without touching any other part of SPILLTRACE.
"""

import pandas as pd
from .models import VesselPing

REQUIRED_COLUMNS = ["vessel_id", "timestamp", "latitude", "longitude"]


def load_ais_csv(filepath: str) -> pd.DataFrame:
    """Load an AIS CSV file into a cleaned pandas DataFrame."""
    df = pd.read_csv(filepath)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"AIS file is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    before = len(df)
    df = df.dropna(subset=["vessel_id", "timestamp", "latitude", "longitude"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[ais_loader] Dropped {dropped} row(s) with missing critical fields.")

    df = df.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
    return df


def dataframe_to_pings(df: pd.DataFrame) -> list:
    """Convert a cleaned AIS DataFrame into a list of VesselPing objects."""
    pings = []
    has_speed = "speed_knots" in df.columns
    has_heading = "heading_deg" in df.columns

    for row in df.itertuples(index=False):
        pings.append(
            VesselPing(
                vessel_id=str(row.vessel_id),
                timestamp=row.timestamp.to_pydatetime(),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                speed_knots=float(row.speed_knots) if has_speed and pd.notna(row.speed_knots) else None,
                heading_deg=float(row.heading_deg) if has_heading and pd.notna(row.heading_deg) else None,
            )
        )
    return pings
