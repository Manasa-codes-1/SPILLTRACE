"""
Fetches REAL historical AIS vessel position data from Global Fishing
Watch (GFW) - a free, research-grade AIS data source, for a given
area and time window.

Requires a free GFW API access token, stored in .env as:
    GFW_ACCESS_TOKEN=your_token_here
"""

import os
import asyncio
from datetime import timedelta

import gfwapiclient as gfw
from dotenv import load_dotenv

from .models import VesselPing

load_dotenv()
GFW_ACCESS_TOKEN = os.getenv("GFW_ACCESS_TOKEN")


async def _fetch_report(origin, token, buffer_hours):
    client = gfw.Client(access_token=token)

    start_time = origin.time_start - timedelta(hours=buffer_hours)
    end_time = origin.time_end + timedelta(hours=buffer_hours)

    lat_buffer = origin.radius_km / 111
    lon_buffer = origin.radius_km / 111

    min_lon = origin.longitude - lon_buffer
    max_lon = origin.longitude + lon_buffer
    min_lat = origin.latitude - lat_buffer
    max_lat = origin.latitude + lat_buffer

    polygon_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]]
    }

    print("Querying Global Fishing Watch for real AIS activity...")
    print(f"  Area (geojson polygon): {polygon_geojson}")
    print(f"  Time: {start_time.date()} to {end_time.date()}")

    result = await client.fourwings.create_report(
        datasets=["public-global-fishing-effort:latest"],
        spatial_resolution="HIGH",
        temporal_resolution="HOURLY",
        group_by="VESSEL_ID",
        start_date=start_time.strftime("%Y-%m-%d"),
        end_date=end_time.strftime("%Y-%m-%d"),
        geojson=polygon_geojson,
    )
    return result


def fetch_real_ais_pings(origin, access_token=None, buffer_hours=1.0):
    """
    Queries Global Fishing Watch for real vessel activity within the
    origin's search radius and time window (plus buffer).

    Returns a list of VesselPing objects - same shape the rest of
    Module 4 already expects.
    """
    token = access_token or GFW_ACCESS_TOKEN
    if not token:
        raise ValueError(
            "No GFW access token found. Make sure .env contains "
            "GFW_ACCESS_TOKEN=your_token_here"
        )

    result = asyncio.run(_fetch_report(origin, token, buffer_hours))

    print(f"[DEBUG] Raw result type: {type(result)}")
    print(f"[DEBUG] Raw result preview: {str(result)[:500]}")

    data = result.data() if hasattr(result, "data") else result

    if not isinstance(data, list):
        data = [data]

    pings = []
    for entry in data:
        if isinstance(entry, dict):
            entry_dict = entry
        elif hasattr(entry, "model_dump"):
            entry_dict = entry.model_dump()
        elif hasattr(entry, "dict"):
            entry_dict = entry.dict()
        else:
            entry_dict = entry.__dict__

        pings.append(VesselPing(
            vessel_id=str(
                entry_dict.get("vessel_id")
                or entry_dict.get("vesselId")
                or entry_dict.get("mmsi")
                or entry_dict.get("id")
                or "UNKNOWN"
            ),
            timestamp=entry_dict.get("date") or entry_dict.get("timestamp"),
            latitude=entry_dict.get("lat") or entry_dict.get("latitude"),
            longitude=entry_dict.get("lon") or entry_dict.get("longitude"),
            speed_knots=entry_dict.get("speed") or entry_dict.get("speed_knots"),
            heading_deg=entry_dict.get("heading") or entry_dict.get("heading_deg"),
        ))

    if len(pings) == 0:
        print("[NOTE] No real vessel activity found in this exact area/time - "
              "this is common for small areas in open ocean.")

    return pings