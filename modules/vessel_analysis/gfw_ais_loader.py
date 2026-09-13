"""
Fetches REAL historical AIS vessel position/activity data from Global
Fishing Watch (GFW) for a given area and time window.

Requires a free GFW API access token, stored in .env as:

    GFW_ACCESS_TOKEN=your_token_here

Important:
- Module 3's radius remains the scientific drift confidence radius.
- Module 4 may use a larger investigation radius to discover vessels
  near the probable origin.
"""

import os
import asyncio
import math
from datetime import datetime, timedelta

import gfwapiclient as gfw
from dotenv import load_dotenv

from .models import VesselPing


load_dotenv()

GFW_ACCESS_TOKEN = os.getenv("GFW_ACCESS_TOKEN")


async def _fetch_report(
    origin,
    token,
    buffer_hours,
    investigation_radius_km,
):
    """Fetch vessel activity report from Global Fishing Watch."""

    client = gfw.Client(access_token=token)

    # Expand Module 3's time window slightly for investigation.
    start_time = origin.time_start - timedelta(hours=buffer_hours)
    end_time = origin.time_end + timedelta(hours=buffer_hours)

    # Module 3 confidence radius and Module 4 investigation radius
    # are intentionally separate.
    search_radius_km = max(
        float(origin.radius_km),
        float(investigation_radius_km),
    )

    # Convert kilometres to latitude/longitude degrees.
    lat_buffer = search_radius_km / 111.0

    latitude_cos = math.cos(math.radians(origin.latitude))

    # Prevent division problems near poles.
    latitude_cos = max(abs(latitude_cos), 0.01)

    lon_buffer = search_radius_km / (
        111.0 * latitude_cos
    )

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
    print(
        f"  Module 3 confidence radius: "
        f"{origin.radius_km:.1f} km"
    )
    print(
        f"  Module 4 investigation radius: "
        f"{search_radius_km:.1f} km"
    )
    print(f"  Area (geojson polygon): {polygon_geojson}")
    print(
        f"  Time window: "
        f"{start_time.isoformat()} to {end_time.isoformat()}"
    )

    result = await client.fourwings.create_report(
        datasets=[
            "public-global-fishing-effort:latest"
        ],
        spatial_resolution="HIGH",
        temporal_resolution="HOURLY",
        group_by="VESSEL_ID",
        start_date=start_time.strftime("%Y-%m-%d"),
        end_date=end_time.strftime("%Y-%m-%d"),
        geojson=polygon_geojson,
    )

    return result


def _entry_to_dict(entry):
    """Convert a GFW response object into a plain dictionary."""

    if isinstance(entry, dict):
        return entry

    if hasattr(entry, "model_dump"):
        return entry.model_dump()

    if hasattr(entry, "dict"):
        return entry.dict()

    if hasattr(entry, "__dict__"):
        return entry.__dict__

    return {}


def _convert_timestamp(timestamp):
    """Convert timestamp values into datetime objects where possible."""

    if isinstance(timestamp, datetime):
        return timestamp

    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )
        except ValueError:
            return timestamp

    return timestamp


def fetch_real_ais_pings(
    origin,
    access_token=None,
    buffer_hours=2.0,
    investigation_radius_km=15.0,
):
    """
    Query Global Fishing Watch for real vessel activity.

    Parameters
    ----------
    origin:
        OriginEstimate from Module 3.

    access_token:
        Optional GFW token. If omitted, uses GFW_ACCESS_TOKEN from .env.

    buffer_hours:
        Extra time before and after the Module 3 origin window.

    investigation_radius_km:
        Search radius used ONLY for discovering potentially relevant
        vessels. Module 3's scientific confidence radius is not changed.

    Returns
    -------
    list[VesselPing]
        Real vessel records when available.
    """

    token = access_token or GFW_ACCESS_TOKEN

    if not token:
        raise ValueError(
            "No GFW access token found. Make sure .env contains "
            "GFW_ACCESS_TOKEN=your_token_here"
        )

    result = asyncio.run(
        _fetch_report(
            origin=origin,
            token=token,
            buffer_hours=buffer_hours,
            investigation_radius_km=investigation_radius_km,
        )
    )

    # ----------------------------------------------------------
    # DEBUG: inspect the actual GFW response structure
    # ----------------------------------------------------------

    print(f"[DEBUG] Raw result type: {type(result)}")
    print(f"[DEBUG] Raw result dir: {dir(result)}")

    if hasattr(result, "model_dump"):
        debug_data = result.model_dump()

    elif hasattr(result, "dict"):
        debug_data = result.dict()

    elif hasattr(result, "__dict__"):
        debug_data = result.__dict__

    else:
        debug_data = str(result)

    print("[DEBUG] Result structure:")
    print(debug_data)

    # ----------------------------------------------------------
    # Extract data from the GFW result
    # ----------------------------------------------------------

    if hasattr(result, "data"):

        try:
            data = result.data()

        except TypeError:
            # Some SDK versions expose data as a property.
            data = result.data

    else:
        data = result

    if data is None:
        print("[NOTE] GFW returned no data.")
        return []

    # Convert response objects to dictionaries if necessary.
    if isinstance(data, dict):

        # Sometimes APIs wrap actual records inside a key.
        possible_keys = [
            "data",
            "entries",
            "results",
            "vessels",
            "features",
        ]

        found_list = None

        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                found_list = data[key]
                break

        if found_list is not None:
            data = found_list

        else:
            data = [data]

    elif not isinstance(data, list):

        # Try converting model/object.
        data_dict = _entry_to_dict(data)

        possible_keys = [
            "data",
            "entries",
            "results",
            "vessels",
            "features",
        ]

        found_list = None

        for key in possible_keys:

            if (
                key in data_dict
                and isinstance(data_dict[key], list)
            ):
                found_list = data_dict[key]
                break

        if found_list is not None:
            data = found_list

        else:
            data = [data]

    # ----------------------------------------------------------
    # Convert usable records into VesselPing objects
    # ----------------------------------------------------------

    pings = []

    for entry in data:

        entry_dict = _entry_to_dict(entry)

        # Handle GeoJSON Feature format.
        if (
            "properties" in entry_dict
            and "geometry" in entry_dict
        ):

            properties = entry_dict.get(
                "properties",
                {}
            )

            geometry = entry_dict.get(
                "geometry",
                {}
            )

            coordinates = geometry.get(
                "coordinates",
                []
            )

            if len(coordinates) >= 2:

                longitude = coordinates[0]
                latitude = coordinates[1]

            else:

                longitude = None
                latitude = None

            vessel_id = (
                properties.get("vessel_id")
                or properties.get("vesselId")
                or properties.get("mmsi")
                or properties.get("id")
            )

            timestamp = (
                properties.get("date")
                or properties.get("timestamp")
            )

            speed = (
                properties.get("speed")
                or properties.get("speed_knots")
            )

            heading = (
                properties.get("heading")
                or properties.get("heading_deg")
            )

        else:

            vessel_id = (
                entry_dict.get("vessel_id")
                or entry_dict.get("vesselId")
                or entry_dict.get("mmsi")
                or entry_dict.get("id")
            )

            timestamp = (
                entry_dict.get("date")
                or entry_dict.get("timestamp")
            )

            latitude = (
                entry_dict.get("lat")
                or entry_dict.get("latitude")
            )

            longitude = (
                entry_dict.get("lon")
                or entry_dict.get("longitude")
            )

            speed = (
                entry_dict.get("speed")
                or entry_dict.get("speed_knots")
            )

            heading = (
                entry_dict.get("heading")
                or entry_dict.get("heading_deg")
            )

        # Skip incomplete records.
        if (
            vessel_id is None
            or timestamp is None
            or latitude is None
            or longitude is None
        ):
            continue

        try:

            ping = VesselPing(
                vessel_id=str(vessel_id),
                timestamp=_convert_timestamp(timestamp),
                latitude=float(latitude),
                longitude=float(longitude),
                speed_knots=(
                    float(speed)
                    if speed is not None
                    else None
                ),
                heading_deg=(
                    float(heading)
                    if heading is not None
                    else None
                ),
            )

            pings.append(ping)

        except (
            TypeError,
            ValueError,
        ) as e:

            print(
                f"[DEBUG] Skipping unusable "
                f"GFW record: {e}"
            )

    if len(pings) == 0:

        print(
            "[NOTE] No usable real vessel records were returned "
            "for this investigation area/time."
        )

    else:

        print(
            f"[INFO] Converted {len(pings)} "
            f"real AIS record(s) from GFW."
        )

    return pings