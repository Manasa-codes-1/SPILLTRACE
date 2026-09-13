import os
import pandas as pd
import requests
import copernicusmarine
import numpy as np
from datetime import datetime, timedelta, timezone

from drift_model import run_drift_analysis
from visualize_drift import plot_corridor


def get_all_spill_detections():
    """
    Reads REAL detection results sent by a teammate (Module 2),
    saved as detections.csv.
    Returns ALL confirmed spill detections (detected == True),
    not just the single best one - since each row may represent
    a separate, independent spill event.

    NOTE: Module 2's coordinates are currently SIMULATED (no real
    satellite georeferencing yet, per Module 2's own documentation).
    State this honestly in your presentation.
    """
    detections = pd.read_csv("detections.csv")

    required_columns = {"image", "detected", "confidence", "centroid_lat", "centroid_lon"}
    missing = required_columns - set(detections.columns)
    if missing:
        raise ValueError(
            f"detections.csv is missing expected columns: {missing}. "
            f"Ask your teammate if the column names changed."
        )

    confirmed = detections[detections["detected"] == True].sort_values(
        "confidence", ascending=False)

    if len(confirmed) == 0:
        raise ValueError("No confirmed spill detections found in detections.csv")

    spill_list = []
    for _, row in confirmed.iterrows():
        spill_list.append({
            "spill_lat": float(row["centroid_lat"]),
            "spill_lon": float(row["centroid_lon"]),
            "detection_confidence": float(row["confidence"]),
            "source_image": row["image"],
        })
    return spill_list


def get_live_wind_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kmh"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["current"]["wind_speed_10m"], data["current"]["wind_direction_10m"]


def get_live_ocean_current(lat, lon):
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "ocean_current_velocity,ocean_current_direction"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    speed_kmh = data["hourly"]["ocean_current_velocity"][0]
    direction_deg = data["hourly"]["ocean_current_direction"][0]
    return speed_kmh, direction_deg


def get_live_ocean_current_copernicus(lat, lon):
    today = datetime.now(timezone.utc)
    start = (today - timedelta(days=2)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        minimum_longitude=lon - 0.2,
        maximum_longitude=lon + 0.2,
        minimum_latitude=lat - 0.2,
        maximum_latitude=lat + 0.2,
        minimum_depth=0.0,
        maximum_depth=1.0,
        start_datetime=start,
        end_datetime=end,
    )

    uo = ds["uo"].isel(time=-1, depth=0).sel(
        latitude=lat, longitude=lon, method="nearest").values.item()
    vo = ds["vo"].isel(time=-1, depth=0).sel(
        latitude=lat, longitude=lon, method="nearest").values.item()

    speed_ms = np.sqrt(uo**2 + vo**2)
    speed_kmh = speed_ms * 3.6
    direction_deg = (np.degrees(np.arctan2(uo, vo)) + 360) % 360

    return round(speed_kmh, 2), round(direction_deg, 1)


def build_input_for_spill(detection, hours_to_simulate=12):
    """
    Given ONE detection (one spill), fetches live wind + current
    for that spill's location and builds the input dictionary.
    """
    spill_lat = detection["spill_lat"]
    spill_lon = detection["spill_lon"]

    wind_speed, wind_dir = get_live_wind_data(spill_lat, spill_lon)

    try:
        current_speed, current_dir = get_live_ocean_current_copernicus(spill_lat, spill_lon)
        current_source = "Copernicus Marine Service (official)"
    except Exception as e:
        print(f"[WARNING] Copernicus fetch failed ({e}); falling back to Open-Meteo Marine.")
        current_speed, current_dir = get_live_ocean_current(spill_lat, spill_lon)
        current_source = "Open-Meteo Marine (fallback)"

    input_data = {
        "spill_lat": spill_lat,
        "spill_lon": spill_lon,
        "current_speed_kmh": round(current_speed, 2),
        "current_dir_deg": round(current_dir, 1),
        "wind_speed_kmh": round(wind_speed, 1),
        "wind_dir_deg": round(wind_dir, 1),
        "hours_to_simulate": hours_to_simulate
    }
    return input_data, current_source


def save_drift_results(detection, input_data, result, current_source,
                        output_path="modules/drift/drift_results.csv"):
    row = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_image": detection["source_image"],
        "detection_confidence": detection["detection_confidence"],
        "spill_lat": input_data["spill_lat"],
        "spill_lon": input_data["spill_lon"],
        "current_speed_kmh": input_data["current_speed_kmh"],
        "current_dir_deg": input_data["current_dir_deg"],
        "current_source": current_source,
        "wind_speed_kmh": input_data["wind_speed_kmh"],
        "wind_dir_deg": input_data["wind_dir_deg"],
        "hours_simulated": input_data["hours_to_simulate"],
        "probable_origin_lat": result["probable_origin"]["lat"],
        "probable_origin_lon": result["probable_origin"]["lon"],
        "predicted_impact_lat": result["predicted_impact_point"]["lat"],
        "predicted_impact_lon": result["predicted_impact_point"]["lon"],
        "forward_confidence_radius_km": result["forward_confidence_radius_km"],
        "backward_confidence_radius_km": result["backward_confidence_radius_km"],
    }

    new_row_df = pd.DataFrame([row])

    if os.path.exists(output_path):
        new_row_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        new_row_df.to_csv(output_path, mode="w", header=True, index=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    OUTPUT_PATH = "modules/drift/drift_results.csv"

    # Start a fresh results file for this pipeline run
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"Cleared previous results: {OUTPUT_PATH}")

    all_spills = get_all_spill_detections()
    print(f"Found {len(all_spills)} confirmed spill(s) in detections.csv:\n")
    for s in all_spills:
        print(f"  {s['source_image']}: confidence={s['detection_confidence']}, "
              f"lat={s['spill_lat']}, lon={s['spill_lon']}")

    for i, detection in enumerate(all_spills, start=1):
        print(f"\n{'=' * 60}")
        print(f"PROCESSING SPILL {i}/{len(all_spills)}: {detection['source_image']}")
        print(f"{'=' * 60}")

        input_data, current_source = build_input_for_spill(detection, hours_to_simulate=12)

        print(f"  Full input data: {input_data}")
        print(f"  Ocean current source: {current_source}")

        result = run_drift_analysis(input_data)

        print(f"  Probable origin: {result['probable_origin']}")
        print(f"  Predicted impact point: {result['predicted_impact_point']}")
        print(f"  Forward confidence radius: +/-{result['forward_confidence_radius_km']} km")
        print(f"  Backward confidence radius: +/-{result['backward_confidence_radius_km']} km")

        # save a separate graph per spill, named using the source image
        image_tag = detection["source_image"].replace(".jpg", "")
        save_path = f"modules/drift/drift_corridor_{image_tag}.png"
        plot_corridor(input_data, save_path=save_path, current_source=current_source)

        save_drift_results(detection, input_data, result, current_source)

    print(f"\nAll {len(all_spills)} spill(s) processed.")