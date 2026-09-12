"""
SPILLTRACE - Module 2: Spill Analysis

Takes Module 1's detection output (mask + probability map) and produces
analysis metrics: estimated real-world area, a location, detection time,
and an honest confidence score.

IMPORTANT ASSUMPTIONS (state these clearly in your presentation/report):
- Pixel resolution (10 m/pixel) is a standard Sentinel-1 GRD approximation,
  not read from real product metadata -- this dataset has none.
- Geographic coordinates are SIMULATED. This dataset has no georeferencing.
  A production system would read real coordinates from the Sentinel-1
  GeoTIFF product using Rasterio.

Run from inside SPILLTRACE (with venv active):
    python modules/detection/spill_analysis.py
"""

import os
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import torch

from train_unet import SmallUNet, IMG_SIZE
from predict import load_model, predict, clean_mask

IMAGES_DIR = "data/satellite/train/images"

PIXEL_RESOLUTION_M = 10  # assumed Sentinel-1 GRD resolution, see note above

# Simulated reference point for demo purposes (stand-in for real georeferencing)
SIM_ORIGIN_LAT = 15.0   # example: west coast of India, open ocean
SIM_ORIGIN_LON = 70.0
SIM_DEGREES_PER_PIXEL = 0.0009  # rough demo-only conversion, not geodetically real


def analyze_spill(prob_map, cleaned_mask, image_name):
    """
    prob_map: raw model probability output, values 0-1, shape (H, W)
    cleaned_mask: post-processed binary mask, values 0/255, shape (H, W)
    """
    spill_pixels = int(np.sum(cleaned_mask == 255))
    detected = spill_pixels > 0

    result = {
        "image": image_name,
        "detected": detected,
        "detection_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if not detected:
        result.update({
            "spill_area_km2": 0.0,
            "confidence": 0.0,
            "centroid_lat": None,
            "centroid_lon": None,
        })
        return result

    # --- Area estimate ---
    area_m2 = spill_pixels * (PIXEL_RESOLUTION_M ** 2)
    area_km2 = area_m2 / 1_000_000

    # --- Confidence: mean probability WITHIN the detected region (honest, not just max) ---
    mask_bool = cleaned_mask == 255
    confidence = float(prob_map[mask_bool].mean())

    # --- Centroid location (pixel space -> simulated lat/lon) ---
    ys, xs = np.where(mask_bool)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    # Simple demo offset from a fixed reference point -- NOT real georeferencing
    centroid_lat = SIM_ORIGIN_LAT - (centroid_y - IMG_SIZE / 2) * SIM_DEGREES_PER_PIXEL
    centroid_lon = SIM_ORIGIN_LON + (centroid_x - IMG_SIZE / 2) * SIM_DEGREES_PER_PIXEL

    result.update({
        "spill_area_km2": round(area_km2, 4),
        "confidence": round(confidence, 3),
        "centroid_lat": round(centroid_lat, 5),
        "centroid_lon": round(centroid_lon, 5),
    })
    return result


def main():
    model, device = load_model()

    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")
    )[:4]

    print(f"{'Image':<15}{'Detected':<10}{'Area (km2)':<12}{'Confidence':<12}{'Lat':<10}{'Lon':<10}")
    print("-" * 70)

    all_results = []  # <-- accumulates every image's result, this was missing before

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        raw_mask, _ = predict(model, device, image_gray)
        cleaned = clean_mask(raw_mask)

        result = analyze_spill(prob_map, cleaned, img_name)
        all_results.append(result)  # <-- add this image's result to the growing list

        lat_str = f"{result['centroid_lat']}" if result["centroid_lat"] is not None else "-"
        lon_str = f"{result['centroid_lon']}" if result["centroid_lon"] is not None else "-"

        print(
            f"{result['image']:<15}{str(result['detected']):<10}"
            f"{result['spill_area_km2']:<12}{result['confidence']:<12}{lat_str:<10}{lon_str:<10}"
        )

    # build ONE table from ALL images, then save it - this now runs
    # inside main(), after the loop, where all_results actually exists
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("detections.csv", index=False)
    print("\nSaved detections.csv - send this file to your teammate")


if __name__ == "__main__":
    main()