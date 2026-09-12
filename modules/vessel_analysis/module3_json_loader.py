"""
Module 4 - Vessel Investigation
Loads Module 3's output from a JSON file, so your friend's updated
numbers can be picked up automatically without editing any Python code.

Expected JSON file format (module3_output.json):
{
    "source_image": "img_0003.jpg",
    "origin_lat": 15.0282,
    "origin_lon": 70.0226,
    "hours_before": 12,
    "backward_radius_km": 1.0
}
"""

import json


def load_module3_output(filepath: str) -> dict:
    """
    Read Module 3's output JSON file and return it as a plain dict,
    ready to pass into build_origin_estimate().
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    required = ["source_image", "origin_lat", "origin_lon", "hours_before", "backward_radius_km"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"module3_output.json is missing required fields: {missing}")

    return data
