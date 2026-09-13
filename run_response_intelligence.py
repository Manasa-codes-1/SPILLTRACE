"""
SPILLTRACE - Runs Module 5 (Response Intelligence) using REAL
Module 2 + Module 3 output already saved on disk.
"""

import json
from modules.response.response_intelligence import build_response_intelligence, format_alert_card

SPILL_JSON_FILES = [
    "modules/drift/module3_output_img_0002.json",
    "modules/drift/module3_output_img_0003.json",
    "modules/drift/module3_output_img_0004.json",
]

if __name__ == "__main__":
    for json_path in SPILL_JSON_FILES:
        with open(json_path, "r") as f:
            m3_data = json.load(f)

        detection = {
            "spill_area_km2": m3_data["spill_area_km2"],
            "detection_confidence": m3_data["detection_confidence"],
        }
        drift_result = {
            "predicted_impact_point": {"lat": m3_data["impact_lat"], "lon": m3_data["impact_lon"]}
        }
        forward_path = m3_data["forward_path"]

        report = build_response_intelligence(detection, drift_result, forward_path)
        card = format_alert_card(
            m3_data["source_image"], m3_data["origin_lat"], m3_data["origin_lon"], report
        )
        print(card)