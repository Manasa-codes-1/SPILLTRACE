"""
SPILLTRACE - Final Incident Report Builder
Combines REAL outputs from Detection + Drift + Vessel Ranking + Risk
Intelligence into ONE incident report per spill, ready for the
Module 6 Dashboard.

Run from SPILLTRACE root (with venv active):
    python build_incident_report.py
"""

import os
import json

from run_full_pipeline import run_pipeline_for_spill
from modules.vessel_analysis.module3_json_loader import load_module3_output
from modules.response.response_intelligence import (
    build_response_intelligence, format_alert_card
)

DRIFT_DIR = "modules/drift"
DETECTIONS_CSV = "modules/detection/detections.csv"
OUTPUT_DIR = "outputs"


def find_all_drift_jsons():
    files = []
    for name in os.listdir(DRIFT_DIR):
        if name.startswith("module3_output_") and name.endswith(".json"):
            files.append(os.path.join(DRIFT_DIR, name))
    return files


def vessel_summary_line(r):
    vessel_id = getattr(r, "vessel_id", "UNKNOWN")
    score = getattr(r, "score", None)
    evidence = getattr(r, "evidence_level", None)
    if score is not None and evidence is not None:
        return f"{vessel_id}: {evidence} EVIDENCE (score: {score})"
    return str(vessel_id)


def build_one_incident(drift_json_path):
    m3_data = load_module3_output(drift_json_path)
    source_image = m3_data["source_image"]
    print(f"\n{'#' * 70}\nBUILDING INCIDENT REPORT FOR: {source_image}\n{'#' * 70}")

    # BACKWARD: vessel investigation (reuses your already-tested pipeline)
    ranked, ais_source = run_pipeline_for_spill(
        module3_json_path=drift_json_path,
        detections_csv_path=DETECTIONS_CSV,
    )

    # FORWARD: risk / response
    detection = {
        "spill_area_km2": m3_data["spill_area_km2"],
        "detection_confidence": m3_data["detection_confidence"],
    }
    drift_result = {
        "predicted_impact_point": {
            "lat": m3_data["impact_lat"],
            "lon": m3_data["impact_lon"],
        }
    }
    forward_path = m3_data["forward_path"]
    spill_lat = forward_path[0][0]
    spill_lon = forward_path[0][1]

    risk_report = build_response_intelligence(detection, drift_result, forward_path)
    alert_text = format_alert_card(source_image, spill_lat, spill_lon, risk_report)

    vessel_summaries = [vessel_summary_line(r) for r in ranked]

    combined = {
        "source_image": source_image,
        "spill_lat": spill_lat,
        "spill_lon": spill_lon,
        "probable_origin": {
            "lat": m3_data["origin_lat"],
            "lon": m3_data["origin_lon"],
            "hours_before": m3_data["hours_before"],
        },
        "ais_data_source": ais_source,
        "candidate_vessels": vessel_summaries,
        "risk_report": risk_report,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_tag = source_image.replace(".jpg", "")

    json_path = os.path.join(OUTPUT_DIR, f"incident_{image_tag}.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    txt_path = os.path.join(OUTPUT_DIR, f"incident_{image_tag}.txt")
    with open(txt_path, "w") as f:
        f.write("BACKWARD INVESTIGATION - CANDIDATE VESSELS\n")
        f.write(f"AIS data source: {ais_source}\n\n")
        for line in vessel_summaries:
            f.write(line + "\n")
        f.write("\n\nFORWARD RESPONSE - RISK & ALERT\n")
        f.write(alert_text)

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")

    return combined


if __name__ == "__main__":
    drift_files = find_all_drift_jsons()
    print(f"Found {len(drift_files)} drift result(s) to process.\n")

    all_incidents = []
    for path in drift_files:
        try:
            incident = build_one_incident(path)
            all_incidents.append(incident)
        except Exception as e:
            print(f"[ERROR] Failed to build incident for {path}: {e}")

    master_path = os.path.join(OUTPUT_DIR, "all_incidents.json")
    with open(master_path, "w") as f:
        json.dump(all_incidents, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"ALL INCIDENTS SAVED TO: {master_path}")
    print(f"Total incidents: {len(all_incidents)}")
    print(f"{'=' * 70}")