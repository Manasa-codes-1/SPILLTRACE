"""
SPILLTRACE - Real Vessel Ranking
Takes REAL vessel data (from Global Fishing Watch, saved by
gfw_real_ais.py) and ranks candidates for each spill using
Module 5's real scoring logic.

NOTE: GFW's public loitering-events dataset covers 2024 (its most
recent available year), while our spill's estimated time window is
based on the real current date. So this ranks vessels by REAL
spatial proximity and REAL loitering behavior in the exact spill
area, not by exact time-of-incident matching - a real limitation of
using free public data for a live/recent incident, stated honestly.
"""

import pandas as pd
from datetime import datetime, timezone

from modules.vessel_analysis.models import OriginEstimate, CandidateVessel, VesselPing
from modules.vessel_ranking.scorer import rank_candidates


def load_real_candidates_for_spill(csv_path, source_image, origin_lat, origin_lon, radius_km):
    df = pd.read_csv(csv_path)
    df = df[df["source_image"] == source_image]

    candidates = []
    for vessel_id, group in df.groupby("vessel_id"):
        pings = []
        for _, row in group.iterrows():
            pings.append(VesselPing(
                vessel_id=str(vessel_id),
                timestamp=datetime.fromisoformat(row["event_start"].replace("Z", "+00:00")),
                latitude=row["event_lat"],
                longitude=row["event_lon"],
                speed_knots=row["avg_speed_knots"],
            ))

        min_dist = group["distance_km_from_origin"].min()
        num_pings = len(pings)
        # simple quality heuristic based on how many real events we have for this vessel
        if num_pings >= 2:
            quality = "good"
        else:
            quality = "sparse"

        candidates.append(CandidateVessel(
            vessel_id=str(group.iloc[0]["vessel_name"]) + f" (MMSI {vessel_id})",
            pings_in_window=pings,
            min_distance_km=float(min_dist),
            closest_ping_time=pings[0].timestamp,
            data_quality=quality,
        ))

    return candidates


if __name__ == "__main__":
    spills = [
        {"source_image": "img_0003.jpg", "lat": 15.0235, "lon": 70.0278, "radius": 51.0},
        {"source_image": "img_0002.jpg", "lat": 15.0103, "lon": 70.0056, "radius": 50.8},
        {"source_image": "img_0004.jpg", "lat": 15.0216, "lon": 70.0168, "radius": 51.0},
    ]

    for spill in spills:
        print(f"\n{'=' * 70}")
        print(f"SPILL: {spill['source_image']} - Probable origin ({spill['lat']}, {spill['lon']})")
        print(f"{'=' * 70}")

        # dummy time window - not used for filtering here, only required
        # by the OriginEstimate/scorer's data structure
        now = datetime.now(timezone.utc)
        origin = OriginEstimate(
            latitude=spill["lat"],
            longitude=spill["lon"],
            time_start=now,
            time_end=now,
            radius_km=spill["radius"],
        )

        candidates = load_real_candidates_for_spill(
            "modules/vessel_analysis/candidate_vessels_real.csv",
            spill["source_image"],
            spill["lat"], spill["lon"], spill["radius"]
        )

        print(f"Loaded {len(candidates)} REAL vessel(s) near this spill\n")

        ranked = rank_candidates(origin, candidates)
        for r in ranked[:10]:
            r.print_report()