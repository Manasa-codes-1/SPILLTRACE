import os
import json
import math
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GFW_ACCESS_TOKEN")
if not TOKEN:
    raise ValueError("GFW_ACCESS_TOKEN not found in .env")

URL = "https://gateway.api.globalfishingwatch.org/v3/events"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# For demo we use fixed real 2024 dates because GFW's latest real data is 2024
# This avoids your system clock being 2026
FAKE_START_DATE = "2024-01-01"
FAKE_END_DATE = "2024-12-31"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_polygon(lat, lon, radius_km=50):
    # approx degrees for radius_km
    d_lat = radius_km / 111.0
    d_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - d_lon, lat - d_lat],
            [lon + d_lon, lat - d_lat],
            [lon + d_lon, lat + d_lat],
            [lon - d_lon, lat + d_lat],
            [lon - d_lon, lat - d_lat]
        ]]
    }

def fetch_real_vessels(lat, lon, radius_km=75, limit=50):
    geometry = build_polygon(lat, lon, radius_km)
    
    # Loitering is most relevant for oil spill - vessel sitting still
    body = {
        "datasets": ["public-global-loitering-events:latest"],
        "startDate": FAKE_START_DATE,
        "endDate": FAKE_END_DATE,
        "geometry": geometry,
    }
    params = {"limit": limit, "offset": 0}
    
    resp = requests.post(URL, headers=HEADERS, json=body, params=params, timeout=90)
    print(f"Querying GFW for real vessels near ({lat}, {lon}) radius {radius_km}km...")
    print(f"Status: {resp.status_code}, Total events in area: {resp.json().get('total', 0) if resp.status_code==201 else 'error'}")
    
    if resp.status_code != 201:
        print(resp.text[:1000])
        return []
    
    data = resp.json()
    candidates = []
    for entry in data.get("entries", []):
        pos = entry.get("position", {})
        vessel = entry.get("vessel", {})
        elat = pos.get("lat")
        elon = pos.get("lon")
        if elat is None or elon is None:
            continue
        dist = haversine(lat, lon, elat, elon)
        candidates.append({
            "vessel_id": vessel.get("id"),
            "vessel_name": vessel.get("name", "Unknown"),
            "ssvid_mmsi": vessel.get("ssvid"),
            "flag": vessel.get("flag"),
            "event_type": entry.get("type"),
            "event_start": entry.get("start"),
            "event_end": entry.get("end"),
            "event_lat": elat,
            "event_lon": elon,
            "distance_km_from_origin": round(dist, 2),
            "loitering_hours": entry.get("loitering", {}).get("totalTimeHours"),
            "avg_speed_knots": entry.get("loitering", {}).get("averageSpeedKnots"),
        })
    
    # sort closest first, deduplicate by vessel_id keep closest
    candidates = sorted(candidates, key=lambda x: x["distance_km_from_origin"])
    seen = {}
    unique = []
    for c in candidates:
        vid = c["vessel_id"]
        if vid not in seen:
            seen[vid] = True
            unique.append(c)
    return unique

if __name__ == "__main__":
    # Try to read your drift results if exists
    drift_path = "modules/drift/drift_results.csv"
    spills = []
    if os.path.exists(drift_path):
        df = pd.read_csv(drift_path)
        # use last run for each image
        for _, row in df.tail(3).iterrows():
            spills.append({
                "source_image": row["source_image"],
                "lat": float(row["probable_origin_lat"]),
                "lon": float(row["probable_origin_lon"]),
                "radius": float(row.get("backward_confidence_radius_km", 1.5)) + 50 # expand for demo to get real events
            })
    else:
        # fallback to your 3 known spills
        spills = [
            {"source_image": "img_0003.jpg", "lat": 15.0279, "lon": 70.0302, "radius": 75},
            {"source_image": "img_0002.jpg", "lat": 15.0136, "lon": 70.0089, "radius": 75},
            {"source_image": "img_0004.jpg", "lat": 15.0267, "lon": 70.0163, "radius": 75},
        ]

    all_candidates = []
    for spill in spills:
        print("\n" + "="*70)
        print(f"SPILL: {spill['source_image']} -> Probable Origin ({spill['lat']}, {spill['lon']})")
        vessels = fetch_real_vessels(spill["lat"], spill["lon"], radius_km=spill["radius"])
        print(f"Found {len(vessels)} unique REAL vessel(s)")
        for v in vessels[:10]:
            print(f"  - {v['vessel_name']} | MMSI:{v['ssvid_mmsi']} | {v['distance_km_from_origin']}km away | Loitered {v['loitering_hours']:.1f}h | Speed {v['avg_speed_knots']:.2f}kts | {v['event_start']}")
            v["source_image"] = spill["source_image"]
            v["probable_origin_lat"] = spill["lat"]
            v["probable_origin_lon"] = spill["lon"]
            all_candidates.append(v)
    
    if all_candidates:
        out_df = pd.DataFrame(all_candidates)
        out_path = "modules/vessel_analysis/candidate_vessels_real.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nReal vessel candidates saved to {out_path}")
    else:
        print("\nNo real vessels found - try increasing radius_km")