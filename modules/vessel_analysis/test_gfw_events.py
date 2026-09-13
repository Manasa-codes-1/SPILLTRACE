import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GFW_ACCESS_TOKEN")

if not TOKEN:
    raise ValueError("GFW_ACCESS_TOKEN not found in .env file.")

URL = "https://gateway.api.globalfishingwatch.org/v3/events"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# Arabian Sea test region around the SPILLTRACE spill locations.
# GeoJSON uses longitude first, then latitude.
GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[
        [69.0, 14.0],
        [71.0, 14.0],
        [71.0, 16.0],
        [69.0, 16.0],
        [69.0, 14.0]
    ]]
}

DATASETS_TO_TRY = [
    "public-global-encounters-events:latest",
    "public-global-loitering-events:latest",
    "public-global-port-visits-events:latest",
]

for dataset in DATASETS_TO_TRY:

    print("\n" + "=" * 70)
    print(f"TRYING DATASET: {dataset}")
    print("=" * 70)

    body = {
        "datasets": [dataset],
        "startDate": "2024-01-01",
        "endDate": "2024-12-31",
        "geometry": GEOMETRY,
    }

    # Pagination parameters are sent in the URL.
    query_params = {
        "limit": 20,
        "offset": 0,
    }

    try:
        response = requests.post(
            URL,
            headers=HEADERS,
            params=query_params,
            json=body,
            timeout=30
        )

        print(f"Request URL: {response.url}")
        print(f"Status code: {response.status_code}")

        try:
            data = response.json()
            print(json.dumps(data, indent=2)[:5000])
        except ValueError:
            print("Response was not JSON:")
            print(response.text[:2000])

    except requests.RequestException as error:
        print(f"Request failed: {error}")