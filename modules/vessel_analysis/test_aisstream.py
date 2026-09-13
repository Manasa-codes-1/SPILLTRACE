import os
import json
import threading
import websocket
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AISSTREAM_API_KEY")
if not API_KEY:
    raise ValueError("AISSTREAM_API_KEY not found in .env file.")

BOUNDING_BOX = [[18.8, 72.7], [19.2, 73.0]]  # Mumbai/JNPT port - very high traffic
TIMEOUT_SECONDS = 30
MAX_VESSELS = 5

vessels_found = {}
ws_app = None


def on_open(ws):
    print("Connected to AISstream.io!")
    print(f"Listening for real ships in area {BOUNDING_BOX} (up to {TIMEOUT_SECONDS}s)...\n")
    subscribe_message = {
        "APIKey": API_KEY,
        "BoundingBoxes": [BOUNDING_BOX],
    }
    ws.send(json.dumps(subscribe_message))


def on_message(ws, message):
    data = json.loads(message)

    if data.get("MessageType") == "PositionReport":
        metadata = data.get("MetaData", {})
        report = data.get("Message", {}).get("PositionReport", {})

        mmsi = metadata.get("MMSI")
        lat = metadata.get("latitude")
        lon = metadata.get("longitude")
        ship_name = metadata.get("ShipName", "").strip() or "Unknown Vessel"
        speed = report.get("Sog", 0)
        heading = report.get("Cog", 0)

        if mmsi and lat is not None and lon is not None:
            if mmsi not in vessels_found:
                vessels_found[mmsi] = {
                    "name": ship_name,
                    "mmsi": mmsi,
                    "lat": lat,
                    "lon": lon,
                    "speed_knots": speed,
                    "heading_deg": heading,
                }
                print(f"REAL VESSEL FOUND:")
                print(f"  Name:    {ship_name}")
                print(f"  MMSI:    {mmsi}")
                print(f"  Lat/Lon: ({lat:.4f}, {lon:.4f})")
                print(f"  Speed:   {speed} knots | Heading: {heading}°")
                print()

            if len(vessels_found) >= MAX_VESSELS:
                print(f"Captured {MAX_VESSELS} real vessels. Closing connection.")
                ws.close()


def on_error(ws, error):
    print(f"[ERROR] {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"\nConnection closed. Total unique real vessels captured: {len(vessels_found)}")
    if len(vessels_found) == 0:
        print("No real ships were found in this area/time window.")
        print("This can genuinely happen in open ocean with sparse AIS receiver coverage.")


def force_timeout():
    print(f"\n[TIMEOUT] {TIMEOUT_SECONDS}s reached with no more data. Closing connection now.")
    if ws_app:
        ws_app.close()


if __name__ == "__main__":
    ws_app = websocket.WebSocketApp(
        "wss://stream.aisstream.io/v0/stream",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # Force-close after TIMEOUT_SECONDS no matter what
    timer = threading.Timer(TIMEOUT_SECONDS, force_timeout)
    timer.daemon = True
    timer.start()

    ws_app.run_forever()