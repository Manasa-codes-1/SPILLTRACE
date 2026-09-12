import numpy as np


def deg_to_rad(deg):
    return deg * np.pi / 180


def move_point(lat, lon, speed_kmh, direction_deg, hours):
    """
    Moves a lat/lon point given a speed (km/h), direction (degrees,
    0 = North, 90 = East), and how many hours to move.
    """
    distance_km = speed_kmh * hours
    direction_rad = deg_to_rad(direction_deg)

    delta_north_km = distance_km * np.cos(direction_rad)
    delta_east_km = distance_km * np.sin(direction_rad)

    delta_lat = delta_north_km / 111
    delta_lon = delta_east_km / (111 * np.cos(deg_to_rad(lat)))

    return lat + delta_lat, lon + delta_lon


def simulate_drift(start_lat, start_lon, current_speed, current_dir,
                    wind_speed, wind_dir, total_hours, step_hours=1,
                    wind_leeway_factor=0.03):
    """
    Simulates forward oil drift over time.
    Returns a list of (lat, lon, hour) points - the predicted path.
    """
    path = [(start_lat, start_lon, 0)]
    lat, lon = start_lat, start_lon

    steps = int(total_hours / step_hours)

    for i in range(1, steps + 1):
        lat, lon = move_point(lat, lon, current_speed, current_dir, step_hours)
        lat, lon = move_point(lat, lon, wind_speed * wind_leeway_factor,
                               wind_dir, step_hours)
        path.append((lat, lon, i * step_hours))

    return path


def backtrack_origin(detected_lat, detected_lon, current_speed, current_dir,
                      wind_speed, wind_dir, hours_back, step_hours=1):
    """
    Estimates where the spill probably came from, by running the
    same drift model backward in time (direction reversed by 180 degrees).
    """
    reverse_current_dir = (current_dir + 180) % 360
    reverse_wind_dir = (wind_dir + 180) % 360

    return simulate_drift(
        detected_lat, detected_lon,
        current_speed, reverse_current_dir,
        wind_speed, reverse_wind_dir,
        total_hours=hours_back,
        step_hours=step_hours
    )


def simulate_corridor(start_lat, start_lon, current_speed, current_dir,
                       wind_speed, wind_dir, total_hours, step_hours=1,
                       num_simulations=30, variation_percent=0.10,
                       backward=False):
    """
    Runs many simulations with randomly varied wind/current values
    to produce a spread of plausible paths - the 'uncertainty corridor'.
    """
    all_paths = []

    for _ in range(num_simulations):
        varied_current_speed = current_speed * np.random.uniform(
            1 - variation_percent, 1 + variation_percent)
        varied_current_dir = current_dir + np.random.uniform(-15, 15)
        varied_wind_speed = wind_speed * np.random.uniform(
            1 - variation_percent, 1 + variation_percent)
        varied_wind_dir = wind_dir + np.random.uniform(-15, 15)

        if backward:
            path = backtrack_origin(
                start_lat, start_lon,
                varied_current_speed, varied_current_dir,
                varied_wind_speed, varied_wind_dir,
                hours_back=total_hours, step_hours=step_hours
            )
        else:
            path = simulate_drift(
                start_lat, start_lon,
                varied_current_speed, varied_current_dir,
                varied_wind_speed, varied_wind_dir,
                total_hours=total_hours, step_hours=step_hours
            )
        all_paths.append(path)

    return all_paths


def summarize_corridor(all_paths):
    """
    Given many simulated paths, returns the average final position
    and how spread out (in km) the final positions are.
    """
    final_points = [path[-1] for path in all_paths]
    lats = [p[0] for p in final_points]
    lons = [p[1] for p in final_points]

    avg_lat = np.mean(lats)
    avg_lon = np.mean(lons)

    distances_km = []
    for lat, lon in zip(lats, lons):
        dlat_km = (lat - avg_lat) * 111
        dlon_km = (lon - avg_lon) * 111 * np.cos(deg_to_rad(avg_lat))
        distances_km.append(np.sqrt(dlat_km**2 + dlon_km**2))

    spread_km = np.mean(distances_km)

    return avg_lat, avg_lon, spread_km


def run_drift_analysis(input_data):
    """
    MODULE 3 - MAIN INTERFACE FUNCTION
    This is the function other teammates/modules should call.

    Expected input_data (a dictionary):
        {
            "spill_lat": float,
            "spill_lon": float,
            "current_speed_kmh": float,
            "current_dir_deg": float,
            "wind_speed_kmh": float,
            "wind_dir_deg": float,
            "hours_to_simulate": int
        }

    Returns a dictionary:
        {
            "forward_path": [(lat, lon, hour), ...],
            "backward_path": [(lat, lon, hour), ...],
            "probable_origin": {"lat": float, "lon": float, "hours_before": int},
            "predicted_impact_point": {"lat": float, "lon": float},
            "forward_confidence_radius_km": float,
            "backward_confidence_radius_km": float,
        }
    """
    spill_lat = input_data["spill_lat"]
    spill_lon = input_data["spill_lon"]
    current_speed = input_data["current_speed_kmh"]
    current_dir = input_data["current_dir_deg"]
    wind_speed = input_data["wind_speed_kmh"]
    wind_dir = input_data["wind_dir_deg"]
    total_hours = input_data["hours_to_simulate"]

    forward_path = simulate_drift(spill_lat, spill_lon, current_speed, current_dir,
                                   wind_speed, wind_dir, total_hours)
    backward_path = backtrack_origin(spill_lat, spill_lon, current_speed, current_dir,
                                      wind_speed, wind_dir, total_hours)

    forward_corridor = simulate_corridor(spill_lat, spill_lon, current_speed, current_dir,
                                          wind_speed, wind_dir, total_hours,
                                          num_simulations=30)
    backward_corridor = simulate_corridor(spill_lat, spill_lon, current_speed, current_dir,
                                           wind_speed, wind_dir, total_hours,
                                           num_simulations=30, backward=True)

    fwd_avg_lat, fwd_avg_lon, fwd_spread = summarize_corridor(forward_corridor)
    bwd_avg_lat, bwd_avg_lon, bwd_spread = summarize_corridor(backward_corridor)

    result = {
        "forward_path": forward_path,
        "backward_path": backward_path,
        "probable_origin": {
            "lat": round(bwd_avg_lat, 4),
            "lon": round(bwd_avg_lon, 4),
            "hours_before": total_hours
        },
        "predicted_impact_point": {
            "lat": round(fwd_avg_lat, 4),
            "lon": round(fwd_avg_lon, 4)
        },
        "forward_confidence_radius_km": round(fwd_spread, 1),
        "backward_confidence_radius_km": round(bwd_spread, 1),
    }

    return result


if __name__ == "__main__":
    # ============================================================
    # SAMPLE TEST DATA ONLY - for testing this module by itself.
    # When integrated, this dictionary will instead be built from
    # real outputs of Module 1/2 (detection) and environmental data.
    # ============================================================
    test_input = {
        "spill_lat": 15.5,
        "spill_lon": 73.8,
        "current_speed_kmh": 1.5,
        "current_dir_deg": 45,
        "wind_speed_kmh": 20,
        "wind_dir_deg": 90,
        "hours_to_simulate": 12
    }

    result = run_drift_analysis(test_input)

    print("MODULE 3 OUTPUT:")
    print(f"  Probable origin: {result['probable_origin']}")
    print(f"  Predicted impact point: {result['predicted_impact_point']}")
    print(f"  Forward confidence radius: +/-{result['forward_confidence_radius_km']} km")
    print(f"  Backward confidence radius: +/-{result['backward_confidence_radius_km']} km")