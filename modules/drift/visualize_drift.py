import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from drift_model import (simulate_drift, backtrack_origin, simulate_corridor,
                          summarize_corridor, deg_to_rad)

COLOR_BG = "#EAF4FB"
COLOR_FORWARD = "#F97316"
COLOR_BACKWARD = "#7C3AED"
COLOR_SPILL = "#DC2626"
COLOR_TEXT = "#1F2937"
COLOR_GRID = "#FFFFFF"
COLOR_CARD_FORWARD = "#FFF1E6"
COLOR_CARD_BACKWARD = "#F1E9FE"
COLOR_CARD_SPILL = "#FDEAEA"
COLOR_CARD_INFO = "#F1F5F9"


def lat_formatter(lat):
    return f"{lat:.3f}N" if lat >= 0 else f"{abs(lat):.3f}S"


def lon_formatter(lon):
    return f"{lon:.3f}E" if lon >= 0 else f"{abs(lon):.3f}W"


def draw_card(fig, x, y, w, h, title, body, accent_color, bg_color):
    card = FancyBboxPatch((x, y - h), w, h, transform=fig.transFigure,
                           boxstyle="round,pad=0.006,rounding_size=0.012",
                           linewidth=0, facecolor=bg_color, zorder=1)
    fig.patches.append(card)
    fig.text(x + 0.015, y - 0.028, title, fontsize=10.5, fontweight="bold",
              color=accent_color, family="sans-serif")
    fig.text(x + 0.015, y - 0.055, body, fontsize=9.5, va="top",
              color=COLOR_TEXT, linespacing=1.7, family="sans-serif")


def plot_corridor(input_data, save_path="drift_corridor.png", current_source="Unknown"):
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
                                          wind_speed, wind_dir, total_hours, num_simulations=30)
    backward_corridor = simulate_corridor(spill_lat, spill_lon, current_speed, current_dir,
                                           wind_speed, wind_dir, total_hours,
                                           num_simulations=30, backward=True)

    fwd_avg_lat, fwd_avg_lon, fwd_spread = summarize_corridor(forward_corridor)
    bwd_avg_lat, bwd_avg_lon, bwd_spread = summarize_corridor(backward_corridor)

    total_drift_km = np.sqrt(
        ((fwd_avg_lat - spill_lat) * 111) ** 2 +
        ((fwd_avg_lon - spill_lon) * 111 * np.cos(deg_to_rad(spill_lat))) ** 2
    )
    total_backtrack_km = np.sqrt(
        ((bwd_avg_lat - spill_lat) * 111) ** 2 +
        ((bwd_avg_lon - spill_lon) * 111 * np.cos(deg_to_rad(spill_lat))) ** 2
    )

    fig = plt.figure(figsize=(16, 9.5))
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.055, 0.09, 0.58, 0.80])
    ax.set_facecolor(COLOR_BG)

    all_lats = [spill_lat] + [p[0] for p in forward_path] + [p[0] for p in backward_path]
    all_lons = [spill_lon] + [p[1] for p in forward_path] + [p[1] for p in backward_path]
    lat_pad = max((max(all_lats) - min(all_lats)) * 0.3, 0.03)
    lon_pad = max((max(all_lons) - min(all_lons)) * 0.3, 0.03)
    ax.set_xlim(min(all_lons) - lon_pad, max(all_lons) + lon_pad)
    ax.set_ylim(min(all_lats) - lat_pad, max(all_lats) + lat_pad)

    for path in forward_corridor:
        ax.plot([p[1] for p in path], [p[0] for p in path],
                color=COLOR_FORWARD, alpha=0.10, linewidth=1.1, zorder=1)
    for path in backward_corridor:
        ax.plot([p[1] for p in path], [p[0] for p in path],
                color=COLOR_BACKWARD, alpha=0.10, linewidth=1.1, zorder=1)

    fwd_lat_r = fwd_spread / 111
    fwd_lon_r = fwd_spread / (111 * np.cos(deg_to_rad(fwd_avg_lat)))
    bwd_lat_r = bwd_spread / 111
    bwd_lon_r = bwd_spread / (111 * np.cos(deg_to_rad(bwd_avg_lat)))
    ax.add_patch(Ellipse((fwd_avg_lon, fwd_avg_lat), width=2 * fwd_lon_r, height=2 * fwd_lat_r,
                          facecolor=COLOR_FORWARD, alpha=0.18, edgecolor=COLOR_FORWARD,
                          linewidth=1.5, linestyle="--", zorder=2))
    ax.add_patch(Ellipse((bwd_avg_lon, bwd_avg_lat), width=2 * bwd_lon_r, height=2 * bwd_lat_r,
                          facecolor=COLOR_BACKWARD, alpha=0.18, edgecolor=COLOR_BACKWARD,
                          linewidth=1.5, linestyle="--", zorder=2))

    outline = [pe.Stroke(linewidth=5, foreground="white"), pe.Normal()]
    ax.plot([p[1] for p in forward_path], [p[0] for p in forward_path],
            color=COLOR_FORWARD, linewidth=3, zorder=3, solid_capstyle="round",
            path_effects=outline)
    ax.plot([p[1] for p in backward_path], [p[0] for p in backward_path],
            color=COLOR_BACKWARD, linewidth=3, zorder=3, solid_capstyle="round",
            path_effects=outline)

    label_hours = sorted(set([0, total_hours // 2, total_hours]))
    for lat, lon, hour in forward_path:
        if hour in label_hours:
            ax.scatter(lon, lat, color=COLOR_FORWARD, s=55, zorder=4,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(f"+{hour}h", (lon, lat), textcoords="offset points",
                        xytext=(7, 7), fontsize=8.5, color=COLOR_FORWARD, fontweight="bold")
    for lat, lon, hour in backward_path:
        if hour in label_hours:
            ax.scatter(lon, lat, color=COLOR_BACKWARD, s=55, zorder=4,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(f"-{hour}h", (lon, lat), textcoords="offset points",
                        xytext=(7, -13), fontsize=8.5, color=COLOR_BACKWARD, fontweight="bold")

    ax.scatter([spill_lon], [spill_lat], color=COLOR_SPILL, s=380, marker="*",
               zorder=6, edgecolor="white", linewidth=1.2)
    ax.scatter([fwd_avg_lon], [fwd_avg_lat], color=COLOR_FORWARD, s=140, marker="X",
               zorder=5, edgecolor="white", linewidth=1.2)
    ax.scatter([bwd_avg_lon], [bwd_avg_lat], color=COLOR_BACKWARD, s=140, marker="X",
               zorder=5, edgecolor="white", linewidth=1.2)

    ax.set_xlabel("Longitude", fontsize=10, color=COLOR_TEXT)
    ax.set_ylabel("Latitude", fontsize=10, color=COLOR_TEXT)
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax.grid(True, color=COLOR_GRID, linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], color=COLOR_FORWARD, lw=3, label="Predicted future path"),
        Line2D([0], [0], color=COLOR_BACKWARD, lw=3, label="Estimated backward path"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_SPILL,
               markeredgecolor="white", markersize=15, label="Detected spill location"),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                        ncol=3, fontsize=9.5, frameon=False)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)

    fig.text(0.055, 0.955, "SPILLTRACE", fontsize=20, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.055, 0.925, "Drift Intelligence  |  Source Estimation & Impact Prediction",
              fontsize=11.5, color="#6B7280")

    impact_body = (
        f"Distance          ~{total_drift_km:.1f} km\n"
        f"Time horizon      {total_hours}h forward\n"
        f"Location          {lat_formatter(fwd_avg_lat)}, {lon_formatter(fwd_avg_lon)}\n"
        f"Confidence        +/-{fwd_spread:.1f} km"
    )
    draw_card(fig, 0.68, 0.90, 0.29, 0.165, "PREDICTED IMPACT ZONE",
              impact_body, COLOR_FORWARD, COLOR_CARD_FORWARD)

    origin_body = (
        f"Distance          ~{total_backtrack_km:.1f} km\n"
        f"Time horizon      {total_hours}h before detection\n"
        f"Location          {lat_formatter(bwd_avg_lat)}, {lon_formatter(bwd_avg_lon)}\n"
        f"Confidence        +/-{bwd_spread:.1f} km"
    )
    draw_card(fig, 0.68, 0.715, 0.29, 0.165, "PROBABLE ORIGIN",
              origin_body, COLOR_BACKWARD, COLOR_CARD_BACKWARD)

    spill_body = (
        f"Location          {lat_formatter(spill_lat)}, {lon_formatter(spill_lon)}\n"
        f"Source            Satellite detection (Module 1/2)"
    )
    draw_card(fig, 0.68, 0.53, 0.29, 0.10, "SPILL DETECTED",
              spill_body, COLOR_SPILL, COLOR_CARD_SPILL)

    param_body = (
        f"Ocean current     {current_speed} km/h @ {current_dir} deg\n"
        f"Wind              {wind_speed} km/h @ {wind_dir} deg\n"
        f"Direction ref.    0=N, 90=E, 180=S, 270=W\n"
        f"Current source    {current_source}\n"
        f"Wind source       Open-Meteo (live)\n"
        f"Simulations       30 runs/direction\n"
        f"Variation         +/-10% speed, +/-15 deg dir."
    )
    draw_card(fig, 0.68, 0.375, 0.29, 0.235, "SIMULATION PARAMETERS",
              param_body, "#374151", COLOR_CARD_INFO)

    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    test_input = {
        "spill_lat": 15.5, "spill_lon": 73.8,
        "current_speed_kmh": 1.5, "current_dir_deg": 45,
        "wind_speed_kmh": 20, "wind_dir_deg": 90,
        "hours_to_simulate": 12
    }
    plot_corridor(test_input, current_source="Sample/test data")