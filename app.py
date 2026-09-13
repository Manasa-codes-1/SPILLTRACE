import os
import json
import glob

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SPILLTRACE",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# LOAD MODULE 1 + MODULE 2 OUTPUT
# ============================================================

@st.cache_data
def load_detection_data():

    possible_paths = [
        "modules/detection/detections.csv",
        "detections.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    return None


# ============================================================
# LOAD COMBINED INCIDENT OUTPUTS
# ============================================================

@st.cache_data
def load_incident_data():

    incidents = []

    # Preferred: load the combined file if it exists
    combined_path = "outputs/all_incidents.json"

    if os.path.exists(combined_path):

        try:
            with open(combined_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

        except Exception as e:
            st.warning(f"Could not read {combined_path}: {e}")

    # Otherwise load individual incident files
    incident_files = sorted(
        glob.glob("outputs/incident_*.json")
    )

    for file_path in incident_files:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                incident = json.load(f)

            incidents.append(incident)

        except Exception as e:
            st.warning(
                f"Could not read {file_path}: {e}"
            )

    return incidents


# ============================================================
# LOAD DATA
# ============================================================

detections_df = load_detection_data()
incidents = load_incident_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_incident_for_image(image_name):

    for incident in incidents:

        if incident.get("source_image") == image_name:
            return incident

    return None


def get_detected_incidents():

    if detections_df is None:
        return []

    detected = detections_df[
        detections_df["detected"] == True
    ]

    return detected.to_dict("records")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌊 SPILLTRACE")

st.sidebar.caption(
    "Oil Spill Incident Intelligence System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Module 1 - Oil Spill Detection",
        "Module 2 - Spill Analysis",
        "Module 3 - Drift Intelligence",
        "Module 4 - Vessel Investigation",
        "Module 5 - Risk & Response",
    ]
)


# ============================================================
# SIDEBAR SYSTEM STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("System Status")

if detections_df is not None:
    st.sidebar.success("Module 1 & 2 Data Loaded")
else:
    st.sidebar.warning("Module 1 & 2 Data Missing")


# Check Module 3 data
module3_loaded = any(
    incident.get("probable_origin") is not None
    for incident in incidents
)

if module3_loaded:
    st.sidebar.success("Module 3 Data Loaded")
else:
    st.sidebar.warning("Module 3 Data Missing")


# Check Module 4 data
module4_loaded = any(
    incident.get("candidate_vessels") is not None
    for incident in incidents
)

if module4_loaded:
    st.sidebar.success("Module 4 Data Loaded")
else:
    st.sidebar.warning("Module 4 Data Missing")


# Check Module 5 data
module5_loaded = any(
    incident.get("risk_report") is not None
    for incident in incidents
)

if module5_loaded:
    st.sidebar.success("Module 5 Data Loaded")
else:
    st.sidebar.warning("Module 5 Data Missing")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🌊 SPILLTRACE")


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.header("System Overview")

    st.write(
        "SPILLTRACE combines AI-based oil spill detection, spill analysis, "
        "drift intelligence, vessel investigation, and risk and response "
        "intelligence into a single incident investigation pipeline."
    )

    if detections_df is None:

        st.warning(
            "Detection output file not found. "
            "Run Module 2 to generate detections.csv."
        )

    else:

        total_images = len(detections_df)

        spills_detected = int(
            detections_df["detected"].astype(bool).sum()
        )

        high_confidence = int(
            (detections_df["confidence"] >= 0.8).sum()
        )

        active_investigations = len(incidents)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Images Analyzed",
            total_images
        )

        col2.metric(
            "Spills Detected",
            spills_detected
        )

        col3.metric(
            "Incident Investigations",
            active_investigations
        )

        col4.metric(
            "High Confidence",
            high_confidence
        )

        st.divider()

        st.subheader("Incident Pipeline")

        st.markdown(
            """
🛰️ **Module 1 — Oil Spill Detection**  
Detect potential oil spill regions in satellite imagery.

↓  

🌊 **Module 2 — Spill Analysis**  
Estimate spill area, centroid location, and confidence.

↓  

🌊 **Module 3 — Drift Intelligence**  
Backtrack the spill to estimate its probable origin.

↓  

🚢 **Module 4 — Vessel Investigation**  
Investigate vessels near the probable origin.

↓  

⚠️ **Module 5 — Risk & Response**  
Assess spill severity, coastal risk, and response priority.
            """
        )

        st.divider()

        st.subheader("Current Detection Summary")

        st.dataframe(
            detections_df,
            use_container_width=True
        )

        if incidents:

            st.divider()

            st.subheader("Active Incident Investigations")

            incident_rows = []

            for incident in incidents:

                risk = incident.get(
                    "risk_report",
                    {}
                )

                incident_rows.append({
                    "Source Image": incident.get(
                        "source_image"
                    ),
                    "Probable Origin Lat": incident.get(
                        "probable_origin",
                        {}
                    ).get("lat"),
                    "Probable Origin Lon": incident.get(
                        "probable_origin",
                        {}
                    ).get("lon"),
                    "Candidate Vessels": len(
                        incident.get(
                            "candidate_vessels",
                            []
                        )
                    ),
                    "Response Priority": risk.get(
                        "response_priority"
                    ),
                    "Recommended Action": risk.get(
                        "recommended_action"
                    ),
                })

            st.dataframe(
                pd.DataFrame(incident_rows),
                use_container_width=True
            )


# ============================================================
# MODULE 1 - OIL SPILL DETECTION
# ============================================================

elif page == "Module 1 - Oil Spill Detection":

    st.header("🛰️ Module 1: Oil Spill Detection")

    st.write(
        "Module 1 uses a U-Net deep learning model to identify "
        "potential oil spill regions in satellite imagery."
    )

    st.divider()

    if detections_df is None:

        st.warning(
            "Detection results are not available."
        )

    else:

        image_dir = "data/satellite/train/images"

        for _, row in detections_df.iterrows():

            image_name = row["image"]

            image_path = os.path.join(
                image_dir,
                image_name
            )

            detected = bool(
                row["detected"]
            )

            if detected:
                status = "🌊 OIL SPILL DETECTED"
            else:
                status = "✅ NO SPILL DETECTED"

            with st.expander(
                f"{image_name} — {status}"
            ):

                col1, col2 = st.columns(
                    [2, 1]
                )

                with col1:

                    if os.path.exists(
                        image_path
                    ):

                        st.image(
                            image_path,
                            caption=(
                                f"Satellite Image: "
                                f"{image_name}"
                            ),
                            use_container_width=True
                        )

                    else:

                        st.warning(
                            f"Image not found: "
                            f"{image_path}"
                        )

                with col2:

                    st.metric(
                        "Detection Status",
                        (
                            "Detected"
                            if detected
                            else "Not Detected"
                        )
                    )

                    st.metric(
                        "Model Confidence",
                        f"{float(row['confidence']):.3f}"
                    )

                    if detected:

                        st.success(
                            "Potential oil spill region identified."
                        )

                    else:

                        st.info(
                            "No significant oil spill "
                            "region identified."
                        )


# ============================================================
# MODULE 2 - SPILL ANALYSIS
# ============================================================

elif page == "Module 2 - Spill Analysis":

    st.header("🌊 Module 2: Spill Analysis")

    st.write(
        "Module 2 converts the AI detection output into incident "
        "intelligence including estimated spill area, centroid "
        "location, confidence, and detection time."
    )

    if detections_df is None:

        st.error(
            "detections.csv was not found."
        )

    else:

        detected_spills = detections_df[
            detections_df["detected"] == True
        ]

        if len(detected_spills) == 0:

            st.info(
                "No spills are currently available for analysis."
            )

        else:

            for _, row in detected_spills.iterrows():

                with st.expander(
                    f"🌊 {row['image']} — Spill Analysis"
                ):

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Estimated Spill Area",
                        (
                            f"{float(row['spill_area_km2']):.4f} "
                            "km²"
                        )
                    )

                    col2.metric(
                        "Analysis Confidence",
                        f"{float(row['confidence']):.3f}"
                    )

                    col3.metric(
                        "Status",
                        "Confirmed Candidate"
                    )

                    st.divider()

                    coord1, coord2 = st.columns(2)

                    with coord1:

                        st.subheader("Location")

                        st.write(
                            f"**Latitude:** "
                            f"{row['centroid_lat']}"
                        )

                        st.write(
                            f"**Longitude:** "
                            f"{row['centroid_lon']}"
                        )

                    with coord2:

                        st.subheader("Detection")

                        st.write(
                            f"**Source Image:** "
                            f"{row['image']}"
                        )

                        st.write(
                            f"**Detection Time:** "
                            f"{row['detection_time_utc']}"
                        )

        st.divider()

        st.subheader("All Module 2 Results")

        st.dataframe(
            detections_df,
            use_container_width=True
        )


# ============================================================
# MODULE 3 - DRIFT INTELLIGENCE
# ============================================================

elif page == "Module 3 - Drift Intelligence":

    st.header("🌊 Module 3: Drift Intelligence")

    st.write(
        "Module 3 estimates the probable origin of a detected oil "
        "spill by backtracking its observed location."
    )

    if not incidents:

        st.warning(
            "No incident investigation outputs were found."
        )

        st.write(
            "Run the pipeline and generate incident JSON files "
            "inside the outputs folder."
        )

    else:

        for incident in incidents:

            origin = incident.get(
                "probable_origin",
                {}
            )

            source_image = incident.get(
                "source_image",
                "Unknown"
            )

            with st.expander(
                f"🌊 {source_image} — Drift Investigation",
                expanded=False
            ):

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Origin Latitude",
                    origin.get("lat", "N/A")
                )

                col2.metric(
                    "Origin Longitude",
                    origin.get("lon", "N/A")
                )

                col3.metric(
                    "Hours Before Detection",
                    origin.get(
                        "hours_before",
                        "N/A"
                    )
                )

                st.divider()

                st.subheader(
                    "Detected Spill Location"
                )

                spill_col1, spill_col2 = st.columns(2)

                spill_col1.write(
                    f"**Spill Latitude:** "
                    f"{incident.get('spill_lat')}"
                )

                spill_col2.write(
                    f"**Spill Longitude:** "
                    f"{incident.get('spill_lon')}"
                )

                st.info(
                    "The probable origin is the estimated location "
                    "from which the observed spill may have "
                    "originated before drifting."
                )


# ============================================================
# MODULE 4 - VESSEL INVESTIGATION
# ============================================================

elif page == "Module 4 - Vessel Investigation":

    st.header("🚢 Module 4: Vessel Investigation")

    st.write(
        "Module 4 investigates vessel activity near the estimated "
        "spill origin during the relevant release time window."
    )

    if not incidents:

        st.warning(
            "No Module 4 incident outputs were found."
        )

    else:

        for incident in incidents:

            source_image = incident.get(
                "source_image",
                "Unknown"
            )

            vessels = incident.get(
                "candidate_vessels",
                []
            )

            ais_source = incident.get(
                "ais_data_source",
                "Unknown"
            )

            origin = incident.get(
                "probable_origin",
                {}
            )

            with st.expander(
                f"🚢 {source_image} — Vessel Investigation",
                expanded=False
            ):

                st.subheader(
                    "Investigation Area"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Origin Latitude",
                    origin.get("lat", "N/A")
                )

                col2.metric(
                    "Origin Longitude",
                    origin.get("lon", "N/A")
                )

                col3.metric(
                    "Candidate Vessels",
                    len(vessels)
                )

                st.divider()

                st.subheader(
                    "AIS Data Source"
                )

                if (
                    "Sample" in str(ais_source)
                    or "synthetic" in str(
                        ais_source
                    ).lower()
                ):

                    st.warning(
                        ais_source
                    )

                    st.caption(
                        "The current investigation used fallback "
                        "sample/synthetic vessel data because no "
                        "usable real vessel activity was returned "
                        "for the queried area and time."
                    )

                else:

                    st.success(
                        ais_source
                    )

                st.divider()

                st.subheader(
                    "Candidate Vessels"
                )

                if not vessels:

                    st.info(
                        "No candidate vessels were identified."
                    )

                else:

                    for vessel in vessels:

                        if "HIGH EVIDENCE" in vessel:

                            st.success(
                                f"🚢 {vessel}"
                            )

                        elif "MEDIUM EVIDENCE" in vessel:

                            st.warning(
                                f"🚢 {vessel}"
                            )

                        else:

                            st.info(
                                f"🚢 {vessel}"
                            )


# ============================================================
# MODULE 5 - RISK & RESPONSE
# ============================================================

elif page == "Module 5 - Risk & Response":

    st.header(
        "⚠️ Module 5: Risk & Response Intelligence"
    )

    st.write(
        "Module 5 assesses spill severity, estimated volume, "
        "coastal impact risk, response priority, and recommended "
        "response actions."
    )

    if not incidents:

        st.warning(
            "No Module 5 incident outputs were found."
        )

    else:

        for incident in incidents:

            source_image = incident.get(
                "source_image",
                "Unknown"
            )

            risk = incident.get(
                "risk_report",
                {}
            )

            with st.expander(
                f"⚠️ {source_image} — Risk Assessment",
                expanded=False
            ):

                priority = risk.get(
                    "response_priority",
                    "UNKNOWN"
                )

                if priority == "HIGH":

                    st.error(
                        f"Response Priority: {priority}"
                    )

                elif priority == "MEDIUM":

                    st.warning(
                        f"Response Priority: {priority}"
                    )

                else:

                    st.success(
                        f"Response Priority: {priority}"
                    )

                st.divider()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Spill Area",
                    (
                        f"{risk.get('spill_area_km2', 0)} "
                        "km²"
                    )
                )

                col2.metric(
                    "Detection Confidence",
                    risk.get(
                        "detection_confidence",
                        "N/A"
                    )
                )

                col3.metric(
                    "Estimated Tonnes",
                    risk.get(
                        "estimated_tonnes",
                        "N/A"
                    )
                )

                st.divider()

                tier_col1, tier_col2 = st.columns(2)

                with tier_col1:

                    st.subheader(
                        "NOSDCP Classification"
                    )

                    st.write(
                        f"**Tier:** "
                        f"{risk.get('nosdcp_tier', 'N/A')}"
                    )

                    st.write(
                        risk.get(
                            "nosdcp_tier_description",
                            "No description available."
                        )
                    )

                with tier_col2:

                    st.subheader(
                        "Coastal Impact"
                    )

                    st.write(
                        f"**Nearest Coast:** "
                        f"{risk.get('nearest_real_coast_km', 'N/A')} km"
                    )

                    impact_time = risk.get(
                        "time_to_impact_hours"
                    )

                    if impact_time is None:

                        st.write(
                            "**Estimated Time to Impact:** "
                            "Not currently applicable"
                        )

                    else:

                        st.write(
                            f"**Estimated Time to Impact:** "
                            f"{impact_time} hours"
                        )

                st.divider()

                st.subheader(
                    "Recommended Response"
                )

                st.info(
                    risk.get(
                        "recommended_action",
                        "No recommendation available."
                    )
                )

                st.caption(
                    "Risk report generated at: "
                    + str(
                        risk.get(
                            "generated_at_utc",
                            "Unknown"
                        )
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SPILLTRACE — Oil Spill Incident Intelligence Prototype"
)