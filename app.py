import streamlit as st

st.set_page_config(page_title="SPILLTRACE", layout="wide")

st.title("🛰️ SPILLTRACE")
st.subheader("Oil Spill Incident Intelligence System — Prototype")

st.markdown(
    """
    Welcome to the first working version of your **SPILLTRACE** dashboard.

    This is just a starting point. As you build each module — detection,
    drift intelligence, vessel investigation, risk analysis, alerts — you'll
    plug its output into pages like this one.
    """
)

st.sidebar.header("Modules")
st.sidebar.info("Modules will appear here as you build them, one at a time.")

col1, col2, col3 = st.columns(3)
col1.metric("Spills Detected", "0")
col2.metric("Active Investigations", "0")
col3.metric("Alerts Sent", "0")

st.divider()
st.success("✅ If you can see this page, your setup works. Time to build Module 1: Oil Spill Detection.")
