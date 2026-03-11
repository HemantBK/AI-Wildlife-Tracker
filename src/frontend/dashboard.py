"""
Monitoring Dashboard - Streamlit
Real-time system health, metrics, alerts, and request history.

Usage:
  streamlit run src/frontend/dashboard.py --server.port 8502
"""

import time

import requests
import streamlit as st

# ─── Configuration ───────────────────────────────────────────────

API_BASE = "http://localhost:8000"


# ─── Page Config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Wildlife Tracker - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Helper Functions ────────────────────────────────────────────


def fetch_health():
    """Fetch health data from API."""
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception:
        return None


def fetch_metrics():
    """Fetch metrics data from API."""
    try:
        return requests.get(f"{API_BASE}/metrics", timeout=5).json()
    except Exception:
        return None


# ─── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("Dashboard")
    st.caption("Wildlife Tracker Monitoring")
    st.divider()

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.caption("Page will refresh every 30 seconds")
        time.sleep(0.1)  # Small delay to let streamlit process
        # Use st.rerun with a counter to auto-refresh
        if "refresh_counter" not in st.session_state:
            st.session_state["refresh_counter"] = 0

    st.divider()
    st.caption(f"API: {API_BASE}")
    if st.button("Refresh Now"):
        st.rerun()


# ─── Main Content ────────────────────────────────────────────────

st.header("System Monitoring Dashboard")

# Fetch data
health = fetch_health()
metrics = fetch_metrics()

if not health and not metrics:
    st.error("Cannot connect to the API. Make sure the server is running: `make api`")
    st.stop()

# ─── Row 1: Key Metrics ─────────────────────────────────────────

st.subheader("Key Metrics")

if metrics:
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Requests",
        metrics.get("total_requests", 0),
    )
    col2.metric(
        "Successful",
        metrics.get("successful_identifications", 0),
    )
    col3.metric(
        "Declined",
        metrics.get("declined_identifications", 0),
    )
    col4.metric(
        "Errors",
        metrics.get("error_count", 0),
    )

    uptime = metrics.get("uptime_seconds", 0)
    if uptime > 3600:
        uptime_str = f"{uptime / 3600:.1f}h"
    elif uptime > 60:
        uptime_str = f"{uptime / 60:.0f}m"
    else:
        uptime_str = f"{uptime:.0f}s"
    col5.metric("Uptime", uptime_str)

    # Second row of metrics
    col6, col7, col8, col9 = st.columns(4)

    col6.metric(
        "Avg Latency",
        f"{metrics.get('avg_latency_seconds', 0):.2f}s",
    )
    col7.metric(
        "P95 Latency",
        f"{metrics.get('p95_latency_seconds', 0):.2f}s",
    )
    col8.metric(
        "Avg Confidence",
        f"{metrics.get('avg_confidence', 0):.0%}",
    )

    accuracy = metrics.get("accuracy_from_feedback")
    if accuracy is not None:
        col9.metric("Accuracy (feedback)", f"{accuracy:.0%}")
    else:
        col9.metric("Accuracy (feedback)", "N/A")

st.divider()

# ─── Row 2: Component Health ────────────────────────────────────

st.subheader("Component Health")

if health:
    overall = health.get("status", "unknown")
    if overall == "healthy":
        st.success(f"Overall Status: {overall.upper()}")
    elif overall == "degraded":
        st.warning(f"Overall Status: {overall.upper()}")
    else:
        st.error(f"Overall Status: {overall.upper()}")

    components = health.get("components", {})
    cols = st.columns(len(components) if components else 1)

    for i, (name, info) in enumerate(components.items()):
        with cols[i % len(cols)]:
            status = info.get("status", "unknown")
            message = info.get("message", "")
            latency = info.get("latency_ms")

            if status == "ok":
                st.success(f"**{name}**")
            elif status == "warning":
                st.warning(f"**{name}**")
            elif status == "error":
                st.error(f"**{name}**")
            elif status == "skipped":
                st.info(f"**{name}**")
            else:
                st.info(f"**{name}**")

            st.caption(message)
            if latency:
                st.caption(f"Latency: {latency}ms")

st.divider()

# ─── Row 3: Top Species & Charts ────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top Identified Species")
    if metrics:
        top_species = metrics.get("top_species", [])
        if top_species:
            # Create a simple bar chart data
            species_data = {s["species"]: s["count"] for s in top_species[:10]}
            st.bar_chart(species_data)
        else:
            st.caption("No species identified yet.")

with col_right:
    st.subheader("Request Volume (Last 24h)")
    if metrics:
        by_hour = metrics.get("requests_by_hour", [])
        if by_hour:
            hour_data = {h["hour"][-5:]: h["count"] for h in by_hour}
            st.bar_chart(hour_data)
        else:
            st.caption("No request data in the last 24 hours.")

st.divider()

# ─── Row 4: Feedback Summary ────────────────────────────────────

st.subheader("Feedback Summary")

if metrics:
    fb_count = metrics.get("feedback_count", 0)
    accuracy = metrics.get("accuracy_from_feedback")

    col_fb1, col_fb2, col_fb3 = st.columns(3)

    col_fb1.metric("Total Feedback", fb_count)

    if accuracy is not None:
        if accuracy >= 0.8 or accuracy >= 0.6:
            col_fb2.metric("Accuracy", f"{accuracy:.0%}")
        else:
            col_fb2.metric("Accuracy", f"{accuracy:.0%}")
    else:
        col_fb2.metric("Accuracy", "Not enough data")

    total = metrics.get("total_requests", 0)
    feedback_rate = (fb_count / total * 100) if total > 0 else 0
    col_fb3.metric("Feedback Rate", f"{feedback_rate:.1f}%")

st.divider()

# ─── Row 5: Alert Thresholds ────────────────────────────────────

st.subheader("Quality Thresholds")

threshold_data = {
    "Metric": ["P95 Latency", "Error Rate", "Min Accuracy", "Min Confidence"],
    "Threshold": ["<= 15.0s", "<= 10.0%", ">= 70.0%", ">= 0.50"],
    "Current": [
        f"{metrics.get('p95_latency_seconds', 0):.2f}s" if metrics else "N/A",
        f"{(metrics.get('error_count', 0) / max(metrics.get('total_requests', 1), 1) * 100):.1f}%"
        if metrics
        else "N/A",
        f"{metrics.get('accuracy_from_feedback', 0):.0%}"
        if metrics and metrics.get("accuracy_from_feedback") is not None
        else "N/A",
        f"{metrics.get('avg_confidence', 0):.2f}" if metrics else "N/A",
    ],
}

# Determine status
if metrics:
    total = max(metrics.get("total_requests", 1), 1)
    statuses = []
    statuses.append("PASS" if metrics.get("p95_latency_seconds", 0) <= 15.0 else "FAIL")
    statuses.append("PASS" if (metrics.get("error_count", 0) / total * 100) <= 10.0 else "FAIL")
    acc = metrics.get("accuracy_from_feedback")
    statuses.append(
        "PASS" if acc is not None and acc >= 0.70 else ("N/A" if acc is None else "FAIL")
    )
    statuses.append("PASS" if metrics.get("avg_confidence", 0) >= 0.50 else "FAIL")
    threshold_data["Status"] = statuses
else:
    threshold_data["Status"] = ["N/A"] * 4

st.table(threshold_data)

# ─── Row 6: Raw Data (Expandable) ───────────────────────────────

with st.expander("Raw API Responses", expanded=False):
    tab1, tab2 = st.tabs(["Health JSON", "Metrics JSON"])
    with tab1:
        if health:
            st.json(health)
        else:
            st.write("Health data unavailable")
    with tab2:
        if metrics:
            st.json(metrics)
        else:
            st.write("Metrics data unavailable")


# ─── Auto-refresh ───────────────────────────────────────────────

if auto_refresh:
    time.sleep(30)
    st.rerun()
