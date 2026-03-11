"""
Wildlife Tracker - Streamlit Frontend
Interactive UI for querying the wildlife identification API.
Supports both text descriptions and photo uploads (multimodal).

Usage:
  streamlit run src/frontend/app.py
"""

import time

import requests
import streamlit as st

# ─── Wikipedia Image Fetcher ─────────────────────────────────────


WIKI_HEADERS = {
    "User-Agent": "WildlifeTrackerBot/1.0 (educational project; Python/requests)",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_species_image(species_name: str, scientific_name: str = "") -> str | None:
    """
    Fetch species image URL from Wikipedia.

    Tries the common name first, then the scientific name as fallback.
    Results are cached for 1 hour to avoid repeated API calls.

    Returns:
        Image URL string, or None if no image found
    """
    names_to_try = [species_name]
    if scientific_name and scientific_name != species_name:
        names_to_try.append(scientific_name)

    for name in names_to_try:
        try:
            # Wikipedia API: get page image (thumbnail up to 500px)
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + name.replace(" ", "_"),
                headers=WIKI_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Try original image first, then thumbnail
                if "originalimage" in data:
                    return data["originalimage"]["source"]
                elif "thumbnail" in data:
                    return data["thumbnail"]["source"]
        except Exception:
            continue

    return None


# ─── Configuration ───────────────────────────────────────────────

API_BASE = "http://localhost:8000"

INDIAN_STATES = [
    "",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Ladakh",
    "Jammu and Kashmir",
]

WILDLIFE_PARKS = [
    "",
    "Ranthambore National Park",
    "Jim Corbett National Park",
    "Kaziranga National Park",
    "Bandhavgarh National Park",
    "Kanha National Park",
    "Periyar National Park",
    "Sundarbans National Park",
    "Gir National Park",
    "Bharatpur Bird Sanctuary",
    "Bandipur National Park",
    "Nagarhole National Park",
    "Mudumalai National Park",
    "Silent Valley National Park",
    "Valley of Flowers",
]

SEASONS = ["", "summer", "monsoon", "post-monsoon", "winter"]

EXAMPLE_QUERIES = [
    "I saw a large orange and black striped cat in the forest",
    "Small bright blue bird with a long tail sitting near a stream",
    "Large grey animal with one horn near a river",
    "Colorful bird with a fan-shaped crest that dances in the rain",
    "Spotted deer grazing in a meadow at dawn",
    "Snake with a distinctive hood and spectacle markings",
    "Large brown bird of prey circling above open fields",
    "Small monkey with a silver-grey body and black face",
]


# ─── Page Config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Wildlife Tracker",
    page_icon="🐾",  # noqa
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("Wildlife Tracker")
    st.caption("AI-powered Indian wildlife identification")

    st.divider()

    # API status
    st.subheader("System Status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success("API: Healthy")
        elif status == "degraded":
            st.warning("API: Degraded")
        else:
            st.error(f"API: {status}")

        for name, info in health.get("components", {}).items():
            comp_status = info.get("status", "unknown")
            msg = info.get("message", "")
            st.caption(f"  {name}: {msg}")

    except requests.ConnectionError:
        st.error("API not reachable. Start with: `make api`")
    except Exception as e:
        st.error(f"Health check error: {e}")

    st.divider()

    # Metrics
    st.subheader("Quick Metrics")
    try:
        metrics = requests.get(f"{API_BASE}/metrics", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Total Queries", metrics.get("total_requests", 0))
        col2.metric("Feedback", metrics.get("feedback_count", 0))

        avg_lat = metrics.get("avg_latency_seconds", 0)
        avg_conf = metrics.get("avg_confidence", 0)
        col3, col4 = st.columns(2)
        col3.metric("Avg Latency", f"{avg_lat:.2f}s")
        col4.metric("Avg Confidence", f"{avg_conf:.0%}")

        accuracy = metrics.get("accuracy_from_feedback")
        if accuracy is not None:
            st.metric("Accuracy (feedback)", f"{accuracy:.0%}")

    except Exception:
        st.caption("Metrics unavailable")

    st.divider()
    st.caption("Built with FastAPI + Streamlit + RAG")


# ─── Main Content ────────────────────────────────────────────────

st.header("Identify Wildlife Species")
st.write("Describe what you saw **or upload a photo** for AI-powered identification.")

# ── Input Tabs: Text vs Photo ────────────────────────────────
tab_text, tab_photo = st.tabs(["Describe", "Upload Photo"])

# ── Shared location/season (outside tabs so both can use it) ──
col_loc, col_park, col_season = st.columns(3)
with col_loc:
    location_state = st.selectbox("State / Region:", INDIAN_STATES)
with col_park:
    location_park = st.selectbox("Wildlife Park (optional):", WILDLIFE_PARKS)
with col_season:
    season = st.selectbox("Season:", SEASONS)

# Determine location
location = location_park if location_park else location_state
if not location:
    location = None
season_val = season if season else None


# ── Tab 1: Text Description ──────────────────────────────────

with tab_text:
    with st.form("identify_form"):
        query = st.text_area(
            "Describe the animal you saw:",
            height=100,
            placeholder="e.g., I saw a large orange and black striped cat in the forest near a river...",
        )
        text_submitted = st.form_submit_button(
            "Identify Species",
            use_container_width=True,
            type="primary",
        )

    # Example queries
    st.caption("Try an example:")
    example_cols = st.columns(4)
    for i, example in enumerate(EXAMPLE_QUERIES[:4]):
        with example_cols[i]:
            if st.button(example[:40] + "...", key=f"ex_{i}", use_container_width=True):
                st.session_state["prefill_query"] = example
                st.rerun()

    # Handle prefilled query
    if "prefill_query" in st.session_state:
        query = st.session_state.pop("prefill_query")
        text_submitted = True

    # Process text query
    if text_submitted and query:
        with st.spinner("Analyzing your description..."):
            try:
                payload = {
                    "query": query,
                    "location": location,
                    "season": season_val,
                }
                start = time.time()
                response = requests.post(
                    f"{API_BASE}/identify",
                    json=payload,
                    timeout=120,
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    # Store for display below
                    st.session_state["last_result"] = response.json()
                    st.session_state["last_mode"] = "text"
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment and try again.")
                elif response.status_code == 504:
                    st.error("Request timed out. The query may be too complex.")
                else:
                    error_data = response.json()
                    st.error(f"Error: {error_data.get('detail', 'Unknown error')}")

            except requests.ConnectionError:
                st.error("Cannot connect to the API. Make sure the server is running.")
            except requests.Timeout:
                st.error("Request timed out. Try a simpler description.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    elif text_submitted and not query:
        st.warning("Please enter a description of the wildlife sighting.")


# ── Tab 2: Photo Upload ──────────────────────────────────────

with tab_photo:
    uploaded_file = st.file_uploader(
        "Upload a wildlife photo:",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPEG, PNG, and WebP. Max 20MB.",
    )

    if uploaded_file:
        # Show preview
        img_col, info_col = st.columns([2, 1])
        with img_col:
            st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
        with info_col:
            st.write(f"**File:** {uploaded_file.name}")
            size_mb = uploaded_file.size / 1024 / 1024
            st.write(f"**Size:** {size_mb:.1f} MB")
            st.write(f"**Type:** {uploaded_file.type}")

    photo_submitted = st.button(
        "Identify from Photo",
        use_container_width=True,
        type="primary",
        disabled=uploaded_file is None,
    )

    if photo_submitted and uploaded_file:
        with st.spinner("Analyzing photo with vision AI + knowledge base..."):
            try:
                # Send image to API
                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                form_data = {}
                if location:
                    form_data["location"] = location
                if season_val:
                    form_data["season"] = season_val

                start = time.time()
                response = requests.post(
                    f"{API_BASE}/identify/image",
                    files=files,
                    data=form_data,
                    timeout=120,
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    st.session_state["last_result"] = response.json()
                    st.session_state["last_mode"] = "image"
                elif response.status_code == 400:
                    error_data = response.json()
                    st.error(f"Invalid image: {error_data.get('detail', 'Unknown error')}")
                elif response.status_code == 504:
                    st.error("Request timed out. Try a smaller or clearer photo.")
                else:
                    error_data = response.json()
                    st.error(f"Error: {error_data.get('detail', 'Unknown error')}")

            except requests.ConnectionError:
                st.error("Cannot connect to the API. Make sure the server is running.")
            except requests.Timeout:
                st.error("Request timed out. Try a smaller photo.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.caption(
        "The photo is analyzed by a vision AI model, then identified using the RAG knowledge base."
    )


# ─── Display Results ─────────────────────────────────────────

if "last_result" in st.session_state:
    data = st.session_state["last_result"]
    mode = st.session_state.get("last_mode", "text")

    st.divider()

    # Vision description (for image mode)
    if mode == "image" and data.get("vision_description"):
        with st.expander("Vision Model Description", expanded=True):
            st.info(data["vision_description"])
            vision_time = data.get("vision_latency_seconds", 0)
            st.caption(f"Vision model: {data.get('vision_model', 'N/A')} | {vision_time:.1f}s")

    # Main result
    species = data.get("species_name", "Unknown")
    confidence = data.get("confidence", 0)
    reasoning = data.get("reasoning", "")

    if species == "DECLINED":
        st.warning("Could not identify the species from this input.")
        st.write(f"**Reason:** {reasoning}")
    else:
        # Fetch species image from Wikipedia
        scientific = data.get("scientific_name", "")
        image_url = fetch_species_image(species, scientific)

        # Layout: Image on left, info on right
        if image_url:
            img_col, info_col = st.columns([1, 2])
            with img_col:
                st.image(image_url, caption=f"{species}", use_container_width=True)
            with info_col:
                st.subheader(f"{species}")
                if scientific:
                    st.caption(f"*{scientific}*")
                if confidence >= 0.8:
                    st.success(f"Confidence: {confidence:.0%}")
                elif confidence >= 0.5:
                    st.warning(f"Confidence: {confidence:.0%}")
                else:
                    st.error(f"Confidence: {confidence:.0%}")
                st.write(f"**Reasoning:** {reasoning}")
        else:
            # No image found — original layout
            result_col1, result_col2 = st.columns([3, 1])
            with result_col1:
                st.subheader(f"{species}")
                if scientific:
                    st.caption(f"*{scientific}*")
            with result_col2:
                if confidence >= 0.8:
                    st.success(f"Confidence: {confidence:.0%}")
                elif confidence >= 0.5:
                    st.warning(f"Confidence: {confidence:.0%}")
                else:
                    st.error(f"Confidence: {confidence:.0%}")
            st.write(f"**Reasoning:** {reasoning}")

        # Details
        with st.expander("Identification Details", expanded=False):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                features = data.get("key_features_matched", [])
                if features:
                    st.write("**Key Features Matched:**")
                    for feat in features:
                        st.write(f"- {feat}")
                habitat = data.get("habitat_match", "")
                if habitat:
                    st.write(f"**Habitat Match:** {habitat}")

            with detail_col2:
                conservation = data.get("conservation_status", "")
                if conservation:
                    st.write(f"**Conservation Status:** {conservation}")
                geo_match = data.get("geographic_match", True)
                st.write(f"**Geographic Match:** {'Yes' if geo_match else 'No'}")
                cited = data.get("cited_sources", [])
                if cited:
                    st.write("**Sources:**")
                    for src in cited:
                        st.write(f"- `{src}`")

        # Performance details
        with st.expander("Performance & Pipeline Details", expanded=False):
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            perf_col1.metric("Total Latency", f"{data.get('total_latency_seconds', 0):.2f}s")
            perf_col2.metric("Chunks Retrieved", data.get("chunks_retrieved", 0))
            perf_col3.metric("Chunks Used", data.get("chunks_used", 0))

            input_mode = data.get("input_mode", "text")
            st.write(f"**Input Mode:** {input_mode}")
            st.write(f"**Inference Mode:** {data.get('inference_mode', 'N/A')}")
            st.write(f"**Request ID:** `{data.get('request_id', 'N/A')}`")

            if input_mode == "image":
                vision_time = data.get("vision_latency_seconds", 0)
                rag_time = data.get("total_latency_seconds", 0) - vision_time
                st.write(f"**Vision Latency:** {vision_time:.2f}s")
                st.write(f"**RAG Latency:** {rag_time:.2f}s")

        # Feedback section
        st.divider()
        st.subheader("Was this identification correct?")
        feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 2])

        request_id = data.get("request_id", "")

        with feedback_col1:
            if st.button("Yes, correct!", key="fb_yes", use_container_width=True):
                try:
                    fb = requests.post(
                        f"{API_BASE}/feedback",
                        json={"request_id": request_id, "was_correct": True},
                        timeout=5,
                    )
                    if fb.status_code == 200:
                        st.success("Thanks for confirming!")
                except Exception:
                    st.error("Could not submit feedback")

        with feedback_col2:
            if st.button("No, incorrect", key="fb_no", use_container_width=True):
                st.session_state["show_correction"] = True

        with feedback_col3:
            if st.session_state.get("show_correction"):
                correct_species = st.text_input("What was the actual species?")
                notes = st.text_input("Additional notes (optional)")
                if st.button("Submit Correction"):
                    try:
                        fb = requests.post(
                            f"{API_BASE}/feedback",
                            json={
                                "request_id": request_id,
                                "was_correct": False,
                                "correct_species": correct_species,
                                "notes": notes,
                            },
                            timeout=5,
                        )
                        if fb.status_code == 200:
                            st.success("Correction recorded. Thank you!")
                            st.session_state["show_correction"] = False
                    except Exception:
                        st.error("Could not submit feedback")

    # Clear result button
    if st.button("Clear Result", key="clear_result"):
        del st.session_state["last_result"]
        if "last_mode" in st.session_state:
            del st.session_state["last_mode"]
        st.rerun()


# ─── API Info ────────────────────────────────────────────────

with st.expander("API Configuration", expanded=False):
    st.write(f"**API Base URL:** `{API_BASE}`")
    st.write("**Endpoints:**")
    st.write("- `POST /identify` - Identify species from text")
    st.write("- `POST /identify/image` - Identify species from photo")
    st.write("- `GET /health` - Health check")
    st.write("- `GET /metrics` - System metrics")
    st.write("- `POST /feedback` - Submit feedback")
    st.write(f"- API Docs: [{API_BASE}/docs]({API_BASE}/docs)")
