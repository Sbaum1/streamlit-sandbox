# ==================================================
# FILE: forecastiq/features/voice/forecast_interpretation.py
# ROLE: VERBAL FORECAST INTERPRETATION CHECKPOINT
# STATUS: ADDITIVE / NON-BLOCKING / EXECUTIVE-GRADE
# ==================================================

import streamlit as st
from pathlib import Path


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

AUDIO_PATH = Path("forecastiq/assets/audio/forecast_interpretation.mp3")

SESSION_KEYS = {
    "played": "forecast_interpretation_voice_played",
    "muted": "forecast_interpretation_voice_muted",
    "shown": "forecast_interpretation_shown",
}


# --------------------------------------------------
# INTERNAL HELPERS (PURE)
# --------------------------------------------------

def _init_session_state():
    for key in SESSION_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = False


def _render_audio_controls():
    cols = st.columns([1, 4, 1])

    with cols[0]:
        if st.button("▶ Play", key="forecast_interp_play"):
            st.session_state[SESSION_KEYS["played"]] = True

    with cols[1]:
        st.caption(
            "Guidance on how to interpret forecast results, confidence, and risk."
        )

    with cols[2]:
        muted = st.checkbox(
            "Mute",
            value=st.session_state[SESSION_KEYS["muted"]],
            key="forecast_interp_mute",
        )
        st.session_state[SESSION_KEYS["muted"]] = muted

    if (
        st.session_state[SESSION_KEYS["played"]]
        and not st.session_state[SESSION_KEYS["muted"]]
        and AUDIO_PATH.exists()
    ):
        st.audio(str(AUDIO_PATH))


# --------------------------------------------------
# PUBLIC RENDER FUNCTION
# --------------------------------------------------

def render_forecast_interpretation_checkpoint():
    """
    Renders a verbal guidance checkpoint for interpreting forecast outputs.

    - Appears once when forecast results are first available
    - Manual playback only
    - Replayable
    - Muted globally
    - Does NOT interfere with forecast rendering
    """

    _init_session_state()

    # Only render if forecast exists
    if st.session_state.get("latest_forecast_df") is None:
        return

    first_show = not st.session_state[SESSION_KEYS["shown"]]

    container = st.container(border=first_show)

    with container:
        if first_show:
            st.caption("Interpretation Guidance")

        st.markdown("### 🎙️ Forecast Interpretation")

        st.markdown(
            """
This forecast represents a **range of possible outcomes**, not a single prediction.

Confidence intervals reflect **uncertainty**, not error.

Focus on **direction, variability, and risk** — not point precision.

Sentinel highlights what deserves attention, not what must be believed.
"""
        )

        _render_audio_controls()

    # Mark as shown (soft, no gating)
    st.session_state[SESSION_KEYS["shown"]] = True
