# ==================================================
# FILE: forecastiq/features/voice/data_commit.py
# ROLE: VERBAL DATASET COMMIT CHECKPOINT
# STATUS: ADDITIVE / NON-BLOCKING / EXECUTIVE-GRADE
# ==================================================

import streamlit as st
from pathlib import Path


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

AUDIO_PATH = Path("forecastiq/assets/audio/data_commit.mp3")

SESSION_KEYS = {
    "played": "data_commit_voice_played",
    "muted": "data_commit_voice_muted",
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
        if st.button("▶ Play", key="data_commit_play"):
            st.session_state[SESSION_KEYS["played"]] = True

    with cols[1]:
        st.caption(
            "Guidance on what happens after data is committed and how responsibility applies."
        )

    with cols[2]:
        muted = st.checkbox(
            "Mute",
            value=st.session_state[SESSION_KEYS["muted"]],
            key="data_commit_mute",
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

def render_data_commit_checkpoint():
    """
    Renders a verbal checkpoint immediately after dataset commit.

    - Appears after data is committed
    - Manual playback only
    - Replayable
    - Respects mute
    - Does NOT interfere with data entry
    """

    _init_session_state()

    container = st.container(border=True)

    with container:
        st.caption("Dataset Commit")

        st.markdown("### 🎙️ Data Commit Guidance")

        st.markdown(
            """
Your data has been committed.

Sentinel will now evaluate patterns, stability, and uncertainty based on the data provided.

Forecast quality depends on data completeness, consistency, and relevance.

Sentinel will highlight signals and risk — interpretation and decisions remain yours.
"""
        )

        _render_audio_controls()
