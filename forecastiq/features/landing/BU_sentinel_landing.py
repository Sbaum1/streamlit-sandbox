# ==================================================
# FILE: forecastiq/features/landing/sentinel_landing.py
# ROLE: SENTINEL ORIENTATION & GOVERNANCE LANDING PAGE
# STATUS: ADDITIVE / NON-BLOCKING / EXECUTIVE-GRADE
# ==================================================

import streamlit as st
from pathlib import Path


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

AUDIO_PATH = Path("forecastiq/assets/audio/sentinel_briefing.mp3")

SESSION_KEYS = {
    "seen": "sentinel_landing_seen",
    "voice_played": "sentinel_voice_played",
    "voice_muted": "sentinel_voice_muted",
}


# --------------------------------------------------
# INTERNAL HELPERS (PURE)
# --------------------------------------------------

def _init_session_state():
    for key in SESSION_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = False


def _render_voice_briefing():
    st.markdown("#### 🎙️ Sentinel Orientation Briefing")

    cols = st.columns([1, 3, 1])

    with cols[0]:
        if st.button("▶ Play"):
            st.session_state[SESSION_KEYS["voice_played"]] = True

    with cols[1]:
        st.caption(
            "A brief executive orientation explaining what Sentinel is — and is not."
        )

    with cols[2]:
        mute = st.checkbox(
            "Mute",
            value=st.session_state[SESSION_KEYS["voice_muted"]],
            help="Disables audio playback",
        )
        st.session_state[SESSION_KEYS["voice_muted"]] = mute

    if (
        st.session_state[SESSION_KEYS["voice_played"]]
        and not st.session_state[SESSION_KEYS["voice_muted"]]
        and AUDIO_PATH.exists()
    ):
        st.audio(str(AUDIO_PATH))


# --------------------------------------------------
# PUBLIC RENDER FUNCTION
# --------------------------------------------------

def render_sentinel_landing():
    """
    Renders the Sentinel landing / orientation surface.

    - Visual, explanatory, governance-focused
    - Non-blocking
    - First-run visual emphasis only
    """

    _init_session_state()

    first_run = not st.session_state[SESSION_KEYS["seen"]]

    # Subtle first-run emphasis (visual only)
    container = st.container(border=first_run)

    with container:
        if first_run:
            st.caption("Orientation")

        st.markdown("## 🛡️ Sentinel Decision Intelligence")

        st.markdown(
            """
**Sentinel** is a continuous decision-intelligence monitoring system designed to
support — **not replace** — executive judgment.

It exists to surface risk, uncertainty, volatility, and signal deviation
*before* decisions are made.
"""
        )

        # --------------------------------------------------
        # WHAT SENTINEL IS / IS NOT
        # --------------------------------------------------

        col_is, col_is_not = st.columns(2)

        with col_is:
            st.markdown("### ✅ What Sentinel *Is*")
            st.markdown(
                """
- A continuous monitoring and signal-detection layer  
- A risk-first intelligence system  
- A confidence and uncertainty interpreter  
- A governance-aware analytical companion  
- A system designed to **inform**, not persuade  
"""
            )

        with col_is_not:
            st.markdown("### ❌ What Sentinel *Is Not*")
            st.markdown(
                """
- A crystal ball  
- An autonomous decision-maker  
- A guarantee engine  
- A black-box AI  
- A replacement for executive accountability  
"""
            )

        st.markdown(
            """
> Sentinel does not decide.  
> Sentinel does not promise.  
> **Sentinel protects judgment — it does not replace it.**
"""
        )

        # --------------------------------------------------
        # VOICE BRIEFING
        # --------------------------------------------------

        _render_voice_briefing()

        # --------------------------------------------------
        # CLOSE
        # --------------------------------------------------

        st.markdown(
            """
When you are ready, proceed to the Sentinel interface below.
"""
        )

    # Mark landing as seen (soft flag, no gating)
    st.session_state[SESSION_KEYS["seen"]] = True
