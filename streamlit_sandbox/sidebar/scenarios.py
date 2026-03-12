import streamlit as st

LEVELS = ["Low", "Medium", "High"]

def render_scenarios(disabled: bool):
    st.subheader("Economic Scenarios")

    # Recession profiles
    st.session_state.recession_profile = st.selectbox(
        "Recession Profile",
        options=["None", "Light", "Moderate", "Severe"],
        index=["None", "Light", "Moderate", "Severe"].index(
            st.session_state.recession_profile
        ),
        disabled=disabled,
        help="Preset macro stress assumptions."
    )

    st.markdown("**Macro Variables**")

    for k, v in st.session_state.macro_factors.items():
        st.session_state.macro_factors[k] = st.selectbox(
            k.replace("_", " ").title(),
            options=LEVELS,
            index=LEVELS.index(v),
            disabled=disabled
        )

