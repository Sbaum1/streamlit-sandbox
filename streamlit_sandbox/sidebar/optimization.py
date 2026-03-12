import streamlit as st

def render_optimization(disabled: bool):
    st.subheader("Optimization")

    st.session_state.optimization_mode = st.radio(
        "Mode",
        options=["Auto", "Manual"],
        index=0 if st.session_state.optimization_mode == "Auto" else 1,
        disabled=disabled,
        help=(
            "Auto uses industry best practices. "
            "Manual allows advanced overrides."
        )
    )

    if st.session_state.optimization_mode == "Manual":
        st.caption("Advanced settings will appear here in a future step.")

