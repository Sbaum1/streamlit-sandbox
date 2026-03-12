import streamlit as st

def render_controls(disabled: bool):
    st.subheader("Forecast Controls")

    st.session_state.forecast_horizon = st.slider(
        "Forecast Horizon (periods)",
        min_value=3, max_value=60,
        value=st.session_state.forecast_horizon,
        disabled=disabled,
        help="How far into the future to project."
    )

    st.session_state.backtest_window = st.slider(
        "Backtest Window (periods)",
        min_value=6, max_value=120,
        value=st.session_state.backtest_window,
        disabled=disabled,
        help="History used to evaluate forecast accuracy."
    )

    st.session_state.confidence_level = st.radio(
        "Confidence Interval",
        options=[0.8, 0.9, 0.95],
        index=[0.8, 0.9, 0.95].index(st.session_state.confidence_level),
        disabled=disabled,
        help="Uncertainty band width."
    )

