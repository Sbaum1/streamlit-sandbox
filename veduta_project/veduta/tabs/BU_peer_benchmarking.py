# ==================================================
# FILE: veduta/tabs/peer_benchmarking.py
# VERSION: 1.0.0
# ROLE: PEER BENCHMARKING TAB
# ENGINE: Self-contained ETS + ARIMA + Ensemble
# ISOLATION: All session keys prefixed pb_
#            Zero collision with home tab state
# ==================================================

import calendar
import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Try ARIMA (statsmodels optional) ─────────────────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _ARIMA_OK = True
except Exception:
    _ARIMA_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _ETS_OK = True
except Exception:
    _ETS_OK = False


# ── VEDUTA design tokens ──────────────────────────────────────────────────────
_NAVY   = "#0E1A2E"
_NAVY2  = "#162440"
_GOLD   = "#C9943A"
_GOLD2  = "#E8B860"
_CREAM  = "#F5F0E8"
_BLUE   = "#2E5B9A"
_GREEN  = "#1A6B42"
_RED    = "#B81C1C"
_SILVER = "#8FA3B8"
_BG     = "#07080F"
_CARD   = "#0D1420"
_BORDER = "#243347"

MONTH_NAMES = [calendar.month_abbr[m] for m in range(1, 13)]

CATEGORIES = [
    "Regions",
    "Sales Reps",
    "SKUs",
    "Segments",
    "Customers",
    "Categories",
    "Attributes",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
/* ── Peer Benchmarking tab — VEDUTA Identity ── */
.pb-section {{
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: {_GOLD};
    margin: 2rem 0 .85rem 0;
    display: flex;
    align-items: center;
    gap: .6rem;
}}
.pb-section::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, {_BORDER} 0%, transparent 100%);
}}
.pb-kpi-card {{
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
}}
.pb-kpi-val {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 300;
    color: {_CREAM};
}}
.pb-kpi-lbl {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: {_SILVER};
    margin-top: 4px;
}}
.pb-rank-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
}}
.pb-rank-1 {{ background: {_GOLD}22; color: {_GOLD2}; border: 1px solid {_GOLD}55; }}
.pb-rank-top {{ background: {_GREEN}22; color: #5EC98A; border: 1px solid {_GREEN}55; }}
.pb-rank-bot {{ background: {_RED}22; color: #E06C6C; border: 1px solid {_RED}55; }}
.pb-rank-mid {{ background: {_NAVY2}; color: {_SILVER}; border: 1px solid {_BORDER}; }}
.pb-note {{
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: {_SILVER};
    margin-top: 6px;
}}
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_month_labels(start_month_abbr: str, n: int = 36) -> list[str]:
    """Return n consecutive 3-letter month abbreviations starting from start_month_abbr."""
    try:
        start_idx = MONTH_NAMES.index(start_month_abbr)
    except ValueError:
        start_idx = 0
    return [MONTH_NAMES[(start_idx + i) % 12] for i in range(n)]


def _build_year_labels(start_month_abbr: str, n: int = 36) -> list[str]:
    """Return year suffixes (Yr1 / Yr2 / Yr3) for each of n months."""
    try:
        start_idx = MONTH_NAMES.index(start_month_abbr)
    except ValueError:
        start_idx = 0
    labels = []
    for i in range(n):
        yr = (start_idx + i) // 12 + 1
        labels.append(f"Yr{yr}")
    return labels


def _mase(actual: np.ndarray, forecast: np.ndarray, naive: np.ndarray) -> float:
    """MASE vs seasonal-naïve baseline."""
    mae_fc = np.nanmean(np.abs(actual - forecast))
    mae_naive = np.nanmean(np.abs(actual - naive))
    if mae_naive == 0 or np.isnan(mae_naive):
        return np.nan
    return float(mae_fc / mae_naive)


def _seasonal_naive(ts: np.ndarray, season: int = 12) -> np.ndarray:
    """One-step-ahead seasonal naïve forecasts (same month last year)."""
    naive = np.full_like(ts, np.nan, dtype=float)
    for i in range(season, len(ts)):
        naive[i] = ts[i - season]
    return naive


# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING ENGINE (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _ets_forecast(ts: np.ndarray, horizon: int = 12, backtest_h: int = 12):
    """ETS with additive trend + additive seasonal. Returns (fc, lo, hi, mase)."""
    if not _ETS_OK or len(ts) < 24:
        return None

    train = ts[:-backtest_h]
    test  = ts[-backtest_h:]

    try:
        m = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        bt = m.forecast(backtest_h)
        naive = _seasonal_naive(ts)[-backtest_h:]
        mase_val = _mase(test, bt, naive)
    except Exception:
        mase_val = np.nan

    try:
        m_full = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        fc = m_full.forecast(horizon)
        resid_std = float(np.nanstd(m_full.resid))
        lo = fc - 1.96 * resid_std
        hi = fc + 1.96 * resid_std
        return {"fc": fc, "lo": lo, "hi": hi, "mase": mase_val, "name": "ETS"}
    except Exception:
        return None


def _arima_forecast(ts: np.ndarray, horizon: int = 12, backtest_h: int = 12):
    """SARIMA(1,1,1)(1,0,1,12). Returns (fc, lo, hi, mase)."""
    if not _ARIMA_OK or len(ts) < 24:
        return None

    train = ts[:-backtest_h]
    test  = ts[-backtest_h:]

    try:
        m_bt = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        bt = m_bt.forecast(backtest_h)
        naive = _seasonal_naive(ts)[-backtest_h:]
        mase_val = _mase(test, bt, naive)
    except Exception:
        mase_val = np.nan

    try:
        m_full = SARIMAX(
            ts,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fc_res = m_full.get_forecast(horizon)
        fc = fc_res.predicted_mean
        ci = fc_res.conf_int()
        lo = ci.iloc[:, 0].values
        hi = ci.iloc[:, 1].values
        return {"fc": fc, "lo": lo, "hi": hi, "mase": mase_val, "name": "ARIMA"}
    except Exception:
        return None


def _ensemble_forecast(ets_res, arima_res):
    """Inverse-MASE weighted ensemble of ETS + ARIMA."""
    results = [r for r in [ets_res, arima_res] if r is not None]
    if not results:
        return None
    if len(results) == 1:
        r = results[0].copy()
        r["name"] = f"{r['name']} only"
        return r

    # Weights
    mases = [r["mase"] if not np.isnan(r.get("mase", np.nan)) else 1.0 for r in results]
    inv   = [1.0 / m if m > 0 else 1.0 for m in mases]
    total = sum(inv)
    w     = [i / total for i in inv]

    fc = sum(w[i] * results[i]["fc"] for i in range(len(results)))
    lo = sum(w[i] * results[i]["lo"] for i in range(len(results)))
    hi = sum(w[i] * results[i]["hi"] for i in range(len(results)))

    # Ensemble MASE = weighted average of component MASEs
    ens_mase = sum(w[i] * (results[i]["mase"] if not np.isnan(results[i].get("mase", np.nan)) else 1.0)
                   for i in range(len(results)))

    return {
        "fc": np.array(fc),
        "lo": np.array(lo),
        "hi": np.array(hi),
        "mase": float(ens_mase),
        "name": f"Ensemble (ETS {w[0]:.2f} / ARIMA {w[1]:.2f})",
        "weights": w,
    }


def run_entity_forecast(values: list, horizon: int = 12, backtest_h: int = 12) -> dict:
    """
    Run ETS + ARIMA + Ensemble on a list of up to 36 numeric values.
    Returns dict with all model results + primary (ensemble) output.
    """
    ts = np.array([float(v) if v is not None and not np.isnan(float(v)) else np.nan
                   for v in values], dtype=float)

    # Forward-fill NaNs (interpolate gaps)
    mask = np.isnan(ts)
    if mask.all():
        return {"error": "All values are missing."}
    idx = np.where(~mask)[0]
    ts_filled = np.interp(np.arange(len(ts)), idx, ts[idx])

    ets   = _ets_forecast(ts_filled, horizon, backtest_h)
    arima = _arima_forecast(ts_filled, horizon, backtest_h)
    ens   = _ensemble_forecast(ets, arima)

    primary = ens or ets or arima
    if primary is None:
        return {"error": "Insufficient data for forecasting (need ≥ 24 months)."}

    return {
        "ts": ts_filled,
        "ets": ets,
        "arima": arima,
        "ensemble": ens,
        "primary": primary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pb_init():
    defaults = {
        "pb_setup_committed":  False,
        "pb_category":         "Sales Reps",
        "pb_entity_count":     3,
        "pb_entity_names":     [],
        "pb_start_month":      "Jan",
        "pb_wide":             None,          # DataFrame: 36 rows × entity cols
        "pb_data_committed":   False,
        "pb_results":          None,          # dict: entity → forecast result
        "pb_run_done":         False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _plotly_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Cormorant Garamond, serif", size=18, color=_CREAM)),
        paper_bgcolor=_CARD,
        plot_bgcolor=_BG,
        font=dict(family="DM Mono, monospace", size=11, color=_SILVER),
        xaxis=dict(gridcolor=_BORDER, linecolor=_BORDER, tickcolor=_BORDER),
        yaxis=dict(gridcolor=_BORDER, linecolor=_BORDER, tickcolor=_BORDER),
        legend=dict(bgcolor=_CARD, bordercolor=_BORDER, borderwidth=1,
                    font=dict(size=10, color=_SILVER)),
        margin=dict(l=50, r=30, t=60, b=50),
        hovermode="x unified",
    )


def _forecast_chart(entity: str, ts: np.ndarray, result: dict,
                    month_labels: list[str], horizon: int = 12) -> go.Figure:
    """Individual entity forecast chart with CI band."""
    hist_labels  = month_labels[:len(ts)]
    fc_labels    = [f"F{i+1}" for i in range(horizon)]
    primary      = result["primary"]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_labels, y=ts,
        mode="lines+markers",
        name="Historical",
        line=dict(color=_GOLD, width=2),
        marker=dict(size=4, color=_GOLD),
    ))

    # ETS (if available, shown as alternative)
    if result.get("ets"):
        fig.add_trace(go.Scatter(
            x=fc_labels, y=result["ets"]["fc"],
            mode="lines",
            name="ETS",
            line=dict(color=_SILVER, width=1, dash="dot"),
            visible="legendonly",
        ))

    # ARIMA (if available)
    if result.get("arima"):
        fig.add_trace(go.Scatter(
            x=fc_labels, y=result["arima"]["fc"],
            mode="lines",
            name="ARIMA",
            line=dict(color=_BLUE, width=1, dash="dot"),
            visible="legendonly",
        ))

    # CI band
    fig.add_trace(go.Scatter(
        x=fc_labels + fc_labels[::-1],
        y=list(primary["hi"]) + list(primary["lo"][::-1]),
        fill="toself",
        fillcolor=f"{_GOLD}18",
        line=dict(width=0),
        hoverinfo="skip",
        name="95% CI",
    ))

    # Ensemble / Primary forecast
    fig.add_trace(go.Scatter(
        x=fc_labels, y=primary["fc"],
        mode="lines+markers",
        name=primary["name"],
        line=dict(color=_GOLD2, width=2.5),
        marker=dict(size=5, color=_GOLD2, symbol="diamond"),
    ))

    fig.update_layout(**_plotly_layout(entity))
    return fig


def _yoy_chart(entity: str, ts: np.ndarray, month_labels: list[str]) -> go.Figure:
    """Year-over-Year seasonal overlay — one line per year."""
    n = len(ts)
    years_data = {}
    for i, val in enumerate(ts):
        yr   = i // 12 + 1
        mo   = i % 12
        label = MONTH_NAMES[mo]
        if yr not in years_data:
            years_data[yr] = {"months": [], "values": [], "labels": []}
        years_data[yr]["months"].append(mo + 1)
        years_data[yr]["values"].append(val)
        years_data[yr]["labels"].append(label)

    colors_yr = [_GOLD, _BLUE, _GOLD2]
    fig = go.Figure()

    for yr, data in sorted(years_data.items()):
        fig.add_trace(go.Scatter(
            x=data["months"],
            y=data["values"],
            mode="lines+markers",
            name=f"Year {yr}",
            line=dict(color=colors_yr[(yr - 1) % len(colors_yr)], width=2),
            marker=dict(size=5),
            hovertemplate="%{text}: %{y:,.1f}<extra>Year " + str(yr) + "</extra>",
            text=data["labels"],
        ))

        # Trendline (hidden by default)
        x_arr = np.array(data["months"], dtype=float)
        y_arr = np.array(data["values"], dtype=float)
        if len(x_arr) >= 2 and not np.isnan(y_arr).all():
            coef = np.polyfit(x_arr, y_arr, 1)
            trend = coef[0] * x_arr + coef[1]
            fig.add_trace(go.Scatter(
                x=x_arr, y=trend,
                mode="lines",
                name=f"Year {yr} Trend",
                line=dict(color=colors_yr[(yr - 1) % len(colors_yr)], width=1.5, dash="dash"),
                visible="legendonly",
            ))

    fig.update_layout(**_plotly_layout(f"{entity} — Year-over-Year Seasonality"))
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=MONTH_NAMES,
        title="Month",
    )
    fig.update_yaxes(title="Value")
    return fig


def _leaderboard_chart(leaderboard: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart ranked by 12-month forecast total."""
    df = leaderboard.sort_values("FC Total (12mo)", ascending=True)
    colors = []
    for i, (_, row) in enumerate(df.iterrows()):
        if row["Rank"] == 1:
            colors.append(_GOLD)
        elif row["Rank"] <= max(2, len(df) // 3):
            colors.append(_GREEN)
        elif row["Rank"] >= len(df) - len(df) // 3:
            colors.append(_RED)
        else:
            colors.append(_BLUE)

    fig = go.Figure(go.Bar(
        x=df["FC Total (12mo)"],
        y=df["Entity"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,.0f}" for v in df["FC Total (12mo)"]],
        textposition="outside",
        textfont=dict(family="DM Mono, monospace", size=10, color=_CREAM),
    ))
    fig.update_layout(**_plotly_layout("12-Month Forecast Leaderboard"))
    fig.update_xaxes(title="Forecasted Total")
    fig.update_yaxes(title="")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _render_setup():
    st.markdown('<div class="pb-section">Configuration</div>', unsafe_allow_html=True)

    with st.form("pb_setup_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            category = st.selectbox(
                "Analysis Category",
                CATEGORIES,
                index=CATEGORIES.index(st.session_state.pb_category)
                      if st.session_state.pb_category in CATEGORIES else 0,
                help="Choose the type of entities you are benchmarking.",
            )
        with col2:
            entity_count = st.number_input(
                "Number of Entities",
                min_value=1,
                max_value=25,
                value=int(st.session_state.pb_entity_count),
                step=1,
                help="1–25 entities. Each will receive an individual forecast.",
            )
        with col3:
            start_month = st.selectbox(
                "Data Start Month",
                MONTH_NAMES,
                index=MONTH_NAMES.index(st.session_state.pb_start_month)
                      if st.session_state.pb_start_month in MONTH_NAMES else 0,
                help="First month of your 36-month history.",
            )

        st.markdown(f"##### {category} Names")
        # Preserve existing names; pad or trim to match count
        prev_names = st.session_state.pb_entity_names or []
        default_names = (
            prev_names + [f"{category[:-1] if category.endswith('s') else category} {i+1}"
                          for i in range(len(prev_names), int(entity_count))]
        )[:int(entity_count)]

        names_per_row = 5
        entity_names = []
        for row_start in range(0, int(entity_count), names_per_row):
            batch = default_names[row_start:row_start + names_per_row]
            cols  = st.columns(len(batch))
            for i, col in enumerate(cols):
                idx  = row_start + i
                name = col.text_input(
                    f"{category} {idx + 1}",
                    value=default_names[idx],
                    key=f"pb_ename_{idx}",
                )
                entity_names.append(name)

        submitted = st.form_submit_button("Save Setup →", use_container_width=True)

    if submitted:
        st.session_state.pb_category        = category
        st.session_state.pb_entity_count    = int(entity_count)
        st.session_state.pb_start_month     = start_month
        st.session_state.pb_entity_names    = entity_names
        st.session_state.pb_setup_committed = True
        # Reset downstream if config changed
        st.session_state.pb_data_committed  = False
        st.session_state.pb_run_done        = False
        st.session_state.pb_results         = None
        st.session_state.pb_wide            = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DATA ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def _build_empty_wide(entity_names: list, start_month: str) -> pd.DataFrame:
    month_labels = _build_month_labels(start_month, 36)
    year_labels  = _build_year_labels(start_month, 36)
    df = pd.DataFrame({
        "#":     list(range(1, 37)),
        "Month": month_labels,
        "Year":  year_labels,
    })
    for nm in entity_names:
        df[nm] = np.nan
    return df


def _year_subtotals(df_wide: pd.DataFrame, entity_names: list) -> pd.DataFrame:
    """Return a 3-row summary of annual totals per entity."""
    rows = []
    for yr in ["Yr1", "Yr2", "Yr3"]:
        mask = df_wide["Year"] == yr
        row  = {"Period": yr}
        for nm in entity_names:
            if nm in df_wide.columns:
                row[nm] = pd.to_numeric(df_wide.loc[mask, nm], errors="coerce").sum()
        rows.append(row)
    return pd.DataFrame(rows)


def _render_data_entry():
    entity_names = st.session_state.pb_entity_names
    start_month  = st.session_state.pb_start_month
    category     = st.session_state.pb_category

    st.markdown('<div class="pb-section">Data Entry</div>', unsafe_allow_html=True)

    # ── Init wide DataFrame ──
    if st.session_state.pb_wide is None:
        st.session_state.pb_wide = _build_empty_wide(entity_names, start_month)

    df_wide = st.session_state.pb_wide.copy()

    # Reconcile columns if setup changed
    fixed = ["#", "Month", "Year"]
    for nm in entity_names:
        if nm not in df_wide.columns:
            df_wide[nm] = np.nan
    extra = [c for c in df_wide.columns if c not in fixed + entity_names]
    df_wide.drop(columns=extra, inplace=True, errors="ignore")
    df_wide = df_wide[fixed + entity_names]
    st.session_state.pb_wide = df_wide

    # ── Input mode toggle ──
    input_mode = st.radio(
        "Data source",
        ["Manual Entry", "Upload CSV / Excel"],
        horizontal=True,
        key="pb_input_mode",
    )

    if input_mode == "Upload CSV / Excel":
        _render_csv_upload(entity_names, start_month)
        return

    # ── Manual: editable data grid ──
    st.markdown(
        f"Enter up to 36 months of values for each {category.lower()[:-1] if category.endswith('s') else category.lower()}. "
        f"**Index** and **Month** columns are locked. Only value cells are editable.",
        unsafe_allow_html=False,
    )

    # Column config: lock #, Month, Year; allow editing entity cols
    col_cfg = {
        "#":     st.column_config.NumberColumn("#", disabled=True, width="small"),
        "Month": st.column_config.TextColumn("Month", disabled=True, width="small"),
        "Year":  st.column_config.TextColumn("Year", disabled=True, width="small"),
    }
    for nm in entity_names:
        col_cfg[nm] = st.column_config.NumberColumn(nm, format="%.2f", min_value=0)

    with st.form("pb_data_form"):
        edited = st.data_editor(
            df_wide,
            column_config=col_cfg,
            hide_index=True,
            use_container_width=True,
            height=600,
            num_rows="fixed",
            key="pb_grid_editor",
        )

        # Annual subtotal preview (computed from current state, not from edited — avoids flicker)
        subtotals = _year_subtotals(edited, entity_names)
        st.markdown("**Annual Totals Preview**")
        st.dataframe(subtotals, hide_index=True, use_container_width=True)

        commit = st.form_submit_button("Commit Data →", use_container_width=True)

    if commit:
        # Merge fixed columns back to ensure no drift
        saved = edited.copy()
        saved["#"]     = df_wide["#"].values
        saved["Month"] = df_wide["Month"].values
        saved["Year"]  = df_wide["Year"].values
        st.session_state.pb_wide          = saved
        st.session_state.pb_data_committed = True
        st.session_state.pb_run_done      = False
        st.session_state.pb_results       = None
        st.success("Data committed. Ready to run forecasts.")
        st.rerun()


def _render_csv_upload(entity_names: list, start_month: str):
    """CSV/Excel upload handler."""
    st.markdown(
        "Upload a file with columns: **Month** (or index 1–36), then one column per entity. "
        "Column headers must match entity names exactly.",
    )

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="pb_file_uploader",
    )

    if uploaded is None:
        st.info("Upload a file to populate the data grid.")
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            df_up = pd.read_csv(io.BytesIO(uploaded.read()))
        else:
            df_up = pd.read_excel(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    df_up.columns = [str(c).strip() for c in df_up.columns]

    # Map entity columns (case-insensitive match)
    col_map = {c.lower(): c for c in df_up.columns}
    df_wide = _build_empty_wide(entity_names, start_month)

    matched  = []
    missing  = []
    for nm in entity_names:
        src = col_map.get(nm.lower())
        if src:
            vals = pd.to_numeric(df_up[src], errors="coerce").values
            n    = min(len(vals), 36)
            df_wide.loc[:n-1, nm] = vals[:n]
            matched.append(nm)
        else:
            missing.append(nm)

    if missing:
        st.warning(f"Columns not found in upload (left blank): {', '.join(missing)}")
    if matched:
        st.success(f"Loaded: {', '.join(matched)}")

    st.dataframe(df_wide, hide_index=True, use_container_width=True)

    if st.button("Commit Uploaded Data →", use_container_width=True):
        st.session_state.pb_wide           = df_wide
        st.session_state.pb_data_committed = True
        st.session_state.pb_run_done       = False
        st.session_state.pb_results        = None
        st.success("Uploaded data committed.")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — RUN FORECASTS
# ─────────────────────────────────────────────────────────────────────────────

def _render_run():
    entity_names = st.session_state.pb_entity_names
    df_wide      = st.session_state.pb_wide
    start_month  = st.session_state.pb_start_month
    month_labels = _build_month_labels(start_month, 36)

    st.markdown('<div class="pb-section">Forecast Engine</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(
            "Runs **ETS + ARIMA + Weighted Ensemble** on each entity. "
            "12-month backtest window · 12-month forward horizon · MASE accuracy metric.",
        )
    with col_b:
        run_btn = st.button("▶ Run All Forecasts", use_container_width=True, type="primary")

    if not (run_btn or st.session_state.pb_run_done):
        return

    if run_btn or not st.session_state.pb_results:
        results = {}
        errors  = []
        progress = st.progress(0, text="Running forecasts…")
        n = len(entity_names)

        for i, nm in enumerate(entity_names):
            progress.progress((i) / n, text=f"Forecasting: {nm}")
            if nm not in df_wide.columns:
                errors.append(nm)
                continue
            vals = pd.to_numeric(df_wide[nm], errors="coerce").tolist()
            res  = run_entity_forecast(vals, horizon=12, backtest_h=12)
            if "error" in res:
                errors.append(f"{nm}: {res['error']}")
            else:
                results[nm] = res

        progress.progress(1.0, text="Complete.")

        if errors:
            for e in errors:
                st.warning(f"Skipped — {e}")

        st.session_state.pb_results  = results
        st.session_state.pb_run_done = True

    _render_outputs(entity_names, df_wide, month_labels, start_month)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def _render_outputs(entity_names, df_wide, month_labels, start_month):
    results  = st.session_state.pb_results
    category = st.session_state.pb_category

    if not results:
        st.error("No forecasts could be generated. Check data and try again.")
        return

    fc_labels = [f"F{i+1}" for i in range(12)]

    # ── Build leaderboard ────────────────────────────────────────────────────
    rows = []
    for nm, res in results.items():
        primary = res["primary"]
        fc_total = float(np.nansum(primary["fc"]))
        hist_vals = pd.to_numeric(df_wide[nm], errors="coerce")
        hist_total = float(hist_vals.sum())

        # MASE
        mase_val = primary.get("mase", np.nan)

        # Trend direction
        fc_arr = primary["fc"]
        if len(fc_arr) >= 2:
            trend_pct = (fc_arr[-1] - fc_arr[0]) / abs(fc_arr[0]) * 100 if fc_arr[0] != 0 else 0
            trend_dir = "▲" if trend_pct > 1 else ("▼" if trend_pct < -1 else "→")
        else:
            trend_dir = "—"
            trend_pct = 0.0

        # YoY comparison: Yr3 actual vs Yr2 actual
        yr3 = pd.to_numeric(df_wide.loc[df_wide["Year"] == "Yr3", nm], errors="coerce").sum()
        yr2 = pd.to_numeric(df_wide.loc[df_wide["Year"] == "Yr2", nm], errors="coerce").sum()
        yoy_pct = (yr3 - yr2) / abs(yr2) * 100 if yr2 != 0 else np.nan

        rows.append({
            "Entity":          nm,
            "FC Total (12mo)": round(fc_total, 1),
            "Hist Total (36mo)": round(hist_total, 1),
            "YoY Chg (Yr3/Yr2)": round(yoy_pct, 1) if not np.isnan(yoy_pct) else None,
            "Trend":           f"{trend_dir} {trend_pct:+.1f}%",
            "MASE":            round(mase_val, 4) if not np.isnan(mase_val) else None,
            "Model":           primary["name"],
        })

    lb_df = pd.DataFrame(rows).sort_values("FC Total (12mo)", ascending=False).reset_index(drop=True)
    lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown('<div class="pb-section">Executive Summary</div>', unsafe_allow_html=True)

    n_entities  = len(results)
    total_fc    = lb_df["FC Total (12mo)"].sum()
    avg_mase    = lb_df["MASE"].mean() if lb_df["MASE"].notna().any() else None
    top_entity  = lb_df.iloc[0]["Entity"]

    kpi_cols = st.columns(4)
    kpi_data = [
        (str(n_entities), f"{category} Analyzed"),
        (f"{total_fc:,.0f}", "Total 12-Month Forecast"),
        (f"{avg_mase:.4f}" if avg_mase else "—", "Avg MASE"),
        (top_entity, "Top Performer"),
    ]
    for col, (val, lbl) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(
                f'<div class="pb-kpi-card">'
                f'<div class="pb-kpi-val">{val}</div>'
                f'<div class="pb-kpi-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Leaderboard chart ─────────────────────────────────────────────────────
    st.markdown('<div class="pb-section">Forecast Leaderboard</div>', unsafe_allow_html=True)
    st.plotly_chart(_leaderboard_chart(lb_df), use_container_width=True)

    # ── Leaderboard table ─────────────────────────────────────────────────────
    def _rank_badge(rank, total):
        if rank == 1:
            return "🥇 #1"
        elif rank <= max(2, total // 3):
            return f"🟢 #{rank}"
        elif rank >= total - total // 3:
            return f"🔴 #{rank}"
        return f"#{rank}"

    display_lb = lb_df.copy()
    display_lb["Rank"] = display_lb.apply(
        lambda r: _rank_badge(r["Rank"], n_entities), axis=1
    )
    st.dataframe(
        display_lb,
        hide_index=True,
        use_container_width=True,
        column_config={
            "FC Total (12mo)":    st.column_config.NumberColumn(format="%.1f"),
            "Hist Total (36mo)":  st.column_config.NumberColumn(format="%.1f"),
            "YoY Chg (Yr3/Yr2)": st.column_config.NumberColumn(format="%.1f %%"),
            "MASE":               st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.markdown(
        '<div class="pb-note">MASE < 1.0 = beats seasonal naïve baseline. '
        'Lower is better. Elite threshold: MASE < 0.70.</div>',
        unsafe_allow_html=True,
    )

    # ── Model performance table ───────────────────────────────────────────────
    st.markdown('<div class="pb-section">Model Performance by Entity</div>', unsafe_allow_html=True)

    model_rows = []
    for nm, res in results.items():
        row = {"Entity": nm}
        for key in ["ets", "arima", "ensemble"]:
            m = res.get(key)
            if m:
                mase = m.get("mase", np.nan)
                row[f"{key.upper()} MASE"] = round(mase, 4) if not np.isnan(mase) else None
            else:
                row[f"{key.upper()} MASE"] = None
        model_rows.append(row)

    model_df = pd.DataFrame(model_rows)
    st.dataframe(
        model_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "ETS MASE":      st.column_config.NumberColumn(format="%.4f"),
            "ARIMA MASE":    st.column_config.NumberColumn(format="%.4f"),
            "ENSEMBLE MASE": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # ── Per-entity charts ─────────────────────────────────────────────────────
    st.markdown('<div class="pb-section">Individual Entity Analysis</div>', unsafe_allow_html=True)

    # Sort by rank (highest FC first) for chart ordering
    ordered_entities = lb_df["Entity"].tolist()

    for nm in ordered_entities:
        if nm not in results:
            continue
        res = results[nm]
        ts  = res["ts"]
        entity_month_labels = month_labels[:len(ts)]

        with st.expander(f"📊 {nm}", expanded=(nm == ordered_entities[0])):
            tab_fc, tab_yoy = st.tabs(["12-Month Forecast", "Year-over-Year Seasonality"])

            with tab_fc:
                fig_fc = _forecast_chart(nm, ts, res, month_labels, horizon=12)
                st.plotly_chart(fig_fc, use_container_width=True)

                # Mini model comparison
                model_summary = []
                for key in ["ets", "arima", "ensemble"]:
                    m = res.get(key)
                    if m:
                        mase = m.get("mase", np.nan)
                        model_summary.append({
                            "Model": key.upper(),
                            "MASE":  round(mase, 4) if not np.isnan(mase) else "—",
                            "FC 12-mo Total": round(float(np.nansum(m["fc"])), 1),
                        })
                if model_summary:
                    st.dataframe(
                        pd.DataFrame(model_summary),
                        hide_index=True,
                        use_container_width=True,
                    )

            with tab_yoy:
                if len(ts) >= 13:
                    fig_yoy = _yoy_chart(nm, ts, month_labels)
                    st.plotly_chart(fig_yoy, use_container_width=True)

                    # Annotated YoY table
                    yoy_rows = []
                    for mo_idx in range(12):
                        mo_nm = MONTH_NAMES[mo_idx]
                        row   = {"Month": mo_nm}
                        for yr in range(1, 4):
                            flat_idx = (yr - 1) * 12 + mo_idx
                            if flat_idx < len(ts):
                                row[f"Year {yr}"] = round(ts[flat_idx], 2)
                            else:
                                row[f"Year {yr}"] = None
                        # YoY change columns
                        if row.get("Year 1") and row.get("Year 2"):
                            row["Yr1→Yr2 %"] = round((row["Year 2"] - row["Year 1"]) / abs(row["Year 1"]) * 100, 1)
                        if row.get("Year 2") and row.get("Year 3"):
                            row["Yr2→Yr3 %"] = round((row["Year 3"] - row["Year 2"]) / abs(row["Year 2"]) * 100, 1)
                        yoy_rows.append(row)

                    st.dataframe(pd.DataFrame(yoy_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("Year-over-Year view requires at least 13 months of data.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown('<div class="pb-section">Export</div>', unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_bytes = lb_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Leaderboard (CSV)",
            data=csv_bytes,
            file_name="veduta_peer_benchmarking_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        # Multi-sheet Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            lb_df.to_excel(writer, sheet_name="Leaderboard", index=False)
            model_df.to_excel(writer, sheet_name="Model_Performance", index=False)
            df_wide.to_excel(writer, sheet_name="Input_Data", index=False)

            for nm in ordered_entities:
                if nm not in results:
                    continue
                res = results[nm]
                primary = res["primary"]
                fc_df = pd.DataFrame({
                    "Period": fc_labels,
                    "Forecast": primary["fc"],
                    "Lower_95": primary["lo"],
                    "Upper_95": primary["hi"],
                })
                sheet_nm = nm[:28].replace("/", "-").replace("\\", "-")
                fc_df.to_excel(writer, sheet_name=sheet_nm, index=False)

        st.download_button(
            "Download Full Workbook (Excel)",
            data=buf.getvalue(),
            file_name="veduta_peer_benchmarking.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_peer_benchmarking():
    st.markdown(_CSS, unsafe_allow_html=True)
    _pb_init()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="pb-section">Peer Benchmarking</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Compare up to **25 entities** across 7 categories. Each entity receives a "
        "**12-month forecast** with confidence interval, **year-over-year seasonality** "
        "analysis, and a **MASE-scored model performance** breakdown.",
    )

    # ── Phase 1: Setup ───────────────────────────────────────────────────────
    _render_setup()

    if not st.session_state.pb_setup_committed:
        return

    # ── Phase 2: Data Entry ──────────────────────────────────────────────────
    _render_data_entry()

    if not st.session_state.pb_data_committed:
        return

    # ── Phase 3: Run + Outputs ───────────────────────────────────────────────
    _render_run()
