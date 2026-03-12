# ==================================================
# FILE: forecastiq/tabs/home.py
# VERSION: 3.0.0
# ROLE: EXECUTIVE HOME — DATA INTAKE + COMMAND DASHBOARD
# DESIGN: Luxury dark financial terminal
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from io import StringIO, BytesIO

from utils.validation import validate_time_series
from utils.frequency import infer_frequency
from utils.hashing import hash_dataframe
from utils.visuals import EXECUTIVE_COLORS

# ── Auto-Intelligence ─────────────────────────────────────────────────────────
def _ai_insight(section_key: str, context: str, force: bool = False):
    """
    Render an AI-generated executive briefing for a section.
    Uses Anthropic API. Results cached in session state.
    Only renders when auto_intelligence is True.
    """
    if not st.session_state.get("auto_intelligence", False):
        return

    cache = st.session_state.get("ai_insights_cache", {})

    if section_key not in cache or force:
        with st.spinner("Auto-Intelligence generating insight…"):
            try:
                import urllib.request, json as _json
                payload = {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": (
                        "You are VEDUTA's Auto-Intelligence engine. "
                        "You receive real data from an executive decision intelligence platform "
                        "and write concise, actionable C-suite briefings. "
                        "Be direct. Lead with the most important insight. "
                        "Use plain English. No bullet points. 2-4 sentences maximum. "
                        "Reference the actual numbers provided."
                    ),
                    "messages": [{"role": "user", "content": context}],
                }
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=_json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read())
                text = data["content"][0]["text"] if data.get("content") else "Insight unavailable."
                cache[section_key] = text
                st.session_state.ai_insights_cache = cache
            except Exception as e:
                cache[section_key] = f"Auto-Intelligence unavailable: {str(e)[:80]}"
                st.session_state.ai_insights_cache = cache

    insight = cache.get(section_key, "")
    if insight:
        st.markdown(
            f'''<div style="background:linear-gradient(135deg,rgba(200,151,74,.08),rgba(200,151,74,.04));
            border:1px solid rgba(200,151,74,.25);border-left:3px solid #C8974A;
            border-radius:8px;padding:.85rem 1.1rem;margin:.5rem 0 1rem 0">
            <div style="font-family:DM Mono,monospace;font-size:.58rem;letter-spacing:.12em;
            text-transform:uppercase;color:#C8974A;margin-bottom:.35rem">
            ⚡ Auto-Intelligence</div>
            <div style="font-family:DM Mono,monospace;font-size:.72rem;color:#E2B96A;
            line-height:1.6">{insight}</div></div>''',
            unsafe_allow_html=True,
        )



# ── Design tokens ─────────────────────────────────────────────────────────────
_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #07080F; }
[data-testid="stSidebar"]          { background: #0D1420; border-right: 1px solid #243347; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

.exec-card {
    background: linear-gradient(135deg, #1B2A40 0%, #152033 100%);
    border: 1px solid #243347;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}
.exec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #C8974A 0%, #E2B96A 50%, #C8974A 100%);
    background-size: 200% 100%;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
}
.kpi-tile {
    background: #1B2A40;
    border: 1px solid #243347;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    transition: border-color .2s, transform .15s;
    cursor: default;
}
.kpi-tile:hover { border-color: #C8974A; transform: translateY(-2px); }
.kpi-label {
    font-family: 'DM Mono', 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    font-weight: 500;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #4A6278;
    margin-bottom: .4rem;
}
.kpi-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 300;
    color: #EDE8DE;
    line-height: 1.05;
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    margin-top: .3rem;
    display: flex;
    align-items: center;
    gap: .25rem;
}
.kpi-delta.pos { color: #6BAF85; }
.kpi-delta.neg { color: #C45858; }
.kpi-delta.neu { color: #4A6278; }

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #C8974A;
    margin: 2.25rem 0 .85rem 0;
    display: flex;
    align-items: center;
    gap: .6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #243347 0%, transparent 100%);
}

.profile-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: .75rem;
    background: #1B2A40;
    border: 1px solid #243347;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: .75rem 0 1.25rem 0;
}
.p-item { text-align: center; }
.p-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #4A6278;
    margin-bottom: .2rem;
}
.p-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 400;
    color: #EDE8DE;
}

.scenario-banner {
    background: linear-gradient(90deg, rgba(220,38,38,.09), rgba(220,38,38,.03));
    border: 1px solid rgba(220,38,38,.25);
    border-left: 3px solid #dc2626;
    border-radius: 8px;
    padding: .75rem 1.25rem;
    margin: 1rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #fca5a5;
    letter-spacing: .04em;
}

[data-testid="stExpander"] {
    background: #1B2A40 !important;
    border: 1px solid #243347 !important;
    border-radius: 10px !important;
    margin-bottom: .6rem !important;
}

hr { border-color: #243347 !important; margin: 1.75rem 0 !important; }

[data-testid="stFileUploadDropzone"] {
    background: #1B2A40 !important;
    border: 1.5px dashed #243347 !important;
    border-radius: 10px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #C8974A !important; }

/* ── Tab labels — broad selector coverage across Streamlit versions ─────── */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    padding: .65rem 1.25rem !important;
}
button[data-baseweb="tab"]:hover {
    color: #8FA3B8 !important;
    background: rgba(200,151,74,.07) !important;
    border-radius: 4px 4px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #EDE8DE !important;
    border-bottom: 2px solid #C8974A !important;
}
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #243347 !important;
    gap: 0 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #C8974A !important;
    height: 2px !important;
}
div[data-baseweb="tab-border"] {
    background-color: #243347 !important;
    height: 1px !important;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
"""

# ── Chart base ────────────────────────────────────────────────────────────────
_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0D1420",
    plot_bgcolor="#0D1420",
    font=dict(family="DM Mono, monospace", color="#8FA3B8", size=11),
    margin=dict(l=56, r=32, t=72, b=52),
    legend=dict(
        orientation="h", y=-0.18, x=0,
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0D1420", bordercolor="#243347",
        font=dict(family="DM Mono, monospace", size=11, color="#EDE8DE"),
    ),
    xaxis=dict(
        gridcolor="#1B2A40", zeroline=False,
        showspikes=True, spikecolor="#C8974A",
        spikethickness=1, spikedash="dot",
    ),
    yaxis=dict(gridcolor="#1B2A40", zeroline=False),
    dragmode="zoom",
)


def _layout(title: str, sub: str = "", height: int = 420) -> dict:
    d = dict(_BASE)
    d["height"] = height
    d["title"] = dict(
        text=(
            f"<span style='font-family:Cormorant Garamond,serif;font-size:16px;"
            f"font-weight:400;color:#EDE8DE;letter-spacing:0.04em'>{title}</span>"
            + (f"<br><span style='font-size:10px;color:#4A6278'>{sub}</span>" if sub else "")
        ),
        x=0.01, xanchor="left", y=0.97, yanchor="top",
    )
    return d


# ── Utility ───────────────────────────────────────────────────────────────────
def _fmt(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v:,.{d}f}"
    return f"{v:.{d}f}"


# Common column name aliases accepted as date or value headers
_DATE_ALIASES  = {"date", "data", "dates", "period", "time", "month",
                   "year", "quarter", "week", "timestamp", "dt"}
_VALUE_ALIASES = {"value", "values", "val", "amount", "qty", "quantity",
                   "sales", "revenue", "units", "count", "total", "volume",
                   "price", "rate", "number", "num", "metric", "measure"}


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = (raw.columns.astype(str).str.strip()
                   .str.replace("\ufeff", "", regex=False).str.lower())

    # Exact match first
    if "date" in raw.columns and "value" in raw.columns:
        return raw

    # Alias matching — e.g. "data"->date, "sales"->value
    rename_map = {}
    for col in raw.columns:
        if col in _DATE_ALIASES and "date" not in rename_map.values():
            rename_map[col] = "date"
        elif col in _VALUE_ALIASES and "value" not in rename_map.values():
            rename_map[col] = "value"
    if rename_map:
        raw = raw.rename(columns=rename_map)
    if "date" in raw.columns and "value" in raw.columns:
        return raw

    # Two-column file with unrecognised names — assume col0=date, col1=value
    if len(raw.columns) == 2:
        raw.columns = ["date", "value"]
        return raw

    # Date-detection fallback: find the most date-like column
    for col in raw.columns:
        if pd.to_datetime(raw[col], errors="coerce").notna().mean() > 0.8:
            others = [c for c in raw.columns if c != col]
            if len(others) == 1:
                return raw.rename(columns={col: "date", others[0]: "value"})
    return raw


def _read_any(source, filename: str = "") -> pd.DataFrame | None:
    if filename.endswith((".xls", ".xlsx")):
        try:
            return _normalise(pd.read_excel(source, header=0))
        except Exception:
            return None
    for hdr in [0, None]:
        try:
            buf = BytesIO(source) if isinstance(source, bytes) else source
            if hasattr(buf, "seek"):
                buf.seek(0)
            raw = pd.read_csv(buf, sep=None, engine="python",
                              header=hdr, encoding_errors="replace")
            if hdr is None:
                raw.columns = (["date", "value"] if raw.shape[1] == 2
                                else [str(i) for i in range(raw.shape[1])])
            else:
                raw = _normalise(raw)
            if "date" in raw.columns:
                return raw
        except Exception:
            pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def render_home():
    st.markdown(_CSS, unsafe_allow_html=True)

    # Page heading
    st.markdown("""
    <div style="margin-bottom:1.75rem">
      <div style="font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.22em;
                  text-transform:uppercase;color:#C8974A;margin-bottom:.3rem">
        VEDUTA · Executive Intelligence
      </div>
      <div style="font-family:'Cormorant Garamond',serif;font-size:2.4rem;font-weight:300;
                  color:#EDE8DE;line-height:1.1;letter-spacing:0.04em">
        The Veduta
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:.7rem;color:#4A6278;
                  margin-top:.35rem">
        Commit your time series data · Configure the Foresight Engine · The veduta is clear.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 01 · DATA INTAKE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">01 · Data Intake</div>',
                unsafe_allow_html=True)

    col_up, col_paste = st.columns(2, gap="large")
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xls", "xlsx"],
            help="Date in first column, value in second. Headers optional.",
        )
    with col_paste:
        pasted_data = st.text_area(
            "Paste Data",
            placeholder="2024-01-01\t12345\n2024-02-01\t13201\n...",
            height=110,
            help="Tab, comma, or space separated.",
        )

    # Manual entry editor — only show when no committed data and no upload/paste
    df = None
    if uploaded_file is not None:
        df = _read_any(uploaded_file.read(), uploaded_file.name)
        if df is None:
            st.error("Unable to read file. Ensure it is a valid CSV or Excel.")
    elif pasted_data.strip():
        df = _read_any(StringIO(pasted_data))
        if df is None:
            st.error("Unable to parse pasted data. Use tab or comma separated format.")
    else:
        # Show manual editor only if no committed data yet
        if st.session_state.committed_df is None:
            manual_df = st.data_editor(
                pd.DataFrame(columns=["date", "value"]),
                num_rows="dynamic",
                key="manual_data_editor",
                use_container_width=True,
                column_config={
                    "date":  st.column_config.TextColumn("Date", help="YYYY-MM-DD"),
                    "value": st.column_config.NumberColumn("Value", format="%.2f"),
                },
            )
            if not manual_df.empty:
                df = _normalise(manual_df.copy())

    if df is not None and not df.empty:
        # Drop any rows where date cannot be parsed — catches stray header rows
        # that slip through when hdr=None fallback ran unnecessarily
        _parsed = pd.to_datetime(df["date"], errors="coerce")
        df = df[_parsed.notna()].copy()
        df["date"] = _parsed[_parsed.notna()]
        df = df.reset_index(drop=True)

        validation = validate_time_series(df)

        if not validation["valid"]:
            st.error("**Data validation failed.**")
            for e in validation["errors"]:
                st.markdown(f"- {e}")
            if validation["failed_rows"]:
                with st.expander("Failed rows"):
                    st.write(validation["failed_rows"])
        else:
            freq_info = infer_frequency(pd.to_datetime(df["date"]))
            _show_profile({
                "rows":            len(df),
                "start_date":      pd.to_datetime(df["date"]).min(),
                "end_date":        pd.to_datetime(df["date"]).max(),
                "frequency":       freq_info["frequency"],
                "confidence":      freq_info["confidence"],
                "duplicate_count": validation["duplicate_count"],
                "missing_periods": validation["missing_periods"],
            })

            if freq_info["confidence"] < 0.7:
                st.warning(
                    f"Frequency inferred as **{freq_info['frequency']}** with low confidence "
                    f"({freq_info['confidence']:.0%}). Verify data cadence before committing."
                )

            col_btn, col_note = st.columns([1, 4])
            with col_btn:
                commit = st.button("SAVE / COMMIT DATA", use_container_width=True)
            with col_note:
                st.markdown(
                    "<div style='font-family:DM Mono,monospace;font-size:.62rem;"
                    "color:#4A6278;padding-top:.7rem'>"
                    "Locking this dataset as the certified baseline for all analyses.</div>",
                    unsafe_allow_html=True,
                )

            if commit:
                committed_df = df.copy()
                committed_df["date"] = pd.to_datetime(committed_df["date"])
                committed_df = committed_df.sort_values("date").reset_index(drop=True)
                fp = hash_dataframe(committed_df)
                st.session_state.committed_df           = committed_df
                st.session_state.data_frequency         = freq_info["frequency"]
                st.session_state.freq_inference_details = freq_info
                st.session_state.committed_meta = {
                    "rows":             len(committed_df),
                    "start_date":       committed_df["date"].min(),
                    "end_date":         committed_df["date"].max(),
                    "frequency":        freq_info["frequency"],
                    "confidence":       freq_info["confidence"],
                    "data_fingerprint": fp,
                    "duplicate_count":  validation["duplicate_count"],
                    "missing_periods":  validation["missing_periods"],
                }
                st.session_state.audit_log.append({
                    "event": "data_commit", "timestamp": datetime.utcnow().isoformat(),
                    "rows": len(committed_df), "frequency": freq_info["frequency"],
                    "fingerprint": fp,
                })
                st.success(
                    f"✓ Committed — {len(committed_df):,} rows · "
                    f"{freq_info['frequency']} · "
                    f"{committed_df['date'].min().strftime('%b %Y')} – "
                    f"{committed_df['date'].max().strftime('%b %Y')}"
                )
    # Always show committed data table if available and no new data being loaded
    if df is None and st.session_state.committed_df is not None:
        _show_profile(st.session_state.committed_meta)
        st.dataframe(
            st.session_state.committed_df,
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.committed_df is None:
        return

    hist_df = st.session_state.committed_df.copy()
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    # ─────────────────────────────────────────────────────────────────────────
    # 02 · DATA INTELLIGENCE (always visible once data committed)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">02 · Data Intelligence</div>',
                unsafe_allow_html=True)
    _data_intelligence(hist_df)

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST GATE
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.latest_forecast_df is None:
        st.markdown(
            '<div class="exec-card" style="text-align:center;padding:2.75rem 1.5rem">'
            '<div style="font-family:DM Mono,monospace;font-size:.6rem;letter-spacing:.18em;'
            'text-transform:uppercase;color:#C8974A;margin-bottom:.6rem">'
            'Foresight Engine Ready</div>'
            '<div style="font-family:Cormorant Garamond,serif;font-size:1.1rem;font-weight:300;'
            'color:#8FA3B8;line-height:1.5">'
            'Select tier · set horizon · click RUN FORECAST in the sidebar<br>'
            '<span style="font-size:.85rem;color:#4A6278">'
            'The full analytics suite will unlock after the first run.'
            '</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 03 · PERFORMANCE COMMAND STRIP
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">03 · Performance Command Strip</div>',
                unsafe_allow_html=True)
    _kpi_strip(hist_df)

    # ─────────────────────────────────────────────────────────────────────────
    # 04 · FORECAST TRAJECTORY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">04 · Forecast Trajectory</div>',
                unsafe_allow_html=True)
    _hero_chart(hist_df)

    # ─────────────────────────────────────────────────────────────────────────
    # 05 · MODEL INTELLIGENCE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">05 · Model Intelligence</div>',
                unsafe_allow_html=True)
    _model_intelligence(hist_df)

    # ─────────────────────────────────────────────────────────────────────────
    # 06 · SCENARIO STRESS TEST (conditional)
    # ─────────────────────────────────────────────────────────────────────────
    if (st.session_state.scenario_state.get("enabled")
            and st.session_state.scenario_forecast_df is not None):
        st.markdown('<div class="section-header">06 · Scenario Stress Test</div>',
                    unsafe_allow_html=True)
        _scenario_overlay(hist_df)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _show_profile(meta: dict):
    start = meta["start_date"].strftime("%b %Y") if meta.get("start_date") else "—"
    end   = meta["end_date"].strftime("%b %Y")   if meta.get("end_date")   else "—"
    freq  = meta.get("frequency") or "Unknown"
    conf  = meta.get("confidence", 0)
    rows  = meta.get("rows", 0)
    dupes = meta.get("duplicate_count", 0)
    gaps  = len(meta.get("missing_periods", []))

    freq_color = "#22c55e" if conf >= 0.8 else "#f59e0b"
    st.markdown(f"""
    <div class="profile-strip">
      <div class="p-item">
        <div class="p-label">Rows</div>
        <div class="p-value">{rows:,}</div>
      </div>
      <div class="p-item">
        <div class="p-label">Start</div>
        <div class="p-value">{start}</div>
      </div>
      <div class="p-item">
        <div class="p-label">End</div>
        <div class="p-value">{end}</div>
      </div>
      <div class="p-item">
        <div class="p-label">Frequency</div>
        <div class="p-value" style="color:{freq_color}">{freq}</div>
      </div>
      <div class="p-item">
        <div class="p-label">Duplicates</div>
        <div class="p-value" style="color:{'#ef4444' if dupes else '#22c55e'}">{dupes}</div>
      </div>
      <div class="p-item">
        <div class="p-label">Gaps</div>
        <div class="p-value" style="color:{'#f59e0b' if gaps else '#22c55e'}">{gaps}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── 02 Data Intelligence ──────────────────────────────────────────────────────
def _data_intelligence(hist_df: pd.DataFrame):
    t1, t2, t3, t4 = st.tabs([
        "Actual Series",
        "Rolling Analytics",
        "Seasonality",
        "Distribution",
    ])

    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df["value"],
            mode="lines", name="Actual",
            line=dict(color="#C8974A", width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
            hovertemplate="<b>%{x|%b %Y}</b> — %{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(**_layout(
            "Historical Time Series",
            "Committed baseline · Drag to zoom · Double-click to reset"
        ))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        _rolling(hist_df)

    with t3:
        _seasonality(hist_df)

    with t4:
        _distribution(hist_df)


def _rolling(hist_df: pd.DataFrame):
    df = hist_df.copy().set_index("date")
    df["MoM"]   = df["value"].pct_change() * 100
    df["YoY"]   = df["value"].pct_change(12) * 100
    df["R3"]    = df["value"].rolling(3).mean()
    df["R12"]   = df["value"].rolling(12).mean()
    df = df.reset_index()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(x=df["date"], y=df["value"], mode="lines",
        name="Actual", line=dict(color="#475569", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["R3"], mode="lines",
        name="3-Period MA", line=dict(color="#C8974A", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["R12"], mode="lines",
        name="12-Period MA", line=dict(color="#52B8B2", width=2, dash="dash")), row=1, col=1)

    bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["MoM"].fillna(0)]
    fig.add_trace(go.Bar(x=df["date"], y=df["MoM"], name="MoM %",
        marker_color=bar_colors, opacity=0.75), row=2, col=1)

    layout = _layout("Rolling Analytics", "Moving averages · MoM momentum", height=480)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#1a2540", zeroline=False)
    fig.update_yaxes(gridcolor="#1a2540", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


def _seasonality(hist_df: pd.DataFrame):
    df = hist_df.copy()
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    years = sorted(df["year"].unique())
    palette = ["#A67835","#C8974A","#E2B96A","#52B8B2","#3A8C8C",
               "#6BAF85","#4A6278","#8FA3B8","#D4834A","#C45858"]
    cmap = {yr: palette[i % len(palette)] for i, yr in enumerate(years)}

    fig = go.Figure()
    for yr in years:
        g = df[df["year"] == yr].sort_values("month")
        fig.add_trace(go.Scatter(
            x=g["month"], y=g["value"],
            mode="lines+markers", name=str(yr),
            line=dict(color=cmap[yr], width=1.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{yr}</b> — %{{x}}<br>%{{y:,.2f}}<extra></extra>",
        ))
    fig.update_layout(**_layout(
        "Seasonal Pattern by Year",
        "Year-over-year monthly rhythm · Toggle years in legend"
    ))
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1,13)),
        ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-Intelligence — Seasonality
    months_present = df["month"].nunique()
    years_present  = df["year"].nunique()
    _ai_insight(
        "seasonality",
        f"Seasonal pattern analysis. Data spans {years_present} years, "
        f"{months_present} months visible. "
        f"Provide 2 sentences on what the seasonal pattern suggests for planning and "
        f"any notable year-over-year shifts the executive should be aware of."
    )

    # Heatmap
    if len(years) > 1:
        pivot = df.pivot_table(index="year", columns="month", values="value", aggfunc="mean")
        mon_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [mon_labels[m-1] for m in pivot.columns]
        hfig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index],
            colorscale=[[0,"#07080F"],[0.5,"#C8974A"],[1,"#E2B96A"]],
            hovertemplate="<b>%{y} %{x}</b><br>%{z:,.2f}<extra></extra>",
        ))
        hfig.update_layout(**_layout(
            "Seasonality Heat Map",
            "Average value by month and year"
        ))
        st.plotly_chart(hfig, use_container_width=True)


def _distribution(hist_df: pd.DataFrame):
    vals = hist_df["value"].dropna()
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        fig = go.Figure(go.Histogram(
            x=vals, nbinsx=30, name="Distribution",
            marker=dict(color="#C8974A", opacity=0.75,
                        line=dict(color="#1e40af", width=0.5)),
        ))
        fig.update_layout(**_layout("Value Distribution", "Frequency histogram"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Box(
            y=vals, name="Values",
            marker_color="#C8974A", line_color="#E2B96A",
            fillcolor="rgba(37,99,235,0.12)",
            boxmean=True, boxpoints="outliers",
            hovertemplate="%{y:,.2f}<extra></extra>",
        ))
        fig2.update_layout(**_layout("Box & Whisker", "Quartiles · Median · Outliers"))
        st.plotly_chart(fig2, use_container_width=True)

    p25, p75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    stats = [
        ("Mean",    float(vals.mean())),
        ("Median",  float(vals.median())),
        ("Std Dev", float(vals.std())),
        ("Min",     float(vals.min())),
        ("Max",     float(vals.max())),
        ("IQR",     p75 - p25),
    ]
    html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:.5rem;margin-top:.5rem">'
    for lbl, val in stats:
        html += (
            f'<div style="background:#1B2A40;border:1px solid #243347;border-radius:8px;'
            f'padding:.75rem;text-align:center">'
            f'<div style="font-family:DM Mono,monospace;font-size:.58rem;letter-spacing:.1em;'
            f'text-transform:uppercase;color:#4A6278;margin-bottom:.2rem">{lbl}</div>'
            f'<div style="font-family:Cormorant Garamond,serif;font-size:1rem;font-weight:400;'
            f'color:#EDE8DE">{_fmt(val)}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── 03 KPI Strip ──────────────────────────────────────────────────────────────
def _kpi_strip(hist_df: pd.DataFrame):
    pe         = st.session_state.sentinel_primary_df
    latest     = float(hist_df["value"].iloc[-1])
    prior      = float(hist_df["value"].iloc[-2]) if len(hist_df) > 1 else None
    nxt        = float(pe.iloc[0]) if pe is not None else None
    mom        = latest - prior if prior is not None else None
    fwd_delta  = nxt - latest   if nxt is not None   else None
    horizon    = (st.session_state.run_signature or {}).get("forecast_horizon", "—")
    tier       = (st.session_state.sentinel_active_tier or "—").title()

    cert = st.session_state.sentinel_cert_metadata or []
    pe_cert  = next((c for c in cert if c.get("model") == "Primary Ensemble"), None)
    mase_val = pe_cert.get("MASE") if pe_cert else None

    def _tile(label, value, delta=None, dlabel=""):
        dc = ("pos" if delta and delta > 0 else "neg" if delta and delta < 0 else "neu")
        sign = "▲" if delta and delta > 0 else ("▼" if delta and delta < 0 else "●")
        d_html = (f'<div class="kpi-delta {dc}">{sign} {_fmt(abs(delta))} {dlabel}</div>'
                  if delta is not None else "")
        return (f'<div class="kpi-tile">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'{d_html}</div>')

    html = '<div class="kpi-grid">'
    html += _tile("Latest Actual",          _fmt(latest),                   mom,       "MoM")
    html += _tile("Next Period Forecast",   _fmt(nxt) if nxt else "—",       fwd_delta, "vs actual")
    html += _tile("Primary Ensemble MASE",  _fmt(mase_val, 4) if mase_val else "—")
    html += _tile(f"Horizon · {tier} Tier", f"{horizon}p")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Auto-Intelligence — KPI Strip
    _kpi_horizon = st.session_state.forecast_horizon or 12
    _kpi_tier    = (st.session_state.sentinel_active_tier or "—").title()
    _ai_insight(
        "kpi_strip",
        f"Executive KPI summary. Latest actual: {_fmt(latest)}. "
        f"Next period forecast: {_fmt(nxt) if nxt else 'N/A'}. "
        f"MoM change: {_fmt(mom) if mom is not None else 'N/A'}. "
        f"Primary Ensemble MASE: {_fmt(mase_val, 4) if mase_val else 'N/A'}. "
        f"Forecast horizon: {_kpi_horizon} periods. Tier: {_kpi_tier}. "
        f"In 2 sentences, tell the executive what these KPIs mean for decision-making."
    )


# ── 04 Hero Chart ─────────────────────────────────────────────────────────────
def _hero_chart(hist_df: pd.DataFrame):
    pe    = st.session_state.sentinel_primary_df
    se    = st.session_state.sentinel_stacked_df
    pe_ci = (st.session_state.latest_intervals or {}).get("Primary Ensemble")
    tier  = (st.session_state.sentinel_active_tier or "").title()
    ver   = st.session_state.sentinel_engine_version or "2.0.0"

    fig = go.Figure()

    # CI band (drawn first so it sits behind lines)
    if pe is not None and pe_ci is not None:
        lower, upper = pe_ci
        fig.add_trace(go.Scatter(
            x=list(upper.index) + list(lower.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.09)",
            line=dict(width=0), name="95% CI", hoverinfo="skip",
        ))

    # Actuals
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["value"],
        mode="lines", name="Actual",
        line=dict(color="#64748b", width=1.5),
        hovertemplate="<b>Actual</b> %{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
    ))

    # Primary Ensemble
    if pe is not None:
        fig.add_trace(go.Scatter(
            x=pe.index, y=pe.values,
            mode="lines", name="Primary Ensemble",
            line=dict(color="#C8974A", width=3),
            hovertemplate="<b>Primary Ensemble</b> %{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
        ))

    # Stacked Ensemble
    if se is not None:
        fig.add_trace(go.Scatter(
            x=se.index, y=se.values,
            mode="lines", name="Stacked Ensemble",
            line=dict(color="#a78bfa", width=2, dash="dot"),
            hovertemplate="<b>Stacked Ensemble</b> %{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
        ))

    # Forecast origin vertical
    origin_str = hist_df["date"].max().strftime("%Y-%m-%d")
    fig.add_shape(type="line",
        x0=origin_str, x1=origin_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#1e2a45", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=origin_str, y=1, xref="x", yref="paper",
        text="Forecast Origin",
        showarrow=False, yanchor="bottom",
        font=dict(size=9, color="#4b5e80", family="DM Mono, monospace"),
    )

    fig.update_layout(**_layout(
        "Forecast Trajectory",
        f"Foresight Engine v{ver} · {tier} Tier · Primary Ensemble (solid) · "
        f"Stacked Ensemble (dotted) · 95% CI (shaded)",
        height=460,
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.session_state.latest_fig_inputs = fig

    # Auto-Intelligence — Forecast Trajectory
    pe_val    = float(pe.iloc[0])  if pe is not None and len(pe) > 0 else None
    hist_last = float(hist_df["value"].iloc[-1]) if len(hist_df) > 0 else None
    pe_end    = float(pe.iloc[-1]) if pe is not None and len(pe) > 0 else None
    _pe_val_str  = f"{pe_val:,.2f}"  if pe_val    is not None else "N/A"
    _pe_end_str  = f"{pe_end:,.2f}"  if pe_end    is not None else "N/A"
    _hist_str    = f"{hist_last:,.2f}" if hist_last is not None else "N/A"
    _ci_str      = f"{int(st.session_state.ci_level * 100)}%"
    horizon      = st.session_state.forecast_horizon or 12
    _ai_insight(
        "forecast_trajectory",
        f"Forecast trajectory for time series. "
        f"Latest actual value: {_hist_str}. "
        f"Primary Ensemble next-period forecast: {_pe_val_str}. "
        f"End-of-horizon forecast ({horizon}p): {_pe_end_str}. "
        f"Engine tier: {tier}. CI level: {_ci_str}. "
        f"Provide a 2-3 sentence executive briefing on the forecast direction, magnitude, and key risk."
    )

    st.markdown(
        '<div style="font-family:DM Mono,monospace;font-size:.62rem;color:#4A6278;'
        'margin-top:-.4rem">◉ Hover for values · Drag to zoom · Double-click to reset</div>',
        unsafe_allow_html=True,
    )


# ── 05 Model Intelligence ─────────────────────────────────────────────────────
def _model_intelligence(hist_df: pd.DataFrame):
    t1, t2, t3 = st.tabs([
        "Model Scorecard",
        "All-Model Overlay",
        "Trend Decomposition",
    ])

    with t1:
        cert = st.session_state.sentinel_cert_metadata
        if not cert:
            st.info("No certification metadata available.")
            return

        cert_df = pd.DataFrame(cert)

        def mase_fmt(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "—"
            lbl = ("Elite" if v < 0.70 else "Strong" if v < 0.85
                   else "Pass" if v < 1.00 else "Fail")
            return f"{v:.4f}  ({lbl})"

        cert_df["MASE_display"] = cert_df["MASE"].apply(mase_fmt)

        # Certification column — works with cert_tier or readiness_tier
        tier_col = "cert_tier" if "cert_tier" in cert_df.columns else None

        def _map_tier(raw):
            """Map any sentinel readiness_tier string to display label."""
            if not raw or str(raw) in ("nan", "—", "None"):
                return ("—", "—")
            s = str(raw)
            if "Tier 1" in s or s == "Elite":
                return ("🟢 Elite", "Elite")
            if "Tier 2" in s or s == "Strong":
                return ("🟡 Strong", "Strong")
            if "Tier 3" in s or s in ("Pass", "Weak"):
                return ("🟠 Pass", "Pass")
            if "Tier 4" in s or s == "Fail" or "Failure" in s:
                return ("🔴 Fail", "Fail")
            # Ensemble labels — show as-is
            if "Ensemble" in s:
                return (f"⬡ {s}", s)
            return (s, s)

        if tier_col:
            mapped = cert_df[tier_col].apply(_map_tier)
            cert_df["Cert"]       = mapped.apply(lambda x: x[0])
            cert_df["_tier_simple"] = mapped.apply(lambda x: x[1])
        else:
            cert_df["Cert"] = "—"
            cert_df["_tier_simple"] = "—"

        cols = [c for c in ["model","MASE_display","Cert","ci_method","active_tier"]
                if c in cert_df.columns]
        st.dataframe(
            cert_df[cols].rename(columns={
                "model":"Model","MASE_display":"MASE",
                "Cert":"Certification","ci_method":"CI Method","active_tier":"Tier",
            }),
            use_container_width=True, hide_index=True,
        )

        # Auto-Intelligence — Model Scorecard
        if cert_df is not None and len(cert_df) > 0:
            elite_n   = (cert_df.get("_tier_simple", pd.Series()) == "Elite").sum()
            strong_n  = (cert_df.get("_tier_simple", pd.Series()) == "Strong").sum()
            fail_n    = (cert_df.get("_tier_simple", pd.Series()) == "Fail").sum()
            best_mase = cert_df["MASE"].dropna().min() if "MASE" in cert_df.columns else None
            _best_mase_str = f"{best_mase:.4f}" if best_mase is not None else "N/A"
            _ai_insight(
                "model_scorecard",
                f"Model certification results. {len(cert_df)} models evaluated. "
                f"Elite tier (MASE<0.70): {elite_n}. Strong tier (MASE<0.85): {strong_n}. "
                f"Fail tier (MASE>=1.00): {fail_n}. "
                f"Best single-model MASE: {_best_mase_str}. "
                f"In 2 sentences, interpret what this certification distribution means for "
                f"forecast reliability and what the executive should know."
            )

        # Cert distribution bar — uses pre-computed _tier_simple column
        if tier_col and "_tier_simple" in cert_df.columns:
            tc = (cert_df["_tier_simple"]
                  .value_counts()
                  .reindex(["Elite","Strong","Pass","Fail"], fill_value=0))
            tc = tc[tc > 0]
            if not tc.empty:
                color_map = {"Elite":"#22c55e","Strong":"#f59e0b",
                             "Pass":"#f97316","Fail":"#ef4444"}
                bar = go.Figure(go.Bar(
                    x=tc.index.tolist(), y=tc.values.tolist(),
                    marker_color=[color_map.get(t,"#64748b") for t in tc.index],
                    text=tc.values.tolist(), textposition="auto",
                ))
                bar.update_layout(**_layout(
                    "Certification Distribution",
                    "M-Competition tier breakdown — Elite < 0.70 · Strong < 0.85 · Pass < 1.00 · Fail ≥ 1.00",
                    height=260,
                ))
                bar.update_layout(showlegend=False)
                st.plotly_chart(bar, use_container_width=True)

    with t2:
        forecasts = st.session_state.latest_forecasts
        if not forecasts:
            st.info("No model outputs available.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["value"],
                mode="lines", name="Actual",
                line=dict(color="#475569", width=1.5),
            ))
            for model, series in forecasts.items():
                if model == "Primary Ensemble":
                    lp = dict(color="#C8974A", width=3)
                    vis = True
                elif model == "Stacked Ensemble":
                    lp = dict(color="#a78bfa", width=2, dash="dot")
                    vis = True
                else:
                    lp = dict(width=1, color="#334155")
                    vis = "legendonly"
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    mode="lines", name=model,
                    line=lp, visible=vis,
                    hovertemplate=f"<b>{model}</b> %{{x|%b %Y}}<br>%{{y:,.2f}}<extra></extra>",
                ))
            fig.update_layout(**_layout(
                "All Model Outputs",
                "Primary + Stacked visible · Toggle models in legend",
                height=440,
            ))
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        _decomposition(hist_df)


def _decomposition(hist_df: pd.DataFrame):
    df = hist_df.copy().set_index("date")
    vals = df["value"].astype(float)
    n = len(vals)
    if n < 8:
        st.info("Minimum 8 data points required for decomposition.")
        return

    window   = 12 if n >= 24 else max(3, n // 4)
    trend    = vals.rolling(window, center=True, min_periods=1).mean()
    residual = vals - trend

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.06,
        subplot_titles=["Actual + Trend", "Residuals", "Residual Distribution"],
    )
    fig.add_trace(go.Scatter(x=vals.index, y=vals.values, mode="lines",
        name="Actual", line=dict(color="#475569", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values, mode="lines",
        name=f"{window}-Period Trend", line=dict(color="#C8974A", width=2.5)), row=1, col=1)

    rc = ["#22c55e" if v >= 0 else "#ef4444" for v in residual.fillna(0)]
    fig.add_trace(go.Bar(x=residual.index, y=residual.values, name="Residual",
        marker_color=rc, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Histogram(x=residual.dropna().values, nbinsx=20,
        name="Residual Dist",
        marker=dict(color="#52B8B2", opacity=0.7)), row=3, col=1)

    layout = _layout(
        "Trend & Residual Decomposition",
        f"Rolling trend (window={window}) · Residuals = unexplained variation",
        height=520,
    )
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#1a2540", zeroline=False)
    fig.update_yaxes(gridcolor="#1a2540", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


# ── 06 Scenario ───────────────────────────────────────────────────────────────
def _scenario_overlay(hist_df: pd.DataFrame):
    baseline = st.session_state.sentinel_primary_df
    scenario = st.session_state.scenario_forecast_df
    params   = st.session_state.scenario_state.get("params", {})
    if baseline is None or scenario is None:
        return

    p_type  = params.get("type", "—")
    p_shock = params.get("shock_pct", 0)
    p_trend = params.get("trend_adjust_pct", 0)
    p_rec   = params.get("recovery_periods", 0)

    st.markdown(
        f'<div class="scenario-banner">'
        f'▲ SCENARIO ACTIVE — {p_type} · Shock {p_shock:+.1f}% · '
        f'Trend Adj {p_trend:+.1f}% · Recovery {p_rec}p · '
        f'Baseline is unchanged'
        f'</div>',
        unsafe_allow_html=True,
    )

    delta     = scenario.values - baseline.values
    direction = "upside" if float(delta.mean()) > 0 else "downside"
    up_col    = "#22c55e" if direction == "upside" else "#ef4444"
    fill_col  = ("rgba(34,197,94,0.07)" if direction == "upside"
                 else "rgba(239,68,68,0.07)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["value"],
        mode="lines", name="Actual",
        line=dict(color="#475569", width=1.5),
    ))
    # Delta fill
    fig.add_trace(go.Scatter(
        x=list(baseline.index) + list(scenario.index[::-1]),
        y=list(baseline.values) + list(scenario.values[::-1]),
        fill="toself", fillcolor=fill_col,
        line=dict(width=0), name="Delta Region", hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=baseline.index, y=baseline.values,
        mode="lines", name="Baseline Forecast",
        line=dict(color="#C8974A", width=3),
        hovertemplate="<b>Baseline</b> %{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=scenario.index, y=scenario.values,
        mode="lines", name=f"Scenario ({p_type})",
        line=dict(color=up_col, width=3, dash="dash"),
        hovertemplate="<b>Scenario</b> %{x|%b %Y}<br>%{y:,.2f}<extra></extra>",
    ))
    origin_str = hist_df["date"].max().strftime("%Y-%m-%d")
    fig.add_shape(type="line",
        x0=origin_str, x1=origin_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#1e2a45", width=1.5, dash="dash"),
    )
    fig.update_layout(**_layout(
        "Baseline vs Scenario Projection",
        f"{p_type} scenario · Direction: {direction.title()} · "
        f"Baseline is the certified forecast",
        height=440,
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Delta table
    delta_df = pd.DataFrame({
        "Period":   [d.strftime("%b %Y") for d in baseline.index],
        "Baseline": [_fmt(v) for v in baseline.values],
        "Scenario": [_fmt(v) for v in scenario.values],
        "Delta":    [_fmt(v) for v in delta],
        "Delta %":  [f"{v/b*100:+.2f}%" if b != 0 else "—"
                     for v, b in zip(delta, baseline.values)],
    })
    with st.expander("Period-by-Period Delta Table"):
        st.dataframe(delta_df, use_container_width=True, hide_index=True)
