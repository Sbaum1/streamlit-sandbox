# ==================================================
# FILE: forecastiq/tabs/help.py
# VERSION: 1.0.0
# ROLE: EXECUTIVE HELP & REFERENCE GUIDE
# Full platform documentation for executives and
# analysts. Covers every control, metric, chart,
# and workflow in VEDUTA.
# ==================================================

import streamlit as st

_CSS = """
<style>
.help-hero {
    background: linear-gradient(135deg, #1B2A40 0%, #152033 100%);
    border: 1px solid #243347;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.help-section {
    background: #1B2A40;
    border: 1px solid #243347;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
}
.help-section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 400;
    color: #EDE8DE;
    margin-bottom: .75rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid #243347;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.help-label {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #C8974A;
    margin: 1rem 0 .2rem;
}
.help-body {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: #8FA3B8;
    line-height: 1.7;
}
.help-callout {
    background: rgba(200,151,74,.08);
    border-left: 3px solid #C8974A;
    border-radius: 0 6px 6px 0;
    padding: .6rem .9rem;
    margin: .75rem 0;
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    color: #E2B96A;
    line-height: 1.6;
}
.help-warn {
    background: rgba(212,131,74,.07);
    border-left: 3px solid #D4834A;
    border-radius: 0 6px 6px 0;
    padding: .6rem .9rem;
    margin: .75rem 0;
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    color: #D4834A;
    line-height: 1.6;
}
.tier-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .75rem;
    margin: .75rem 0;
}
.tier-card {
    background: #07080F;
    border: 1px solid #243347;
    border-radius: 8px;
    padding: .75rem;
    text-align: center;
}
.tier-card .tier-emoji { font-size: 1.2rem; }
.tier-card .tier-name {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    font-weight: 600;
    margin: .25rem 0 .1rem;
}
.tier-card .tier-range {
    font-family: 'DM Mono', monospace;
    font-size: .6rem;
    color: #4A6278;
}
.metric-row {
    display: flex;
    gap: 1rem;
    margin: .5rem 0;
    align-items: flex-start;
}
.metric-name {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    font-weight: 600;
    color: #C8974A;
    min-width: 120px;
    letter-spacing: .06em;
}
.metric-desc {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    color: #8FA3B8;
    line-height: 1.6;
}
.shortcut-row {
    display: flex;
    justify-content: space-between;
    padding: .3rem 0;
    border-bottom: 1px solid #1B2A40;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
}
.shortcut-key {
    background: #243347;
    border-radius: 4px;
    padding: .1rem .4rem;
    color: #EDE8DE;
    font-size: .62rem;
}
.shortcut-desc { color: #4A6278; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
"""

def render_help():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="help-hero">
      <div style="font-size:.55rem;letter-spacing:.2em;text-transform:uppercase;
                  color:#C8974A;margin-bottom:.4rem;font-family:'DM Mono',monospace">
        VEDUTA · Executive Intelligence · Help & Reference
      </div>
      <div style="font-family:'Cormorant Garamond',serif;font-size:1.8rem;font-weight:300;
                  color:#EDE8DE;line-height:1.2;margin-bottom:.6rem;letter-spacing:0.04em">
        Platform Guide
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:.75rem;color:#4A6278;
                  line-height:1.7">
        Complete reference for every control, metric, chart, and workflow in VEDUTA.
        Built for executives and analysts working with enterprise time series forecasting.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search ────────────────────────────────────────────────────────────────
    search = st.text_input(
        "",
        placeholder="Search help topics… (e.g. MASE, scenario, FRED, CI)",
        label_visibility="collapsed",
    )

    # ── Navigation tabs ───────────────────────────────────────────────────────
    sections = [
        "Getting Started",
        "Data Intake",
        "Forecast Controls",
        "Reading Results",
        "Model Certification",
        "Scenario Planning",
        "Macro Variables",
        "Auto-Intelligence",
        "Metrics Glossary",
        "Troubleshooting",
    ]

    nav = st.tabs(sections)

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. GETTING STARTED
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[0]:
        _section("◎ What is VEDUTA?", """
VEDUTA is an enterprise-grade time series forecasting platform powered by the
Foresight Engine — a 19-model ensemble that runs statistical, machine learning, and
hybrid forecasting methods simultaneously, certifies each model against rigorous
M-Competition benchmarks, and delivers a single certified Primary Ensemble forecast
with full confidence intervals.

The platform is designed for C-suite executives who need trustworthy forward projections
for planning, scenario analysis, and board-level reporting — without requiring data
science expertise to operate.
        """)

        _section("◎ Recommended Workflow", """
Follow this sequence for every forecasting session:

1. **Commit Data** — Upload or paste your time series on the Home tab. Click SAVE / COMMIT DATA.
2. **Review Data Intelligence** — Verify the Actual Series, Rolling Analytics, Seasonality,
   and Distribution charts before running any forecast.
3. **Configure the Engine** — Set Model Tier, Horizon, Backtest Window, and CI Level in the sidebar.
4. **Run Forecast** — Click RUN FORECAST. The Foresight Engine runs all models simultaneously.
5. **Read the Hero Chart** — Review the Forecast Trajectory with CI bands.
6. **Check Model Certification** — Review the Model Scorecard to understand which models
   earned Elite, Strong, Pass, or Fail certification.
7. **Apply Scenarios** — Use Scenario Simulation or Macro Multipliers to stress-test assumptions.
8. **Enable Auto-Intelligence** — Toggle Auto-Intelligence for AI-generated executive briefings
   on each section.
9. **Export** — Use the Report Builder tab to generate board-ready PDF reports.
        """)

        _callout("The Primary Ensemble is your certified baseline. It is always the "
                 "recommended output for planning and reporting. Individual models are "
                 "available for analysis but should not replace the ensemble.")

        _section("◎ Platform Architecture", """
**Tabs:**
- **Home** — Data intake, commitment, Data Intelligence visuals, Forecast Trajectory,
  KPI strip, Model Intelligence, Scenario Stress Test
- **Executive Insight & Trust** — Confidence scoring, model audit, risk flags,
  executive certification summary
- **Report Builder** — Automated board-ready PDF/export generation
- **Help** — This guide

**Sidebar zones:**
- System Status — live forecast and data state
- Auto-Intelligence — AI insight toggle
- Baseline Forecast — all engine controls
- Macro Variables — FRED regressors and multipliers
- Scenario Simulation — shock, trend, and recovery overlays
- Quick Reference — run signature summary
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. DATA INTAKE
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[1]:
        _section("◎ Data Requirements", """
VEDUTA expects a two-column time series: one date column and one numeric value column.

**Supported formats:**
- CSV (comma or tab separated, with or without headers)
- Excel (.xls, .xlsx)
- Paste directly into the Paste Data box (tab or comma separated)

**Date formats accepted:** YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YYYY, and most standard formats.
Pandas date parsing handles the majority of cases automatically.

**Minimum observations:** 36 periods for backtest certification. Shorter series will still
forecast but MASE scores will be marked ineligible.

**Frequency support:** Monthly (default), Weekly, Daily, Quarterly. Frequency is
auto-detected from your date column spacing.
        """)

        _warn("Data must be clean and consistent. Gaps, duplicates, and mixed frequencies "
              "will be flagged in the Data Profile strip. Address these before committing.")

        _section("◎ The Data Profile Strip", """
After loading data, the six-column profile strip shows:

- **ROWS** — Total observation count
- **START / END** — Date range of your series
- **FREQUENCY** — Inferred cadence (Monthly, Weekly, etc.) with confidence color:
  Green = high confidence (>85%), Amber = medium, Red = low (<70%)
- **DUPLICATES** — Count of duplicate date entries (should be 0)
- **GAPS** — Count of missing periods (should be 0)
        """)

        _section("◎ Frequency Override", """
If frequency is detected with low confidence (amber or red), verify your data before
committing. Common causes:

- Mixed month lengths causing 28-31 day variance (normal for monthly data — our
  4-strategy inferrer handles this)
- Irregular business day data mistaken for daily
- Quarterly data with end-of-quarter vs start-of-quarter dates

If auto-detection is wrong, standardize your date column to first-of-period format
(e.g. 2024-01-01 for January) before uploading.
        """)

        _section("◎ SAVE / COMMIT DATA", """
Clicking SAVE / COMMIT DATA locks the dataset as the certified baseline for this session.

What happens on commit:
- Data is fingerprinted (SHA-256 hash) for audit trail
- Frequency and confidence are stored to session state
- All downstream analysis is tied to this exact dataset
- The Data Intelligence suite (charts, seasonality, distribution) becomes available
- A commit entry is written to the audit log

**Important:** Running a new forecast after changing data requires re-committing first.
The run signature records which data fingerprint was used for every forecast.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. FORECAST CONTROLS
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[2]:
        _section("◎ Model Tier", """
The Model Tier controls which models the Foresight Engine runs.

- **Enterprise (19 models)** — Full suite: ARIMA, SARIMA, SARIMAX, ETS, STL+ETS, TBATS,
  BSTS, Prophet, Theta, Naive, and all ensemble variants. Recommended for production forecasts.
- **Pro (13 models)** — Balanced accuracy and speed. Removes experimental and
  computationally intensive models.
- **Essentials (10 models)** — Core statistical models only. Fastest run time.
  Suitable for quick exploratory analysis.

Higher tiers produce more robust Primary Ensemble weights because MASE scores are
computed across more diverse models.
        """)

        _section("◎ Forecast Horizon", """
The number of future periods the engine projects forward from the last historical observation.

- Monthly data: 12 = one year forward, 24 = two years forward
- The hero chart shows all forecast periods with CI bands
- Longer horizons increase forecast uncertainty — CI bands widen
- Recommended maximum: 2× the seasonal period (24 for monthly data)
        """)

        _section("◎ Backtest Window", """
The number of held-out periods used to evaluate each model's MASE score.

This is separate from Forecast Horizon. The backtest runs walk-forward folds against
historical data — the engine never sees the test periods during training.

- Minimum: 4 periods
- Recommended: 12 periods (one full seasonal cycle for monthly data)
- Maximum: 36 periods
- Larger windows = more reliable MASE scores but longer run time
        """)

        _section("◎ Backtest Strategy", """
Controls how training data is sized in each backtest fold.

- **Expanding Window (default)** — Each fold uses all available history up to the
  cutoff point. Training set grows with each fold. Mimics real-world deployment
  where you always have access to all historical data. Recommended.
- **Rolling Window** — Training set has a fixed size. Older data is dropped as
  folds advance. Tests the model's ability to adapt to recent patterns specifically.
  Useful when you suspect structural breaks in older data.
        """)

        _section("◎ Confidence Interval", """
The probability band shown around the Primary Ensemble forecast.

- **80%** — Narrow band. 80% of actual outcomes should fall within this range.
  Use for optimistic planning scenarios.
- **90%** — Moderate band. Good balance of precision and coverage.
- **95% (default)** — Standard statistical convention. Recommended for
  board-level reporting.
- **99%** — Wide band. Very conservative. Use when downside protection is
  critical (e.g. cash flow planning, inventory buffers).

Wider CI = more conservative = less precise but more reliable coverage guarantee.
        """)

        _section("◎ Ensemble Weights", """
Controls how individual model forecasts are combined into the Primary Ensemble.

- **MASE-Weighted (active)** — Each model's weight is proportional to its
  inverse backtest MASE. Models with lower error get more influence.
  This is the statistically rigorous default and the recommended choice.
- **Equal Weights (planned)** — All models contribute equally regardless of
  performance. Will be available in a future engine release.

The weight bridge runs automatically after all models are scored. You can see
which models received median-fallback weights in the run metadata.
        """)

        _section("◎ Outlier Sensitivity", """
Pre-processing applied to your data before the engine runs.

- **None** — Raw data passed through unchanged. Use if you trust your data
  completely or have already cleaned it.
- **Low** — Identifies outliers (values beyond ±3σ) and flags them in the
  run log but does not modify the data.
- **Medium (default)** — Winsorises at ±3 standard deviations. Caps extreme
  values without removing observations. Recommended for most datasets.
- **High** — Winsorises at ±2 standard deviations. More aggressive capping.
  Use when your series has known extreme spikes that are not meaningful.

The number of capped observations and their bounds are recorded in the run signature.
        """)

        _section("◎ Analyst Mode", """
Enables extended diagnostic output in all tab views. When active:
- Additional model metadata columns appear in the scorecard
- Fold-level backtest detail becomes accessible
- Trend decomposition shows extended residual analysis

Stored in the run signature so reports can be annotated with whether
Analyst Mode was active during the forecast run.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. READING RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[3]:
        _section("◎ The Forecast Trajectory (Hero Chart)", """
The main forecast visualization. Five overlaid layers from bottom to top:

1. **95% CI Band** (shaded blue) — Confidence interval around the Primary Ensemble
2. **Actuals** (gray line) — Your historical data
3. **Primary Ensemble** (solid blue, width 3) — The certified forecast baseline
4. **Stacked Ensemble** (dotted purple, width 2) — Ridge meta-learner alternative
5. **Forecast Origin Line** (dashed) — Where history ends and forecast begins

**Interactions:**
- Drag to zoom into any region
- Double-click to reset to full view
- Hover to see unified tooltip with values across all series
- Click legend items to show/hide individual traces
        """)

        _callout("Use the Primary Ensemble for all planning and reporting. The Stacked "
                 "Ensemble is shown for comparison only. When they diverge significantly, "
                 "it may signal uncertainty in the forecast direction.")

        _section("◎ KPI Command Strip", """
Four tiles below the hero chart providing immediate executive context:

- **Latest Actual** — Most recent observed value with month-over-month delta arrow
- **Next Period Forecast** — Primary Ensemble prediction for the next period
- **Primary Ensemble MASE** — Certified accuracy score (see Metrics Glossary)
- **Forecast Horizon + Tier** — Active configuration summary

Delta arrows: ▲ green = positive, ▼ red = negative, ● gray = flat/neutral
        """)

        _section("◎ Data Intelligence Tabs", """
Available immediately after data commit — before any forecast is run.

**Actual Series** — Raw time series with fill shading. Use to identify trends,
cycles, and structural breaks.

**Rolling Analytics** — Two-panel view:
- Upper: Actual + 3-period MA + 12-period MA overlaid
- Lower: Month-over-month % change bars (green/red)
Useful for identifying trend acceleration or deceleration.

**Seasonality** — Year-over-year line overlay (each year in a distinct color)
plus a year × month heatmap. Identifies stable vs shifting seasonal patterns.
If the same months consistently spike or trough, that is actionable seasonality.

**Distribution** — Histogram (30 bins) and box plot showing value spread.
Stats strip below: Mean, Median, Std Dev, Min, Max, IQR.
Wide distribution with heavy tails = harder to forecast precisely.
        """)

        _section("◎ All-Model Overlay", """
The second tab in Model Intelligence shows every model's forecast simultaneously.
Primary Ensemble and Stacked Ensemble are visible by default; individual models
are hidden (legendonly) to avoid visual clutter.

Click any model name in the legend to show/hide it.

Use this view to:
- Check whether models broadly agree (tight cluster = confident forecast)
- Identify outlier model predictions
- Understand the range of expert opinion across the ensemble
        """)

        _section("◎ Trend Decomposition", """
Three-panel chart in the Trend Decomposition tab:

- **Top panel** — Actual series + trend line (rolling mean)
- **Middle panel** — Residuals (actual minus trend) as bars
- **Bottom panel** — Residual distribution histogram

Flat residuals with no pattern = good model fit.
Structured residuals = unmodeled seasonal or cyclical component.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. MODEL CERTIFICATION
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[4]:
        _section("◎ What is Certification?", """
VEDUTA certifies every model against the M-Competition benchmark standard —
the same framework used by academic forecasting competitions to rank methods.

Certification answers: "Does this model perform better than a naive baseline
on your specific data?" A model that cannot beat a simple last-value-forward
prediction (MASE ≥ 1.0) fails certification regardless of its sophistication.

This prevents overconfident forecasts from entering the ensemble with high weight.
        """)

        st.markdown("""
        <div class="tier-grid">
          <div class="tier-card" style="border-color:#22c55e33">
            <div class="tier-emoji">🟢</div>
            <div class="tier-name" style="color:#22c55e">Elite</div>
            <div class="tier-range">MASE &lt; 0.70</div>
          </div>
          <div class="tier-card" style="border-color:#f59e0b33">
            <div class="tier-emoji">🟡</div>
            <div class="tier-name" style="color:#f59e0b">Strong</div>
            <div class="tier-range">MASE 0.70–0.85</div>
          </div>
          <div class="tier-card" style="border-color:#f9731633">
            <div class="tier-emoji">🟠</div>
            <div class="tier-name" style="color:#f97316">Pass</div>
            <div class="tier-range">MASE 0.85–1.00</div>
          </div>
          <div class="tier-card" style="border-color:#ef444433">
            <div class="tier-emoji">🔴</div>
            <div class="tier-name" style="color:#ef4444">Fail</div>
            <div class="tier-range">MASE ≥ 1.00</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        _section("◎ What Does Every Model Failing Mean?", """
If all models show Fail (MASE ≥ 1.00), your data is harder to forecast than a
naive baseline. This is not a bug — it is an honest assessment.

Common causes:

**High volatility with structural breaks** — Data from crisis periods (2008-2020
financial data, COVID-era revenue) often has regime changes that make historical
patterns unreliable guides to the future. No statistical model can reliably beat
naive on such data.

**Too-short backtest window** — With only 12 periods to evaluate against, a few
bad predictions drive MASE above 1.0 even for reasonable models.

**Seasonal data with irregular shocks** — If your series has strong, consistent
seasonality interrupted by one-off spikes, MASE calculation penalizes the spikes heavily.

**What to do:**
- Increase the Backtest Window to 24 periods
- Switch to Essentials tier (faster models, less overfitting)
- Review the Distribution tab — very high IQR indicates fundamental forecast difficulty
- The Primary Ensemble forecast is still usable for directional planning even with
  Fail-tier MASE scores. The certification reflects backtested accuracy, not future validity.
        """)

        _warn("All models failing does not mean the forecast is wrong. It means "
              "historical accuracy was below the naive baseline threshold. Use the "
              "forecast for directional guidance and apply wider CI bands (99%).")

        _section("◎ Primary vs Stacked Ensemble", """
**Primary Ensemble** — MASE-weighted mean of all certified member models.
Lower-error models get higher influence. This is the certified baseline.

**Stacked Ensemble** — Ridge regression meta-learner trained on out-of-fold predictions.
Learns the optimal linear combination of member forecasts. Can outperform the Primary
Ensemble on complex patterns but is more prone to overfitting on short series.

When they agree closely: high confidence in the forecast direction.
When they diverge: treat the gap as the uncertainty range and use wider CI bands.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. SCENARIO PLANNING
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[5]:
        _section("◎ How Scenarios Work", """
Scenarios apply a structured overlay to the Primary Ensemble baseline forecast.
The baseline is never modified — scenarios are always additive overlays visible
on a separate chart layer with a clear "Scenario Active" banner.

This design allows you to present multiple futures to the board without
compromising the certified baseline integrity.
        """)

        _section("◎ Scenario Types", """
**Shock** — Immediate percentage impact applied uniformly across all forecast periods.
Use for: sudden demand shocks, policy changes, one-time events.
Example: -15% shock for a major customer loss.

**Trend Shift** — Progressive growth or decline applied incrementally each period.
Use for: gradual market share erosion, sustained expansion, demographic trends.
Example: -0.5% per period for secular demand decline.

**Ramp Recovery** — Initial shock with gradual recovery toward baseline over N periods.
Use for: supply chain disruptions, temporary market exits, seasonal dislocations.
Example: -20% shock recovering over 6 periods.

**Shock + Recovery** — Sharp immediate impact followed by a defined recovery arc.
Use for: recession scenarios, natural disasters, product recalls.
Example: -30% shock recovering over 12 periods.
        """)

        _section("◎ Scenario Controls", """
- **Shock (%)** — Immediate percentage impact. Negative = downside. Positive = upside.
- **Trend Adjustment (%)** — Per-period drift rate applied progressively.
- **Recovery Periods** — How many periods before the forecast returns to baseline.
  Set to 0 for permanent shift scenarios.

All three parameters interact. A Shock + Recovery scenario uses Shock (%) for the
initial drop and Recovery Periods for the return arc.
        """)

        _callout("Best practice: Run your board meeting with three scenarios — "
                 "Base (no scenario), Bear (-10% to -25% shock), and Bull (+5% to +15%). "
                 "This gives leadership a decision range rather than a single number.")

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. MACRO VARIABLES
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[6]:
        _section("◎ Two Approaches to Macro Variables", """
VEDUTA supports two distinct methods for incorporating macroeconomic conditions
into your forecast. They serve different purposes and can be used together.

**Tier A — FRED Regressors** (requires API key + engine update)
Pulls live data from the Federal Reserve Economic Database and uses it as
exogenous input features in statistical models. This is statistically rigorous —
the macro variable actually influences the model's forecast math.
Requires: free FRED API key + foresight engine exog support.

**Tier B — Macro Multipliers** (fully active today)
You specify assumed future changes in macro variables (e.g. +1.5% interest rates).
VEDUTA applies calibrated economic multipliers to translate each assumption
into a % shock on the baseline forecast. Fast, transparent, assumption-driven.
No API key required.
        """)

        _section("◎ Tier A — FRED API Setup", """
1. Register for a free API key at **fred.stlouisfed.org/docs/api/api_key.html**
2. Paste your key into the FRED API Key field in the sidebar
3. Select series from the catalogue (25 series across 8 categories)
4. Click **Fetch FRED Data** — data is pulled, resampled to your frequency,
   and normalised to zero-mean unit-variance
5. On the next RUN FORECAST, the fetched data will be available as exogenous regressors

**Current status:** FRED fetch is live. Injection into model training requires a
foresight engine update to accept exog_df as a parameter in run_all_models().
This is on the engineering roadmap.

**Available categories:** Interest Rates, Labor Market, Inflation & Prices,
Growth & Activity, Credit, Housing, FX, Commodities.
        """)

        _section("◎ Tier B — Macro Multiplier Reference", """
Each variable has a calibrated multiplier based on empirical economic relationships.
The multiplier is the % change in your forecast per 1-unit change in the macro variable.

These are general-purpose estimates. Your specific series may respond differently.
        """)

        _multiplier_table()

        _section("◎ Net Macro Impact Calculator", """
The Net Macro Impact display shows the combined effect of all active multipliers in real time.

Formula: For each variable, impact = shock_value × multiplier / 100
Total impact = sum of all variable impacts

Example:
- Fed Funds Rate: +1.5 shock × -2.5 multiplier = -3.75%
- Consumer Sentiment: +8 shock × +1.2 multiplier = +0.096%
- Net: approximately -3.65%

Click **Apply to Forecast** to overlay this as a scenario on the Home tab hero chart.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. AUTO-INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[7]:
        _section("◎ What is Auto-Intelligence?", """
Auto-Intelligence connects VEDUTA to Claude (Anthropic's AI) to generate
plain-English executive briefings below each section of the Home tab.

Unlike generic AI descriptions, Auto-Intelligence reads your actual numbers —
the specific MASE scores, forecast values, seasonal patterns, and model distribution
from your current run — and interprets them in context.

**Where briefings appear (when enabled):**
- Below the Forecast Trajectory chart
- Below the KPI Command Strip
- Below the Model Scorecard
- Below the Seasonality chart
        """)

        _section("◎ How to Use It", """
1. Toggle **Enable Auto-Intelligence** in the sidebar Zone 2
2. Run a forecast (or use an existing one)
3. Purple briefing panels appear below each relevant section
4. Each insight is specific to your live data — not a template

**Cache behavior:** Insights are cached per run. Running a new forecast automatically
clears the cache so insights regenerate from your new results.
Clicking **Clear Insight Cache** in the sidebar forces all sections to regenerate.

**What to expect:** 2-4 sentences per section. Direct, executive-grade language.
References your actual numbers. Leads with the most decision-relevant finding.
        """)

        _callout("Auto-Intelligence is powered by claude-sonnet-4-20250514. Insights "
                 "are generated fresh from your data but should be reviewed before "
                 "inclusion in formal board reports. Use them as a starting point "
                 "for your own narrative, not as final copy.")

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. METRICS GLOSSARY
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[8]:
        st.markdown("""
        <div class="help-section">
          <div class="help-section-title">◎ Forecast Accuracy Metrics</div>
        """, unsafe_allow_html=True)

        metrics = [
            ("MASE", "Mean Absolute Scaled Error. The primary certification metric. "
             "MASE = model MAE ÷ naive forecast MAE. "
             "MASE < 1.0 = beats naive. MASE < 0.70 = Elite tier. "
             "This is scale-independent, making it valid for comparison across series."),
            ("MAE", "Mean Absolute Error. Average absolute difference between "
             "forecast and actual. In your data's units (e.g. dollars, units). "
             "Easy to interpret but scale-dependent."),
            ("RMSE", "Root Mean Squared Error. Like MAE but squares errors first, "
             "penalizing large deviations more heavily. More sensitive to outliers."),
            ("MAPE", "Mean Absolute Percentage Error. Error as a % of actual value. "
             "Useful for communication but undefined when actuals are near zero."),
            ("SMAPE", "Symmetric MAPE. Addresses MAPE asymmetry around zero. "
             "Bounded between 0% and 200%."),
            ("Theil's U", "Ratio of forecast error to naive forecast error. "
             "U < 1 = better than naive. Similar concept to MASE but quadratic."),
            ("CI Coverage", "Percentage of actual values that fell inside the "
             "confidence interval during backtesting. A 95% CI should cover ~95% "
             "of actuals. Significant deviation indicates miscalibrated intervals."),
            ("Directional Accuracy", "Percentage of periods where the forecast "
             "correctly predicted the direction of change (up vs down). "
             "Random guessing = 50%. Good models target 60-75%+."),
        ]

        for name, desc in metrics:
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-name">{name}</div>
              <div class="metric-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        _section("◎ Chart Elements", """
**Confidence Interval Band** — Shaded region around forecast. Width controlled by CI Level setting.
95% CI means: if the model is well-calibrated, 95% of future actuals will fall in this range.

**Forecast Origin Line** — Dashed vertical line marking where historical data ends.
Everything to the right is model projection.

**Rolling MA** — Moving average over N periods. Smooths noise to reveal underlying trend.
3-period MA is more reactive; 12-period MA shows longer-cycle trends.

**MoM % Bars** — Month-over-month percentage change. Green = growth, Red = decline.
Consecutive red bars = sustained contraction. Alternating = volatile.

**Year-over-Year Heatmap** — Grid of years (rows) × months (columns), colored by value intensity.
Consistent dark columns = strong seasonal months. Shifting patterns = seasonality evolution.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # 10. TROUBLESHOOTING
    # ═══════════════════════════════════════════════════════════════════════════
    with nav[9]:
        _section("◎ Common Issues & Fixes", "")

        issues = [
            ("Frequency shows 'Unknown' or low confidence",
             "Your date column may have irregular spacing or mixed formats. "
             "Standardize to YYYY-MM-DD first-of-period format (e.g. 2024-01-01 for January). "
             "Re-upload and re-commit."),
            ("All models fail certification (MASE ≥ 1.0)",
             "Your data may have high volatility or structural breaks. Try: "
             "(1) Increase Backtest Window to 24 periods. "
             "(2) Switch to Essentials tier. "
             "(3) Review Distribution tab — if IQR is very high, the series is inherently hard to forecast. "
             "The forecast is still usable for directional planning."),
            ("Data table doesn't appear after committing",
             "Hard-restart Streamlit (Ctrl+C, rerun app.py). Browser refresh alone "
             "may not reload the Python module."),
            ("Scenario overlay not showing",
             "Ensure a forecast has been run first. The scenario section only unlocks "
             "after sentinel_primary_df is populated. Check that APPLY was clicked "
             "and the success message appeared."),
            ("Auto-Intelligence shows 'unavailable'",
             "Check your network connection. The Anthropic API requires outbound HTTPS. "
             "If running in a restricted environment, Auto-Intelligence will not be available."),
            ("Tab labels appear blank",
             "Streamlit's CSS selector for tab labels varies by version. "
             "The platform targets button[data-baseweb='tab'] selectors. "
             "If blank, your Streamlit version may use different DOM structure. "
             "Upgrade to Streamlit >= 1.28."),
            ("Forecast runs very slowly",
             "Enterprise tier runs 19 models with full backtesting. For faster results: "
             "(1) Switch to Essentials tier. "
             "(2) Reduce Backtest Window to 8 periods. "
             "(3) Reduce Forecast Horizon."),
            ("FRED Fetch returns errors",
             "Verify your API key is correct. FRED API keys are free but must be "
             "registered. Rate limits apply — wait 30 seconds between fetch attempts. "
             "Some series may not have monthly data available for your date range."),
        ]

        for title, body in issues:
            with st.expander(title):
                st.markdown(
                    f'<div class="help-body">{body}</div>',
                    unsafe_allow_html=True,
                )

        _section("◎ Run Signature Audit Trail", """
Every forecast run generates a run signature stored in session state and the audit log.

The run signature records:
- Run ID and timestamp
- Data fingerprint (SHA-256 hash of committed dataset)
- All engine configuration parameters (tier, horizon, backtest, CI, strategy, weights, outliers)
- Engine version
- Winner model
- Active macro configuration

This allows you to reconstruct the exact conditions of any forecast for audit,
compliance, or board reporting purposes. The Quick Reference zone in the sidebar
shows the most recent run signature at a glance.
        """)

        _section("◎ Contact & Support", """
VEDUTA is powered by the Foresight Engine v2.0.0.

For platform issues, engine updates, or feature requests — consult your platform
engineer or administrator. The foresight engine source files are located at the
repository root. The VEDUTA application layer is in the veduta/ directory.

**File structure:**
- veduta/app.py — Main Streamlit entry point
- veduta/tabs/ — All tab render functions
- veduta/sidebar/ — Veduta Overview
- veduta/engine/ — Foresight adapter + macro variables
- veduta/state/ — Session state management
- veduta/utils/ — Frequency inference, validation, hashing
        """)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str, body: str):
    icon = title[:2]
    label = title[2:].strip()
    st.markdown(f"""
    <div class="help-section">
      <div class="help-section-title">{icon} {label}</div>
      <div class="help-body">{body.strip()}</div>
    </div>
    """, unsafe_allow_html=True)


def _callout(text: str):
    st.markdown(f'<div class="help-callout">{text}</div>', unsafe_allow_html=True)


def _warn(text: str):
    st.markdown(f'<div class="help-warn">{text}</div>', unsafe_allow_html=True)


def _multiplier_table():
    rows = [
        ("Federal Funds Rate",  "−2.5% per +1%",  "Rate hike suppresses borrowing and demand"),
        ("Unemployment Rate",   "−3.0% per +1pp", "Rising unemployment reduces consumer activity"),
        ("CPI Inflation",       "−1.5% per +1%",  "Real spending power erosion"),
        ("Credit Spread (HY)",  "−1.8% per +100bp","Credit tightening reduces business investment"),
        ("GDP Growth",          "+2.0% per +1pp", "Macro expansion lifts activity broadly"),
        ("Consumer Sentiment",  "+1.2% per +10pt","Confidence translates to spending"),
        ("Oil Price (WTI)",     "−0.8% per +$10", "Input cost pressure on margins and activity"),
        ("USD Strength Index",  "−1.0% per +5%",  "USD appreciation headwind for exports"),
    ]
    html = """
    <div style="margin:.5rem 0">
    <div style="display:grid;grid-template-columns:1.8fr 1.2fr 2fr;gap:.3rem;
                font-family:'DM Mono',monospace;font-size:.6rem;
                color:#4A6278;letter-spacing:.08em;text-transform:uppercase;
                padding:.3rem 0;border-bottom:1px solid #243347;margin-bottom:.3rem">
      <div>Variable</div><div>Multiplier</div><div>Rationale</div>
    </div>
    """
    for var, mult, rationale in rows:
        html += f"""
    <div style="display:grid;grid-template-columns:1.8fr 1.2fr 2fr;gap:.3rem;
                font-family:'DM Mono',monospace;font-size:.68rem;
                padding:.35rem 0;border-bottom:1px solid #0f1929;align-items:start">
      <div style="color:#EDE8DE">{var}</div>
      <div style="color:#60a5fa;font-weight:600">{mult}</div>
      <div style="color:#4A6278">{rationale}</div>
    </div>
    """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
