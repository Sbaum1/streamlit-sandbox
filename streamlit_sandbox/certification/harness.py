# FILE: streamlit_sandbox/certification/harness.py
# ROLE: MODEL CERTIFICATION HARNESS (CLI / OFFLINE)
# STATUS: CANONICAL / EXECUTIVE-GRADE
# ==================================================
#
# ALIGNMENT FIX:
# - Explicit future-block extraction using horizon
# - Prevents misaligned evaluation for SARIMA/SARIMAX
# - No model math changed
# ==================================================

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import pandas as pd
import numpy as np

from streamlit_sandbox.models.registry import get_model_registry
from streamlit_sandbox.certification.windows import rolling_windows
from streamlit_sandbox.certification.metrics_ext import smape, beat_naive
from streamlit_sandbox.certification.regimes import classify_regime
from streamlit_sandbox.certification.logging import append_log
from streamlit_sandbox.certification.scoring import calibrate_confidence


DEBUG_LOG_PATH = "logs/certification_debug.log"

try:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=DEBUG_LOG_PATH,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
except Exception:
    logging.getLogger().addHandler(logging.NullHandler())


def _safe_log(level: str, message: str):
    try:
        getattr(logging, level)(message)
    except Exception:
        pass


def _safe_json_row(row: dict) -> dict:
    safe = {}
    for k, v in row.items():
        if isinstance(v, (pd.Timestamp, np.generic)):
            safe[k] = str(v)
        else:
            safe[k] = v
    return safe


def certify_models(
    df: pd.DataFrame,
    horizons=(3, 6, 12),
    log_path: str = "logs/model_certification.jsonl",
) -> dict:

    results: list[dict] = []
    scorecard: list[dict] = []

    registry = get_model_registry()
    timestamp = datetime.now(timezone.utc).isoformat()

    _safe_log("info", f"CERTIFICATION START | rows={len(df)}")

    for model in registry:
        model_name = model["name"]
        runner = model["runner"]

        model_rows: list[dict] = []

        _safe_log("info", f"MODEL START | model={model_name}")

        try:
            for horizon in horizons:
                for idx, (train_df, test_df) in enumerate(
                    rolling_windows(df=df, horizon=horizon, min_train=24)
                ):

                    # -----------------------------
                    # MODEL RUN
                    # -----------------------------
                    output = runner(
                        df=train_df,
                        horizon=horizon,
                        confidence_level=0.8,
                    )

                    forecast_df = getattr(output, "forecast_df", None)

                    if (
                        forecast_df is None
                        or "forecast" not in forecast_df
                        or test_df is None
                        or test_df.empty
                    ):
                        continue

                    # -----------------------------
                    # STRICT FUTURE BLOCK EXTRACTION
                    # -----------------------------
                    future_block = forecast_df.iloc[-horizon:]

                    if len(future_block) != len(test_df):
                        _safe_log(
                            "debug",
                            f"ALIGNMENT SKIP | model={model_name} | idx={idx}",
                        )
                        continue

                    actual = test_df["value"].to_numpy(dtype=float)
                    forecast = future_block["forecast"].to_numpy(dtype=float)

                    if actual.size == 0 or forecast.size == 0:
                        continue

                    # -----------------------------
                    # METRICS
                    # -----------------------------
                    errors = forecast - actual
                    mae = float(np.mean(np.abs(errors)))
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                    smape_val = float(
                        smape(pd.Series(actual), pd.Series(forecast))
                    )

                    last_value = float(train_df["value"].iloc[-1])
                    naive_forecast = np.full_like(actual, last_value)
                    naive_mae = float(np.mean(np.abs(naive_forecast - actual)))

                    beat_naive_flag = beat_naive(
                        model_mae=mae,
                        naive_mae=naive_mae,
                    )

                    row = {
                        "timestamp": timestamp,
                        "model": model_name,
                        "horizon": horizon,
                        "window_end": train_df["date"].iloc[-1],
                        "mae": mae,
                        "rmse": rmse,
                        "smape": smape_val,
                        "beat_naive": bool(beat_naive_flag),
                    }

                    append_log(_safe_json_row(row), log_path)
                    model_rows.append(row)

            windows_tested = len(model_rows)

            beat_naive_count = (
                sum(1 for r in model_rows if r["beat_naive"])
                if windows_tested > 0
                else 0
            )

            smape_vals = [
                r["smape"] for r in model_rows if not np.isnan(r["smape"])
            ]

            stability_score = (
                float(pd.Series(smape_vals).std())
                if len(smape_vals) >= 3
                else None
            )

            confidence = calibrate_confidence(
                beat_naive_count / windows_tested if windows_tested else 0.0,
                stability_score,
                False,
            )

            scorecard.append(
                {
                    "model": model_name,
                    "confidence": confidence,
                    "windows_tested": windows_tested,
                }
            )

            results.extend(model_rows)

        except Exception as e:
            _safe_log(
                "error",
                f"MODEL FAIL | model={model_name} | error={e}",
            )

    _safe_log("info", "CERTIFICATION COMPLETE")

    return {
        "scorecard": pd.DataFrame(scorecard),
        "raw_results": pd.DataFrame(results),
    }
