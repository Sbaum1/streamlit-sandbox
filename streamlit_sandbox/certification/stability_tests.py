# ==================================================
# FILE: streamlit_sandbox/certification/stability_tests.py
# ROLE: STABILITY TESTS (STL+ETS DIAGNOSTIC EXTENSION)
# STANDARD: FORTUNE 100 / ZERO REGRESSION
# ==================================================

from __future__ import annotations

import json
import os
import numpy as np

from streamlit_sandbox.execution.engine import run_all_models, MODEL_REGISTRY
from streamlit_sandbox.certification.dataset_factory import seasonal_dataset

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
STABILITY_ARTIFACT = os.path.join(ARTIFACT_DIR, "stability_diagnostics.json")

VARIANCE_THRESHOLD = 1e-2


def stability_test():

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = seasonal_dataset()

    base_results = run_all_models(df, horizon=6, confidence_level=0.95)
    noisy_df = df.copy()
    noisy_df["value"] = noisy_df["value"] * 1.02
    noisy_results = run_all_models(noisy_df, horizon=6, confidence_level=0.95)

    diagnostics = []
    variance_metrics = {}
    excluded_models = []
    overall_status = "PASS"

    for model_name in MODEL_REGISTRY.keys():

        base = base_results.get(model_name)
        noisy = noisy_results.get(model_name)

        if not isinstance(base, dict) or not isinstance(noisy, dict):
            excluded_models.append(model_name)
            continue

        if base.get("status") != "success" or noisy.get("status") != "success":
            excluded_models.append(model_name)
            continue

        base_df = base.get("forecast_df")
        noisy_df_model = noisy.get("forecast_df")

        if base_df is None or noisy_df_model is None:
            excluded_models.append(model_name)
            continue

        horizon = 6
        base_future = base_df["forecast"].values[-horizon:]
        noisy_future = noisy_df_model["forecast"].values[-horizon:]

        if len(base_future) != horizon or len(noisy_future) != horizon:
            excluded_models.append(model_name)
            continue

        diff = noisy_future - base_future
        variance_metric_future = float(np.var(diff))
        variance_metrics[model_name] = variance_metric_future

        if variance_metric_future > VARIANCE_THRESHOLD:
            overall_status = "FAIL"

        model_diag = {
            "model_name": model_name,
            "status": "PASS" if variance_metric_future <= VARIANCE_THRESHOLD else "FAIL",
            "variance_metric_future": round(variance_metric_future, 10),
        }

        if model_name == "STL+ETS":
            base_meta = base.get("metadata", {})
            noisy_meta = noisy.get("metadata", {})

            clean_base = np.array(base_meta.get("base_future", []), dtype="float64")
            shocked_base = np.array(noisy_meta.get("base_future", []), dtype="float64")

            clean_trend = np.array(base_meta.get("future_trend", []), dtype="float64")
            shocked_trend = np.array(noisy_meta.get("future_trend", []), dtype="float64")

            clean_seasonal = np.array(base_meta.get("seasonal_future", []), dtype="float64")
            shocked_seasonal = np.array(noisy_meta.get("seasonal_future", []), dtype="float64")

            total_diff = shocked_base - clean_base
            trend_diff = shocked_trend - clean_trend
            seasonal_diff = shocked_seasonal - clean_seasonal

            var_future_total = float(round(np.var(total_diff, ddof=0), 10))
            var_future_trend = float(round(np.var(trend_diff, ddof=0), 10))
            var_future_seasonal = float(round(np.var(seasonal_diff, ddof=0), 10))

            if var_future_total == 0:
                amplification_ratio_trend = 0.0
                amplification_ratio_seasonal = 0.0
            else:
                amplification_ratio_trend = float(round(var_future_trend / var_future_total, 10))
                amplification_ratio_seasonal = float(round(var_future_seasonal / var_future_total, 10))

            model_diag.update({
                "var_future_total": var_future_total,
                "var_future_trend": var_future_trend,
                "var_future_seasonal": var_future_seasonal,
                "amplification_ratio_trend": amplification_ratio_trend,
                "amplification_ratio_seasonal": amplification_ratio_seasonal,
            })

        diagnostics.append(model_diag)

    artifact_payload = {
        "dataset": "seasonal_dataset",
        "test": "stability",
        "models": diagnostics,
    }

    with open(STABILITY_ARTIFACT, "w") as f:
        json.dump(artifact_payload, f, indent=2, sort_keys=True)

    return {
        "status": overall_status,
        "variance_metrics": variance_metrics,
        "excluded_models": excluded_models,
    }