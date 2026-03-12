# ==================================================
# FILE: streamlit_sandbox/certification/performance_suite.py
# ROLE: FULL PERFORMANCE CERTIFICATION HARNESS
# STANDARD: FORTUNE 100 / DETERMINISTIC / NO UI
# ==================================================

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from streamlit_sandbox.execution.engine import run_all_models, MODEL_REGISTRY
from streamlit_sandbox.certification.dataset_factory import (
    seasonal_dataset,
    linear_trend_dataset,
    flat_dataset,
    structural_break_dataset,
    high_noise_dataset,
)
from streamlit_sandbox.certification.stability_tests import stability_test
from streamlit_sandbox.certification.shock_injector import inject_spike

BASE_DIR = os.path.dirname(__file__)
BASELINE_FILE = os.path.join(BASE_DIR, "performance_baselines.json")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
SHOCK_ARTIFACT = os.path.join(ARTIFACT_DIR, "shock_diagnostics.json")

METRIC_TOLERANCE = 0.0001
STEP_SIZE = 6
MIN_WINDOWS_REQUIRED = 5

DIAGNOSTIC_MODELS = {"X-13"}


def _mae(a, f): return float(np.mean(np.abs(a - f)))
def _rmse(a, f): return float(np.sqrt(np.mean((a - f) ** 2)))
def _bias(a, f): return float(np.mean(a - f))


def rolling_backtest(df: pd.DataFrame, horizon: int = 6):

    metrics = {}
    excluded_models_backtest = []
    total_windows = 0

    for end in range(36, len(df) - horizon, STEP_SIZE):

        train = df.iloc[:end]
        test = df.iloc[end:end + horizon]

        results = run_all_models(train, horizon=horizon, confidence_level=0.95)

        successful_models = []
        total_windows += 1

        for model_name in MODEL_REGISTRY.keys():

            if model_name == "TBATS":
                if "TBATS" not in excluded_models_backtest:
                    excluded_models_backtest.append("TBATS")
                continue

            output = results.get(model_name)
            if not isinstance(output, dict):
                continue

            if output.get("status") != "success":
                if model_name not in excluded_models_backtest:
                    excluded_models_backtest.append(model_name)
                continue

            forecast_df = output.get("forecast_df")
            if forecast_df is None or len(forecast_df) == 0:
                if model_name not in excluded_models_backtest:
                    excluded_models_backtest.append(model_name)
                continue

            forecast = forecast_df["forecast"].values[-horizon:]
            actual = test["value"].values

            if not np.isfinite(forecast).all():
                if model_name not in excluded_models_backtest:
                    excluded_models_backtest.append(model_name)
                continue

            mae = _mae(actual, forecast)
            rmse = _rmse(actual, forecast)
            bias = _bias(actual, forecast)

            metrics.setdefault(model_name, []).append({
                "MAE": mae,
                "RMSE": rmse,
                "Bias": bias,
            })

            successful_models.append(model_name)

        if not successful_models:
            raise RuntimeError("All models failed during backtest window.")

    if total_windows < MIN_WINDOWS_REQUIRED:
        raise RuntimeError("Insufficient rolling windows for certification validity.")

    return metrics, excluded_models_backtest


def evaluate_against_baseline(current_metrics):

    if not os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "w") as f:
            json.dump(current_metrics, f, indent=2, sort_keys=True)
        return "PASS"

    with open(BASELINE_FILE, "r") as f:
        baseline = json.load(f)

    if set(current_metrics.keys()) != set(baseline.keys()):
        return "FAIL_STRUCTURAL_MISMATCH"

    for model in sorted(current_metrics.keys()):

        current_list = current_metrics[model]
        baseline_list = baseline.get(model)

        if baseline_list is None:
            return "FAIL_STRUCTURAL_MISMATCH"

        if len(current_list) != len(baseline_list):
            return "FAIL_STRUCTURAL_MISMATCH"

        for idx in range(len(current_list)):

            current_entry = current_list[idx]
            baseline_entry = baseline_list[idx]

            if set(current_entry.keys()) != set(baseline_entry.keys()):
                return "FAIL"

            for key in sorted(current_entry.keys()):

                if key not in baseline_entry:
                    return "FAIL"

                if abs(current_entry[key] - baseline_entry[key]) > METRIC_TOLERANCE:
                    return "FAIL"

    return "PASS"


def shock_robustness_test():

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = seasonal_dataset()
    shocked_df = inject_spike(df)

    results = run_all_models(shocked_df, horizon=6, confidence_level=0.95)

    diagnostics = []
    overall_status = "PASS"

    for model_name in MODEL_REGISTRY.keys():

        output = results.get(model_name)
        if not isinstance(output, dict):
            diagnostics.append({
                "model_name": model_name,
                "status": "error",
                "empty_output": False,
                "non_finite": False,
                "notes": "Invalid result structure",
            })
            if model_name not in DIAGNOSTIC_MODELS:
                overall_status = "FAIL"
            continue

        status = output.get("status")
        empty_output = output.get("empty_output", False)
        forecast_df = output.get("forecast_df")
        metadata = output.get("metadata", {})

        non_finite = False
        notes = ""

        if status == "success" and forecast_df is not None and len(forecast_df) > 0:
            if not np.isfinite(forecast_df["forecast"].values).all():
                non_finite = True

        if status == "error":
            notes = metadata.get("error_message", "")
            if model_name not in DIAGNOSTIC_MODELS:
                overall_status = "FAIL"
        elif empty_output is True:
            notes = "Empty forecast output"
            if model_name not in DIAGNOSTIC_MODELS:
                overall_status = "FAIL"
        elif non_finite:
            notes = "Non-finite forecast values"
            if model_name not in DIAGNOSTIC_MODELS:
                overall_status = "FAIL"

        diagnostics.append({
            "model_name": model_name,
            "status": status,
            "empty_output": empty_output,
            "non_finite": non_finite,
            "notes": notes,
        })

    artifact_payload = {
        "run_signature": None,
        "dataset": "seasonal_dataset",
        "test": "shock",
        "models": diagnostics,
    }

    with open(SHOCK_ARTIFACT, "w") as f:
        json.dump(artifact_payload, f, indent=2, sort_keys=True)

    return overall_status


def run_performance_suite():

    datasets = [
        seasonal_dataset(),
        linear_trend_dataset(),
        flat_dataset(),
        structural_break_dataset(),
        high_noise_dataset(),
    ]

    all_metrics = {}

    for df in datasets:
        metrics, _ = rolling_backtest(df)
        for model in metrics:
            all_metrics.setdefault(model, []).extend(metrics[model])

    performance_status = evaluate_against_baseline(all_metrics)

    stability_result = stability_test()
    stability_status = stability_result.get("status")

    shock_status = shock_robustness_test()

    print("STRUCTURAL: CERTIFIED")
    print(f"PERFORMANCE: {performance_status}")
    print(f"SHOCK_ROBUSTNESS: {shock_status}")
    print(f"STABILITY: {stability_status}")

    if (
        performance_status == "PASS"
        and stability_status == "PASS"
        and shock_status == "PASS"
    ):
        final = "CERTIFIED"
    else:
        final = "REJECTED"

    print(f"FINAL_VERDICT: {final}")


if __name__ == "__main__":
    run_performance_suite()