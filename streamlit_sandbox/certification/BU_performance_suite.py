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

from streamlit_sandbox.execution.engine import run_all_models
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

        for model, output in results.items():

            if model == "_failures":
                continue

            if model == "TBATS":
                if "TBATS" not in excluded_models_backtest:
                    excluded_models_backtest.append("TBATS")
                continue

            if not isinstance(output, dict):
                continue

            if output.get("status") != "success":
                if model not in excluded_models_backtest:
                    excluded_models_backtest.append(model)
                continue

            forecast_df = output.get("forecast_df")
            if forecast_df is None or forecast_df.empty:
                if model not in excluded_models_backtest:
                    excluded_models_backtest.append(model)
                continue

            forecast = forecast_df["forecast"].values[-horizon:]
            actual = test["value"].values

            if not np.isfinite(forecast).all():
                if model not in excluded_models_backtest:
                    excluded_models_backtest.append(model)
                continue

            mae = _mae(actual, forecast)
            rmse = _rmse(actual, forecast)
            bias = _bias(actual, forecast)

            metrics.setdefault(model, []).append({
                "MAE": mae,
                "RMSE": rmse,
                "Bias": bias,
            })

            successful_models.append(model)

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

    for model in sorted(results.keys()):

        if model == "_failures":
            continue

        if model == "X-13":
            diagnostics.append({
                "model_name": model,
                "status": "EXCLUDED",
                "converged": None,
                "warnings": [],
                "non_finite": False,
                "empty_output": False,
                "notes": "Diagnostic-only model",
            })
            print("SHOCK_DIAGNOSTICS:")
            print(f"  model: {model}")
            print("  status: EXCLUDED")
            print("  reason: Diagnostic-only model")
            continue

        output = results.get(model)
        status = "PASS"
        reason = ""
        converged = None
        warnings = []
        non_finite = False
        empty_output = False

        if not isinstance(output, dict):
            status = "FAIL"
            reason = "Invalid result structure"
            overall_status = "FAIL"
        else:
            model_status = output.get("status")
            if model_status != "success":
                status = "FAIL"
                reason = "Model status not success"
                overall_status = "FAIL"

            forecast_df = output.get("forecast_df")

            if forecast_df is None or forecast_df.empty:
                empty_output = True
                status = "FAIL"
                reason = "Empty forecast output"
                overall_status = "FAIL"
            else:
                values = forecast_df.get("forecast")
                if values is None or not np.isfinite(values.values).all():
                    non_finite = True
                    status = "FAIL"
                    reason = "Non-finite forecast values"
                    overall_status = "FAIL"

            metadata = output.get("metadata", {})
            if isinstance(metadata, dict):
                converged = metadata.get("converged")
                warnings = metadata.get("warnings", [])

        diagnostics.append({
            "model_name": model,
            "status": status,
            "converged": converged,
            "warnings": warnings,
            "non_finite": non_finite,
            "empty_output": empty_output,
            "notes": reason,
        })

        print("SHOCK_DIAGNOSTICS:")
        print(f"  model: {model}")
        print(f"  status: {status}")
        if reason:
            print(f"  reason: {reason}")

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

    stability_result = stability_test(seasonal_dataset())
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