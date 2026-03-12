from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

import pandas as pd

from forecastiq.core.forecasting import forecast_series

app = FastAPI(title="ForecastIQ API")


class TimePoint(BaseModel):
    date: str
    value: float


class ForecastRequest(BaseModel):
    series: List[TimePoint]
    freq_choice: str
    horizon: int
    horizon_unit: str
    prophet_cfg: Dict[str, Any] = {}
    model_strategy: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/forecast")
def run_forecast(req: ForecastRequest):
    # Build dataframe
    df = pd.DataFrame([p.dict() for p in req.series])
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date")

    ts = pd.Series(df["value"].values, index=df["date"])

    # Enforce monthly frequency
    ts = ts.asfreq("MS")

    (
        forecast,
        ci_low,
        ci_high,
        info,
        metrics,
        comparison,
        audit,
        ui_summary,
    ) = forecast_series(
        ts=ts,
        freq_choice=req.freq_choice,
        horizon=req.horizon,
        horizon_unit=req.horizon_unit,
        prophet_cfg=req.prophet_cfg,
        model_strategy=req.model_strategy,
    )

    return {
        "forecast": None if forecast is None else forecast.to_dict(),
        "ci_low": None if ci_low is None else ci_low.to_dict(),
        "ci_high": None if ci_high is None else ci_high.to_dict(),
        "info": info,
        "metrics": metrics,
        "comparison": comparison,
        "audit": audit,
        "ui_summary": ui_summary,
    }
