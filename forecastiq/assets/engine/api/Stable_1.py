from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
import pandas as pd

from forecastiq.core.forecasting import forecast_series

# ============================================================
# App
# ============================================================

app = FastAPI(
    title="ForecastIQ API",
    version="1.0.0",
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Models
# ============================================================

class TimePoint(BaseModel):
    date: str
    value: float | None  # allow nulls (real-world data)


class ForecastRequest(BaseModel):
    series: List[TimePoint]
    freq_choice: Literal["Daily", "Weekly", "Monthly"]
    horizon: int = Field(..., gt=0)
    horizon_unit: Literal["days", "weeks", "months"]
    model_strategy: Literal["auto", "ETS", "Linear"]
    prophet_cfg: Dict[str, Any] = {}

# ============================================================
# Helpers
# ============================================================

def normalize_frequency(freq_choice: str) -> str:
    return {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "MS",
    }[freq_choice]


def build_timeseries(series: List[TimePoint], freq_choice: str) -> pd.Series:
    df = pd.DataFrame([p.dict() for p in series])

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date")

    ts = pd.Series(df["value"].values, index=df["date"])

    # Force frequency but DO NOT FAIL
    freq = normalize_frequency(freq_choice)
    ts = ts.asfreq(freq)

    # Intelligent gap handling (TOP-TIER behavior)
    if ts.isna().any():
        # linear interpolation for internal gaps
        ts = ts.interpolate(method="linear")

        # forward/backward fill edges
        ts = ts.ffill().bfill()

    return ts

# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/forecast")
def run_forecast(req: ForecastRequest):
    try:
        ts = build_timeseries(req.series, req.freq_choice)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Engine operates strictly in Periods
    horizon_unit = "Periods"

    try:
        (
            forecast,
            ci_low,
            ci_high,
            info,
            metrics,
            comparison,
            audit,
        ) = forecast_series(
            ts=ts,
            freq_choice=req.freq_choice,
            horizon=req.horizon,
            horizon_unit=horizon_unit,
            prophet_cfg=req.prophet_cfg,
            model_strategy=req.model_strategy,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast engine error: {e}")

    return {
        "forecast": forecast.to_dict(),
        "ci_low": ci_low.to_dict(),
        "ci_high": ci_high.to_dict(),
        "info": info,
        "metrics": metrics,
        "comparison": comparison,
        "audit": audit,
    }
