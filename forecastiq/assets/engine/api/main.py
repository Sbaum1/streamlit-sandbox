from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Literal
import pandas as pd

from forecastiq.core.forecasting import forecast_series

# ============================================================
# App
# ============================================================

app = FastAPI(
    title="ForecastIQ API",
    version="1.1.1",
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
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
    value: float | None  # real-world safe (allows sparse series)


class ForecastRequest(BaseModel):
    series: List[TimePoint]

    # Explicit user-facing frequency
    freq_choice: Literal["Daily", "Weekly", "Monthly"]

    # Forecast horizon
    horizon: int = Field(..., gt=0)

    # User-friendly unit
    horizon_unit: Literal["days", "weeks", "months"]

    # Model control
    model_strategy: Literal["auto", "ETS", "Linear"] = "auto"

    # Optional Prophet overrides
    prophet_cfg: Dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_horizon_alignment(self):
        alignment = {
            "Daily": "days",
            "Weekly": "weeks",
            "Monthly": "months",
        }

        expected_unit = alignment[self.freq_choice]
        if self.horizon_unit != expected_unit:
            raise ValueError(
                f"horizon_unit '{self.horizon_unit}' does not align with freq_choice '{self.freq_choice}'"
            )

        return self

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
    df = pd.DataFrame([p.model_dump() for p in series])

    if df.empty:
        raise ValueError("Series is empty")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date")

    ts = pd.Series(df["value"].values, index=df["date"])

    freq = normalize_frequency(freq_choice)

    # Align frequency
    ts = ts.asfreq(freq)

    # ===============================
    # Intelligent gap recovery
    # ===============================
    if ts.isna().any():
        ts = ts.interpolate(method="linear", limit_direction="both")
        ts = ts.ffill().bfill()

    if ts.isna().any():
        raise ValueError(
            "Series contains unrecoverable missing values after alignment"
        )

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

    # Internal engine standardization
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
        raise HTTPException(
            status_code=500,
            detail=f"Forecast engine error: {e}",
        )

    return {
        "forecast": forecast.to_dict(),
        "ci_low": ci_low.to_dict(),
        "ci_high": ci_high.to_dict(),
        "info": info,
        "metrics": metrics,
        "comparison": comparison,
        "audit": audit,
    }
