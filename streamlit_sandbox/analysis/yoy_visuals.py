# FILE: analysis/yoy_visuals.py
# ROLE: YEAR-OVER-YEAR VISUALIZATION (EXECUTIVE CONTEXT)
# STATUS: GOVERNED / CANONICAL
# ==================================================

import altair as alt
import pandas as pd
import numpy as np


def yoy_chart(df: pd.DataFrame) -> alt.Chart:
    data = df.copy()

    # --------------------------------------------------
    # CANONICAL TIME KEYS
    # --------------------------------------------------
    data["year"] = data["date"].dt.year.astype(str)
    data["month"] = data["date"].dt.month.astype(int)
    data["month_name"] = data["date"].dt.strftime("%b")

    month_sort = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec",
    ]

    # --------------------------------------------------
    # VALUE LINES (RAW DATA)
    # --------------------------------------------------
    value_df = data.assign(
        series=lambda d: d["year"],
        line_type="Value",
    )

    # --------------------------------------------------
    # TREND LINES (ONLY WHEN STATISTICALLY VALID)
    # --------------------------------------------------
    trend_rows = []

    for yr, g in data.groupby("year"):
        g = g.sort_values("month")

        # Require at least 3 points for a meaningful trend
        if len(g) < 3:
            continue

        x = g["month"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept

        trend_rows.append(
            g.assign(
                value=y_hat,
                series=f"{yr} Trend",
                line_type="Trend",
            )
        )

    trend_df = (
        pd.concat(trend_rows, ignore_index=True)
        if trend_rows
        else pd.DataFrame(columns=value_df.columns)
    )

    plot_df = pd.concat([value_df, trend_df], ignore_index=True)

    # --------------------------------------------------
    # LEGEND SELECTION (OPACITY ONLY â€” NEVER REMOVES DATA)
    # --------------------------------------------------
    selector = alt.selection_point(
        fields=["series"],
        bind="legend",
        toggle=True,
    )

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X(
                "month_name:N",
                sort=month_sort,
                title="Month",
            ),
            y=alt.Y("value:Q", title="Sales"),
            color=alt.Color(
                "series:N",
                legend=alt.Legend(title="Series"),
            ),
            strokeWidth=alt.condition(
                "datum.line_type === 'Trend'",
                alt.value(3),
                alt.value(2),
            ),
            strokeDash=alt.condition(
                "datum.line_type === 'Trend'",
                alt.value([6, 4]),
                alt.value([1, 0]),
            ),
            opacity=alt.condition(
                selector,
                alt.value(1.0),
                alt.value(0.08),
            ),
        )
        .add_params(selector)
        .properties(height=360)
    )

    return chart

