# FILE: streamlit_sandbox/certification/cli.py
# ROLE: CERTIFICATION CLI ENTRYPOINT
# STATUS: CANONICAL
# ==================================================

import pandas as pd
from pathlib import Path

from certification.harness import certify_models


def main():
    """
    CLI entrypoint for model certification.
    """

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "input.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Input data not found at expected location: {data_path}"
        )

    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    results = certify_models(df)

    print("\n=== MODEL CERTIFICATION SCORECARD ===\n")
    print(results["scorecard"].to_string(index=False))

    print("\nCertification run complete.\n")


if __name__ == "__main__":
    main()
