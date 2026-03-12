import pandas as pd
import re

ILLEGAL_SHEET_CHARS = r"[:\\/?*\[\]]"


def sanitize_sheet_name(name: str) -> str:
    clean = re.sub(ILLEGAL_SHEET_CHARS, "_", name)
    return clean[:31]


def export_excel(package: dict, file_path: str):
    """
    Export a dictionary of DataFrames to a multi-sheet Excel workbook.
    Keys become sheet names (sanitized).
    """
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name, df in package.items():
            safe_name = sanitize_sheet_name(sheet_name)
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=safe_name, index=False)
