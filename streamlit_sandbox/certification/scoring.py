def calibrate_confidence(
    beat_naive_rate: float,
    stability_score: float | None,
    ci_valid: bool,
) -> str:
    """
    Returns High / Medium / Low.
    """
    if beat_naive_rate >= 0.65 and ci_valid and stability_score is not None and stability_score < 0.5:
        return "High"

    if beat_naive_rate >= 0.45:
        return "Medium"

    return "Low"

