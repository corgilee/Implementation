def detect_anomalies(df, threshold=1.5):
    # Sort by time so rolling window is correct
    df = df.sort_values("timestamp")

    # Moving average of the past 2 hours (12 intervals), excluding current point
    df["moving_avg_2hr"] = (
        df["value"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    # Anomaly flag
    df["is_anomaly"] = (
        df["value"] / df["moving_avg_2hr"] >= threshold
    ).astype(int)

    # Return requested columns
    return df[["timestamp", "value", "is_anomaly"]]
