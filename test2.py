def group_anomalies(anomaly_results, max_gap_minutes=60):
    # Keep only anomalies
    a = anomaly_results[anomaly_results["is_anomaly"] == 1].copy()

    # Sort by time
    a = a.sort_values("timestamp")

    # Previous anomaly timestamp
    a["prev_timestamp"] = a["timestamp"].shift(1)

    # Gap in minutes
    a["gap_minutes"] = (a["timestamp"] - a["prev_timestamp"]).dt.total_seconds() / 60

    # New group if first anomaly or gap > 60 minutes
    a["new_group"] = a["prev_timestamp"].isna() | (a["gap_minutes"] > max_gap_minutes)

    # Group id (1, 2, 3, ...)
    a["group_id"] = a["new_group"].cumsum()

    # Group-level summary
    groups = (
        a.groupby("group_id", as_index=False)
         .agg(
             start_time=("timestamp", "min"),
             end_time=("timestamp", "max"),
             peak_value=("value", "max"),
             anomaly_count=("timestamp", "count"),
         )
    )

    # Duration in minutes
    groups["duration_minutes"] = (
        (groups["end_time"] - groups["start_time"]).dt.total_seconds() / 60
    ).astype(int)

    # Order columns as required
    return groups[["group_id", "start_time", "end_time", "duration_minutes", "peak_value", "anomaly_count"]]
