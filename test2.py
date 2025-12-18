'''
Solution steps
Sort by user and time
Compute rolling average of previous 5 pages
Compare ratio against threshold
Mark boolean flag
'''

def detect_high_engagement(df, threshold=2.0):
    # 1) Sort data
    df = df.sort_values(["user_id", "view_timestamp"])

    # 2) Rolling average of previous 5 pages
    df["avg_time_last_5_pages"] = (
        df.groupby("user_id")["time_on_page_seconds"]
          .shift(1)
          .rolling(5, min_periods=1)
          .mean()
    )

    # 3) High engagement flag
    df["is_high_engagement"] = (
        df["time_on_page_seconds"] / df["avg_time_last_5_pages"]
    ) >= threshold

    # 4) Return requested columns
    return df[
        ["user_id", "view_timestamp", "time_on_page_seconds", "is_high_engagement"]
    ]
