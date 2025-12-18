WITH page_views_with_date AS (
    SELECT
        user_id,
        page_id,
        view_timestamp,
        CAST(view_timestamp AS DATE) AS date,
        time_on_page_seconds
    FROM page_views
),

daily_time AS (
    SELECT
        user_id,
        date,
        SUM(time_on_page_seconds) AS total_time_seconds
    FROM page_views_with_date
    GROUP BY user_id, date
)

SELECT
    pv.user_id,
    pv.date,
    pv.page_id,
    pv.view_timestamp,
    db.daily_charge
        * pv.time_on_page_seconds
        / dt.total_time_seconds AS attributed_revenue
FROM page_views_with_date pv
JOIN daily_time dt
    ON pv.user_id = dt.user_id
   AND pv.date = dt.date
JOIN daily_billing db
    ON pv.user_id = db.user_id
   AND pv.date = db.date;
