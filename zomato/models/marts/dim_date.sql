with spine as (
    select generate_series(
        '2024-01-01'::date,
        '2026-12-31'::date,
        interval '1 day'
    )::date as date_day
)
select
    date_day,
    extract(year from date_day)::int as year,
    extract(month from date_day)::int as month,
    to_char(date_day, 'Mon') as month_name,
    to_char(date_day, 'Dy') as day_name,
    (extract(isodow from date_day) >= 6) as is_weekend
from spine