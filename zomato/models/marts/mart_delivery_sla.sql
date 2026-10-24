select
    city,
    extract(hour from order_timestamp)::int as order_hour,
    count(*) filter (where is_delivered) as delivered_orders,
    round((percentile_cont(0.5) within group (order by delivery_time_min))::numeric, 1) as p50,
    round((percentile_cont(0.9) within group (order by delivery_time_min))::numeric, 1) as p90
from {{ ref('fct_orders') }}
where is_delivered
group by 1, 2