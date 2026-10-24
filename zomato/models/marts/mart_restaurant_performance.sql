select
    f.restaurant_id,
    r.restaurant_name,
    r.city,
    r.cuisine,
    count(*) as orders,
    sum(case when f.is_delivered then f.sales_amount else 0 end) as revenue,
    round(avg(f.customer_rating), 2) as avg_customer_rating,
    round(avg(f.delivery_time_min), 1) as avg_delivery_min
from {{ ref('fct_orders') }} f
left join {{ ref('dim_restaurants') }} r using (restaurant_id)
group by 1, 2, 3, 4