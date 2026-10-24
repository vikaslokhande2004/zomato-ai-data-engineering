select
    r.review_id,
    r.order_id,
    r.user_id::bigint as customer_id,
    r.restaurant_id::text as restaurant_id,
    r.rating::int as rating,
    r.comment::text as comment,
    r.review_date::date as review_date,
    res.city as city
from {{ source('raw', 'reviews') }} r
left join {{ ref('stg_restaurants') }} res
    on r.restaurant_id = res.restaurant_id
where r.comment is not null