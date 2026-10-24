select
    menu_id,
    trim(r_id)::bigint as restaurant_id,
    f_id,
    cuisine,
    round(trim(price)::numeric, 2) as price
from {{ source('raw', 'menu') }}
where trim(r_id) ~ '^[0-9]+$'
  and case
        when trim(price) ~ '^[0-9]+(\.[0-9]+)?$' then trim(price)::numeric > 0
        else false
      end