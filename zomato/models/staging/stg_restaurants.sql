-- parse the messy dimension (-- →null, 50+ ratings→50, ₹ 200→200, city after last comma):

with parsed as (
    select
        trim(id)::bigint                                          as restaurant_id,
        nullif(trim(name), '')                                    as restaurant_name,
        trim(coalesce(substring(city from '[^,]+$'), city))       as city,
        case
            when trim(rating) ~ '^[0-9]+(\.[0-9]+)?$'
            then round(trim(rating)::numeric, 1)
        end                                                       as rating,
        substring(rating_count from '[0-9]+')::int                as rating_count,
        substring(cost from '[0-9]+')::int                        as cost_for_two,
        nullif(trim(cuisine), '')                                 as cuisine,
        nullif(trim(lic_no), '')                                  as license_no
    from {{ source('raw', 'restaurants') }}
    where trim(id) ~ '^[0-9]+$'
)

select distinct on (restaurant_id) *
from parsed
order by restaurant_id