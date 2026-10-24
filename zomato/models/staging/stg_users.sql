select
    trim(user_id)::bigint as customer_id,
    name as customer_name,
    lower(email) as email,
    case when trim(age) ~ '^[0-9]+$' then trim(age)::int end as age,
    gender,
    marital_status,
    occupation,
    monthly_income as income_band,
    education,
    case when trim(family_size) ~ '^[0-9]+$' then trim(family_size)::int end as family_size
from {{ source('raw', 'users') }}
where trim(user_id) ~ '^[0-9]+$'