{{ config(
    tags=['ai'],
    materialized='table'
) }}

select
    rr.city,
    e.topic,
    e.sentiment_label,
    count(*) as reviews,
    round(avg(e.sentiment_score)::numeric, 3) as avg_sentiment_score,
    round(avg(rr.rating)::numeric, 2) as avg_star_rating,
    count(*) filter (where e.key_issue is not null) as flagged_issues
from {{ source('ai', 'review_enriched') }} e
inner join {{ ref('stg_reviews') }} rr using (review_id)
group by 1, 2, 3
