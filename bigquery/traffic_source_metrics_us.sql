-- USA pipeline: analytics_382008010.traffic_source_metrics
-- Partitioned by date, clustered by source, campaign.
-- First block: full backfill — promoted daily exports (events_*) UNION ALL all
-- intraday tables (events_intraday_*), deduped at event level so any day still
-- living in intraday (because GA hasn't promoted the daily shard yet) is included
-- and any day that exists in both tables transiently is not counted twice.
-- Second block: scheduled refresh that overwrites yesterday + today partitions atomically.
-- Run in BigQuery. If replacing a table with a different partition/cluster spec, DROP the table first, then run this.
--
-- Page-level columns are grouped into 8 funnel-stage buckets (`_f1_home` ...
-- `_f8_other`) instead of the previous per-page `_<slug>` and `_cat_*` columns.
-- The `events_funnel` CTE assigns each page_view to exactly one bucket via an
-- ordered CASE WHEN (most-specific first: f6 form_submitted -> f5 form_loaded ->
-- f7 patient_portal -> f1 home -> f4 partner -> f3 service_page -> f2 info ->
-- f8 other). Dimensions and shared metrics are unchanged. See
-- bigquery/funnel_buckets_plan.md and bigquery/funnel_buckets.csv for the full
-- bucket definitions and URL patterns.

CREATE OR REPLACE TABLE `data-dashboard-490713.analytics_382008010.traffic_source_metrics`
PARTITION BY date
CLUSTER BY source, campaign
AS

WITH raw_events AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS date,
    event_name,
    event_timestamp,
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'session_engaged') AS session_engaged,
    COALESCE(TRIM(collected_traffic_source.manual_source), traffic_source.source, '(direct)') AS source,
    COALESCE(TRIM(collected_traffic_source.manual_medium), traffic_source.medium, '(none)') AS medium,
    COALESCE(TRIM(collected_traffic_source.manual_campaign_name), traffic_source.name, '(not set)') AS campaign,
    TRIM(collected_traffic_source.manual_term) AS term,
    TRIM(collected_traffic_source.manual_content) AS content,
    platform
  FROM `data-dashboard-490713.analytics_382008010.events_*`
  -- The events_* wildcard also matches events_intraday_* tables, so restrict to
  -- 8-digit YYYYMMDD suffixes (i.e. only the promoted daily shards).
  WHERE _TABLE_SUFFIX NOT LIKE 'intraday_%'
    AND _TABLE_SUFFIX >= '20200101'

  UNION ALL

  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS date,
    event_name,
    event_timestamp,
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'session_engaged') AS session_engaged,
    COALESCE(TRIM(collected_traffic_source.manual_source), traffic_source.source, '(direct)') AS source,
    COALESCE(TRIM(collected_traffic_source.manual_medium), traffic_source.medium, '(none)') AS medium,
    COALESCE(TRIM(collected_traffic_source.manual_campaign_name), traffic_source.name, '(not set)') AS campaign,
    TRIM(collected_traffic_source.manual_term) AS term,
    TRIM(collected_traffic_source.manual_content) AS content,
    platform
  FROM `data-dashboard-490713.analytics_382008010.events_intraday_*`
),

events AS (
  -- Dedup events that may appear in both intraday and the promoted daily.
  -- (user_pseudo_id, event_timestamp, event_name) is effectively unique in GA4
  -- since event_timestamp is microsecond-precision.
  SELECT * EXCEPT(event_timestamp)
  FROM raw_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY user_pseudo_id, event_timestamp, event_name
    ORDER BY event_timestamp
  ) = 1
),

events_funnel AS (
  -- Tag each page_view with exactly one funnel bucket. Order matters:
  -- the first matching WHEN wins, so more-specific buckets must come first.
  SELECT
    *,
    CASE
      WHEN event_name != 'page_view' THEN NULL

      -- _f6 form submitted (thank-you pages — covers all variants: -usa, -usa-others,
      -- -maryland, -ny, -cali-cash-pay, -cali-insurance, -payment-received, -staging,
      -- -usa-staging-page, -usacs).
      WHEN page_location LIKE '%/thank-you-%'
        THEN '_f6_form_submitted'

      -- _f5 intake form loaded (booking / appointment / payment forms, including
      -- the state-specific contact forms which capture intake info).
      WHEN page_location LIKE '%/california-book-appointment%'
        OR page_location LIKE '%/maryland-book-appointment%'
        OR page_location LIKE '%/rd-new-york-appointment%'
        OR page_location LIKE '%/engagewell-appointment%'
        OR page_location LIKE '%/ozempic-appointment%'
        OR page_location LIKE '%/contact/california%'
        OR page_location LIKE '%/contact/new-york%'
        OR page_location LIKE '%/agency-name-appointment%'
        OR page_location LIKE '%/housing-works-appointment%'
        OR page_location LIKE '%/pearsuite-appointment%'
        OR page_location LIKE '%/usacs-form%'
        OR page_location LIKE '%/usacs-survey-form%'
        OR page_location LIKE '%/device-submission%'
        OR page_location LIKE '%/us-payment%'
        OR page_location LIKE '%/rd-us-pay-for-visit%'
        THEN '_f5_form_loaded'

      -- _f7 patient portal (separate apps / subdomains)
      WHEN page_location LIKE '%patient.rocketdoctor.us%'
        OR page_location LIKE '%us-rocketdoctor.adracare.com%'
        OR page_location LIKE '%try.rocketdoctor.io%'
        THEN '_f7_patient_portal'

      -- _f1 home (homepage only — anchored to bare domain)
      WHEN REGEXP_CONTAINS(page_location, r'^https?://(www\.)?rocketdoctor\.io/?(\?|#|$)')
        THEN '_f1_home'

      -- _f4 partner / branded community-org & employer programs
      WHEN page_location LIKE '%/acacia%'
        OR page_location LIKE '%/agency-name/%'
        OR page_location LIKE '%/camba%'
        OR page_location LIKE '%/ccah%'
        OR page_location LIKE '%/consulthealth%'
        OR page_location LIKE '%/harlem-united%'
        OR page_location LIKE '%/housing-works/%'
        OR page_location LIKE '%/pearsuite/%'
        THEN '_f4_partner'

      -- _f3 service page (state / coverage type / condition / specialty)
      WHEN page_location LIKE '%/online-doctor-california%'
        OR page_location LIKE '%/medi-cal-doctors-california-lp%'
        OR page_location LIKE '%/medicaid-doctors-maryland%'
        OR page_location LIKE '%/medicare-doctors-%'
        OR page_location LIKE '%/medi-cal-other-care-plan%'
        OR page_location LIKE '%/california-skin-doctors%'
        OR page_location LIKE '%/doctors-online-%'
        OR page_location LIKE '%/see-online-doctor-%'
        OR page_location LIKE '%/mental-health/%'
        THEN '_f3_service_page'

      -- _f2 info / corporate (the state-specific /contact/* pages were already
      -- routed to _f5 above, so it is safe to use a broad %/contact/% catch here).
      WHEN page_location LIKE '%/about/%'
        OR page_location LIKE '%/partners/%'
        OR page_location LIKE '%/patients/%'
        OR page_location LIKE '%/physicians/%'
        OR page_location LIKE '%/our-technology/%'
        OR page_location LIKE '%/coverage-pricing/%'
        OR page_location LIKE '%/feedback/%'
        OR page_location LIKE '%/news-blog%'
        OR page_location LIKE '%/telehealth-informed-consent/%'
        OR page_location LIKE '%/style-guide/%'
        OR page_location LIKE '%/terms-of-use%'
        OR page_location LIKE '%/terms-of-service%'
        OR page_location LIKE '%/privacy-policy%'
        OR page_location LIKE '%/bridging-access-to-care/%'
        OR page_location LIKE '%/contact/%'
        THEN '_f2_info'

      ELSE '_f8_other'
    END AS funnel_bucket
  FROM events
)

SELECT
  e.date,
  e.source,
  e.medium,
  e.campaign,
  e.platform,

  -- Users & Sessions
  COUNT(DISTINCT e.user_pseudo_id) AS unique_users,
  COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS total_sessions,
  COUNT(DISTINCT CASE WHEN e.session_engaged = '1' THEN CONCAT(e.user_pseudo_id, '-', e.session_id) END) AS engaged_sessions,
  COUNT(DISTINCT CASE WHEN e.session_engaged = '1' THEN CONCAT(e.user_pseudo_id, '-', e.session_id) END)
    / COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS engaged_session_rate,
  COUNTIF(e.event_name = 'page_view') AS total_pageviews,
  COUNTIF(e.event_name = 'page_view')
    / COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS avg_pages_per_session,
  SUM(e.engagement_time_msec) / 1000 AS total_engagement_sec,
  SUM(e.engagement_time_msec) / 1000
    / COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS avg_engagement_time_sec,
  COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id))
    - COUNT(DISTINCT CASE WHEN e.session_engaged = '1' THEN CONCAT(e.user_pseudo_id, '-', e.session_id) END) AS bounced_sessions,

  -- Funnel stages: unique users with at least one matching page_view per row.
  COUNT(DISTINCT IF(e.funnel_bucket = '_f1_home',           e.user_pseudo_id, NULL)) AS _f1_home,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f2_info',           e.user_pseudo_id, NULL)) AS _f2_info,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f3_service_page',   e.user_pseudo_id, NULL)) AS _f3_service_page,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f4_partner',        e.user_pseudo_id, NULL)) AS _f4_partner,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f5_form_loaded',    e.user_pseudo_id, NULL)) AS _f5_form_loaded,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f6_form_submitted', e.user_pseudo_id, NULL)) AS _f6_form_submitted,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f7_patient_portal', e.user_pseudo_id, NULL)) AS _f7_patient_portal,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f8_other',          e.user_pseudo_id, NULL)) AS _f8_other

FROM events_funnel e
GROUP BY date, source, medium, campaign, platform
HAVING COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) > 0;

-- =============================================================================
-- Scheduled refresh: DELETE + INSERT for yesterday and today only (multi-statement).
-- Reads ALL of events_intraday_* for [yesterday, today] PLUS events_* for
-- yesterday, then dedups at the event level so nothing is missed if GA hasn't
-- promoted yesterday's daily shard yet, and nothing is double counted if both
-- intraday and daily exist transiently. Schema, partitioning and clustering on
-- the existing table are preserved (DML cannot change them).
-- =============================================================================

BEGIN TRANSACTION;

DELETE FROM `data-dashboard-490713.analytics_382008010.traffic_source_metrics`
WHERE date IN (DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY), CURRENT_DATE());

INSERT INTO `data-dashboard-490713.analytics_382008010.traffic_source_metrics`
WITH raw_events AS (
  -- All intraday rows for yesterday and today.
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS date,
    event_name,
    event_timestamp,
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'session_engaged') AS session_engaged,
    COALESCE(TRIM(collected_traffic_source.manual_source), traffic_source.source, '(direct)') AS source,
    COALESCE(TRIM(collected_traffic_source.manual_medium), traffic_source.medium, '(none)') AS medium,
    COALESCE(TRIM(collected_traffic_source.manual_campaign_name), traffic_source.name, '(not set)') AS campaign,
    TRIM(collected_traffic_source.manual_term) AS term,
    TRIM(collected_traffic_source.manual_content) AS content,
    platform
  FROM `data-dashboard-490713.analytics_382008010.events_intraday_*`
  WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))
                          AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())

  UNION ALL

  -- Promoted daily shard for yesterday (may overlap with yesterday's intraday;
  -- duplicates are removed in the next CTE).
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS date,
    event_name,
    event_timestamp,
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') AS engagement_time_msec,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'session_engaged') AS session_engaged,
    COALESCE(TRIM(collected_traffic_source.manual_source), traffic_source.source, '(direct)') AS source,
    COALESCE(TRIM(collected_traffic_source.manual_medium), traffic_source.medium, '(none)') AS medium,
    COALESCE(TRIM(collected_traffic_source.manual_campaign_name), traffic_source.name, '(not set)') AS campaign,
    TRIM(collected_traffic_source.manual_term) AS term,
    TRIM(collected_traffic_source.manual_content) AS content,
    platform
  FROM `data-dashboard-490713.analytics_382008010.events_*`
  WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))
),

events AS (
  SELECT * EXCEPT(event_timestamp)
  FROM raw_events
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY user_pseudo_id, event_timestamp, event_name
    ORDER BY event_timestamp
  ) = 1
),

events_funnel AS (
  SELECT
    *,
    CASE
      WHEN event_name != 'page_view' THEN NULL

      WHEN page_location LIKE '%/thank-you-%'
        THEN '_f6_form_submitted'

      WHEN page_location LIKE '%/california-book-appointment%'
        OR page_location LIKE '%/maryland-book-appointment%'
        OR page_location LIKE '%/rd-new-york-appointment%'
        OR page_location LIKE '%/engagewell-appointment%'
        OR page_location LIKE '%/ozempic-appointment%'
        OR page_location LIKE '%/contact/california%'
        OR page_location LIKE '%/contact/new-york%'
        OR page_location LIKE '%/agency-name-appointment%'
        OR page_location LIKE '%/housing-works-appointment%'
        OR page_location LIKE '%/pearsuite-appointment%'
        OR page_location LIKE '%/usacs-form%'
        OR page_location LIKE '%/usacs-survey-form%'
        OR page_location LIKE '%/device-submission%'
        OR page_location LIKE '%/us-payment%'
        OR page_location LIKE '%/rd-us-pay-for-visit%'
        THEN '_f5_form_loaded'

      WHEN page_location LIKE '%patient.rocketdoctor.us%'
        OR page_location LIKE '%us-rocketdoctor.adracare.com%'
        OR page_location LIKE '%try.rocketdoctor.io%'
        THEN '_f7_patient_portal'

      WHEN REGEXP_CONTAINS(page_location, r'^https?://(www\.)?rocketdoctor\.io/?(\?|#|$)')
        THEN '_f1_home'

      WHEN page_location LIKE '%/acacia%'
        OR page_location LIKE '%/agency-name/%'
        OR page_location LIKE '%/camba%'
        OR page_location LIKE '%/ccah%'
        OR page_location LIKE '%/consulthealth%'
        OR page_location LIKE '%/harlem-united%'
        OR page_location LIKE '%/housing-works/%'
        OR page_location LIKE '%/pearsuite/%'
        THEN '_f4_partner'

      WHEN page_location LIKE '%/online-doctor-california%'
        OR page_location LIKE '%/medi-cal-doctors-california-lp%'
        OR page_location LIKE '%/medicaid-doctors-maryland%'
        OR page_location LIKE '%/medicare-doctors-%'
        OR page_location LIKE '%/medi-cal-other-care-plan%'
        OR page_location LIKE '%/california-skin-doctors%'
        OR page_location LIKE '%/doctors-online-%'
        OR page_location LIKE '%/see-online-doctor-%'
        OR page_location LIKE '%/mental-health/%'
        THEN '_f3_service_page'

      WHEN page_location LIKE '%/about/%'
        OR page_location LIKE '%/partners/%'
        OR page_location LIKE '%/patients/%'
        OR page_location LIKE '%/physicians/%'
        OR page_location LIKE '%/our-technology/%'
        OR page_location LIKE '%/coverage-pricing/%'
        OR page_location LIKE '%/feedback/%'
        OR page_location LIKE '%/news-blog%'
        OR page_location LIKE '%/telehealth-informed-consent/%'
        OR page_location LIKE '%/style-guide/%'
        OR page_location LIKE '%/terms-of-use%'
        OR page_location LIKE '%/terms-of-service%'
        OR page_location LIKE '%/privacy-policy%'
        OR page_location LIKE '%/bridging-access-to-care/%'
        OR page_location LIKE '%/contact/%'
        THEN '_f2_info'

      ELSE '_f8_other'
    END AS funnel_bucket
  FROM events
)

SELECT
  e.date,
  e.source,
  e.medium,
  e.campaign,
  e.platform,

  COUNT(DISTINCT e.user_pseudo_id) AS unique_users,
  COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS total_sessions,
  COUNT(DISTINCT CASE WHEN e.session_engaged = '1' THEN CONCAT(e.user_pseudo_id, '-', e.session_id) END) AS engaged_sessions,
  COUNT(DISTINCT CASE WHEN e.session_engaged = '1' THEN CONCAT(e.user_pseudo_id, '-', e.session_id) END)
    / COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS engaged_session_rate,
  COUNTIF(e.event_name = 'page_view') AS total_pageviews,
  COUNTIF(e.event_name = 'page_view')
    / COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS avg_pages_per_session,
  SUM(e.engagement_time_msec) / 1000 AS total_engagement_sec,
  SUM(e.engagement_time_msec) / 1000
    / COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) AS avg_engagement_time_sec,
  COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id))
    - COUNT(DISTINCT CASE WHEN e.session_engaged = '1' THEN CONCAT(e.user_pseudo_id, '-', e.session_id) END) AS bounced_sessions,

  COUNT(DISTINCT IF(e.funnel_bucket = '_f1_home',           e.user_pseudo_id, NULL)) AS _f1_home,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f2_info',           e.user_pseudo_id, NULL)) AS _f2_info,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f3_service_page',   e.user_pseudo_id, NULL)) AS _f3_service_page,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f4_partner',        e.user_pseudo_id, NULL)) AS _f4_partner,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f5_form_loaded',    e.user_pseudo_id, NULL)) AS _f5_form_loaded,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f6_form_submitted', e.user_pseudo_id, NULL)) AS _f6_form_submitted,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f7_patient_portal', e.user_pseudo_id, NULL)) AS _f7_patient_portal,
  COUNT(DISTINCT IF(e.funnel_bucket = '_f8_other',          e.user_pseudo_id, NULL)) AS _f8_other

FROM events_funnel e
GROUP BY date, source, medium, campaign, platform
HAVING COUNT(DISTINCT CONCAT(e.user_pseudo_id, '-', e.session_id)) > 0;

COMMIT TRANSACTION;
