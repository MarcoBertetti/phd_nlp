-- Recreates analytics_379142691.traffic_source_metrics, partitioned by `date` (daily), no clustering.
-- Full history: all promoted daily exports (events_*) UNION ALL all intraday tables
-- (events_intraday_*), then dedup at event level so that any day still living in
-- intraday (because GA hasn't promoted the daily shard yet) is included, and any
-- day that exists in both tables transiently is not counted twice.
-- Run in BigQuery. If replacing a table with a different partition/cluster spec, DROP the table first, then run this.
-- Second block (below): scheduled incremental refresh (yesterday + today). Run or schedule that block separately from the CREATE above.
--
-- Page-level columns are grouped into 8 funnel-stage buckets (`_f1_home` ...
-- `_f8_other`) instead of the previous ~25 per-page `_<slug>` and `_cat_*` columns.
-- The `events_funnel` CTE assigns each page_view to exactly one bucket via an
-- ordered CASE WHEN (most-specific first: f6 form_submitted -> f5 form_loaded ->
-- f7 patient_portal -> f1 home -> f4 partner -> f3 service_page -> f2 info ->
-- f8 other). Dimensions and shared metrics are unchanged. See
-- bigquery/funnel_buckets_plan.md and bigquery/funnel_buckets.csv for the full
-- bucket definitions and URL patterns.

CREATE OR REPLACE TABLE `data-dashboard-490713.analytics_379142691.traffic_source_metrics`
PARTITION BY date
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
  FROM `data-dashboard-490713.analytics_379142691.events_*`
  -- The events_* wildcard also matches events_intraday_* tables, so restrict to
  -- 8-digit YYYYMMDD suffixes (i.e. only the promoted daily shards).
  WHERE _TABLE_SUFFIX NOT LIKE 'intraday_%'

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
  FROM `data-dashboard-490713.analytics_379142691.events_intraday_*`
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

      -- _f6 form submitted (thank-you pages)
      WHEN page_location LIKE '%/thank-you-%'
        OR page_location LIKE '%/single-chat-thank-you%'
        OR page_location LIKE '%/private-pay-thank-you%'
        THEN '_f6_form_submitted'

      -- _f5 intake form loaded (booking / appointment forms)
      WHEN page_location LIKE '%/rd-%-booking%'
        OR page_location LIKE '%/rd-ontario-app%'
        OR page_location LIKE '%/uninsured-private-pay-appointment%'
        OR page_location LIKE '%/rapid-addictions-medicine-clinic-appt2%'
        THEN '_f5_form_loaded'

      -- _f7 patient portal (separate apps / subdomains)
      WHEN page_location LIKE '%patient.rocketdoctor.ca%'
        OR page_location LIKE '%patient.starship.usacs.com%'
        THEN '_f7_patient_portal'

      -- _f1 home (homepage only — anchored to bare domain so /patient.rocketdoctor.ca/ is not picked up here)
      WHEN REGEXP_CONTAINS(page_location, r'^https?://(www\.)?rocketdoctor\.ca/?(\?|#|$)')
        THEN '_f1_home'

      -- _f4 partner / branded
      WHEN page_location LIKE '%/dr-%'
        OR page_location LIKE '%/drhoffmann%'
        OR page_location LIKE '%/genaissance%'
        OR page_location LIKE '%/synergy-orthopedics%'
        OR page_location LIKE '%/embrace-program%'
        OR page_location LIKE '%/centrum-pharmacy%'
        OR page_location LIKE '%/olive-fertility%'
        OR page_location LIKE '%/the-reproductive-care-centre%'
        OR page_location LIKE '%/ottawa-fertility-centre%'
        OR page_location LIKE '%/georgian-bay-general-hospital%'
        THEN '_f4_partner'

      -- _f3 service page
      WHEN page_location LIKE '%/online-doctor-%'
        OR page_location LIKE '%/online-doctors-%'
        OR page_location LIKE '%/virtual-walk-in-clinic-%'
        OR page_location LIKE '%/online-walk-in-clinic-%'
        OR page_location LIKE '%/doctors-notes-%'
        OR page_location LIKE '%/dermatologists-online-%'
        OR page_location LIKE '%/ontario-%-online%'
        OR page_location LIKE '%/ontario-%-treatment-online%'
        OR page_location LIKE '%/alberta-%-online%'
        OR page_location LIKE '%/alberta-%-treatment-online%'
        OR page_location LIKE '%/family-doctor-ontario%'
        OR page_location LIKE '%/medical-clinic-ontario%'
        OR page_location LIKE '%/virtual-care-ontario%'
        OR page_location LIKE '%/virtual-doctor-ontario%'
        OR page_location LIKE '%/medication-refills-online-ontario%'
        OR page_location LIKE '%/prescription-refills-ontario%'
        OR page_location LIKE '%/specialist-referrals-ontario%'
        OR page_location LIKE '%/pediatrics-doctors%'
        OR page_location LIKE '%/obgyn-ont%'
        OR page_location LIKE '%/virtual-adhd-clinic%'
        OR page_location LIKE '%/rapid-addictions-medicine-clinic/%'
        OR page_location LIKE '%/rapid-addictions-medicine-clinic-ad/%'
        OR page_location LIKE '%/addiction-withdrawal-doctor%'
        OR page_location LIKE '%/alberta/%'
        OR page_location LIKE '%/ontario/%'
        OR page_location LIKE '%/bc-online-doctor%'
        OR page_location LIKE '%/rocket-doctor-bc%'
        OR page_location LIKE '%/rocket-doctor-alberta%'
        OR page_location LIKE '%/no-ohip/%'
        OR page_location LIKE '%/doctor-no-ohip/%'
        OR page_location LIKE '%/on-free-with-ohip%'
        OR page_location LIKE '%/see-a-doctor%'
        OR page_location LIKE '%/see-a-virtual-doctor%'
        OR page_location LIKE '%/chat-with-a-doctor%'
        OR page_location LIKE '%/telehealth-consultation-canada-lander%'
        OR page_location LIKE '%/chat-55-availability%'
        OR page_location LIKE '%/single-chat-74-availability%'
        OR page_location LIKE '%/pa-65-availability%'
        OR page_location LIKE '%/private-pay-65-availability%'
        THEN '_f3_service_page'

      -- _f2 info / corporate
      WHEN page_location LIKE '%/about/%'
        OR page_location LIKE '%/careers/%'
        OR page_location LIKE '%/partners/%'
        OR page_location LIKE '%/our-technology/%'
        OR page_location LIKE '%/a-guide-to-virtual-care/%'
        OR page_location LIKE '%/referrals/%'
        OR page_location LIKE '%/start-your-virtual-care-practice%'
        OR page_location LIKE '%/subscription-plans%'
        OR page_location LIKE '%/ai-research-study/%'
        OR page_location LIKE '%/savevirtualcare%'
        OR page_location LIKE '%/patient-email-communications/%'
        OR page_location LIKE '%/rd-devices-terms-and-conditions/%'
        OR page_location LIKE '%/test-new-availability/%'
        OR page_location LIKE '%/staging-%'
        OR page_location LIKE '%/terms-of-use/%'
        OR page_location LIKE '%/privacy-policy%'
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
-- intraday and daily exist transiently. Keeps table schema and PARTITION BY date.
-- Requires DML on traffic_source_metrics.
-- =============================================================================

BEGIN TRANSACTION;

DELETE FROM `data-dashboard-490713.analytics_379142691.traffic_source_metrics`
WHERE date IN (DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY), CURRENT_DATE());

INSERT INTO `data-dashboard-490713.analytics_379142691.traffic_source_metrics`
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
  FROM `data-dashboard-490713.analytics_379142691.events_intraday_*`
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
  FROM `data-dashboard-490713.analytics_379142691.events_*`
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
        OR page_location LIKE '%/single-chat-thank-you%'
        OR page_location LIKE '%/private-pay-thank-you%'
        THEN '_f6_form_submitted'

      WHEN page_location LIKE '%/rd-%-booking%'
        OR page_location LIKE '%/rd-ontario-app%'
        OR page_location LIKE '%/uninsured-private-pay-appointment%'
        OR page_location LIKE '%/rapid-addictions-medicine-clinic-appt2%'
        THEN '_f5_form_loaded'

      WHEN page_location LIKE '%patient.rocketdoctor.ca%'
        OR page_location LIKE '%patient.starship.usacs.com%'
        THEN '_f7_patient_portal'

      WHEN REGEXP_CONTAINS(page_location, r'^https?://(www\.)?rocketdoctor\.ca/?(\?|#|$)')
        THEN '_f1_home'

      WHEN page_location LIKE '%/dr-%'
        OR page_location LIKE '%/drhoffmann%'
        OR page_location LIKE '%/genaissance%'
        OR page_location LIKE '%/synergy-orthopedics%'
        OR page_location LIKE '%/embrace-program%'
        OR page_location LIKE '%/centrum-pharmacy%'
        OR page_location LIKE '%/olive-fertility%'
        OR page_location LIKE '%/the-reproductive-care-centre%'
        OR page_location LIKE '%/ottawa-fertility-centre%'
        OR page_location LIKE '%/georgian-bay-general-hospital%'
        THEN '_f4_partner'

      WHEN page_location LIKE '%/online-doctor-%'
        OR page_location LIKE '%/online-doctors-%'
        OR page_location LIKE '%/virtual-walk-in-clinic-%'
        OR page_location LIKE '%/online-walk-in-clinic-%'
        OR page_location LIKE '%/doctors-notes-%'
        OR page_location LIKE '%/dermatologists-online-%'
        OR page_location LIKE '%/ontario-%-online%'
        OR page_location LIKE '%/ontario-%-treatment-online%'
        OR page_location LIKE '%/alberta-%-online%'
        OR page_location LIKE '%/alberta-%-treatment-online%'
        OR page_location LIKE '%/family-doctor-ontario%'
        OR page_location LIKE '%/medical-clinic-ontario%'
        OR page_location LIKE '%/virtual-care-ontario%'
        OR page_location LIKE '%/virtual-doctor-ontario%'
        OR page_location LIKE '%/medication-refills-online-ontario%'
        OR page_location LIKE '%/prescription-refills-ontario%'
        OR page_location LIKE '%/specialist-referrals-ontario%'
        OR page_location LIKE '%/pediatrics-doctors%'
        OR page_location LIKE '%/obgyn-ont%'
        OR page_location LIKE '%/virtual-adhd-clinic%'
        OR page_location LIKE '%/rapid-addictions-medicine-clinic/%'
        OR page_location LIKE '%/rapid-addictions-medicine-clinic-ad/%'
        OR page_location LIKE '%/addiction-withdrawal-doctor%'
        OR page_location LIKE '%/alberta/%'
        OR page_location LIKE '%/ontario/%'
        OR page_location LIKE '%/bc-online-doctor%'
        OR page_location LIKE '%/rocket-doctor-bc%'
        OR page_location LIKE '%/rocket-doctor-alberta%'
        OR page_location LIKE '%/no-ohip/%'
        OR page_location LIKE '%/doctor-no-ohip/%'
        OR page_location LIKE '%/on-free-with-ohip%'
        OR page_location LIKE '%/see-a-doctor%'
        OR page_location LIKE '%/see-a-virtual-doctor%'
        OR page_location LIKE '%/chat-with-a-doctor%'
        OR page_location LIKE '%/telehealth-consultation-canada-lander%'
        OR page_location LIKE '%/chat-55-availability%'
        OR page_location LIKE '%/single-chat-74-availability%'
        OR page_location LIKE '%/pa-65-availability%'
        OR page_location LIKE '%/private-pay-65-availability%'
        THEN '_f3_service_page'

      WHEN page_location LIKE '%/about/%'
        OR page_location LIKE '%/careers/%'
        OR page_location LIKE '%/partners/%'
        OR page_location LIKE '%/our-technology/%'
        OR page_location LIKE '%/a-guide-to-virtual-care/%'
        OR page_location LIKE '%/referrals/%'
        OR page_location LIKE '%/start-your-virtual-care-practice%'
        OR page_location LIKE '%/subscription-plans%'
        OR page_location LIKE '%/ai-research-study/%'
        OR page_location LIKE '%/savevirtualcare%'
        OR page_location LIKE '%/patient-email-communications/%'
        OR page_location LIKE '%/rd-devices-terms-and-conditions/%'
        OR page_location LIKE '%/test-new-availability/%'
        OR page_location LIKE '%/staging-%'
        OR page_location LIKE '%/terms-of-use/%'
        OR page_location LIKE '%/privacy-policy%'
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
