-- View: data-dashboard-490713.paid_marketing_data.vw_marketing_performance
-- Combines unified_paid_ads with GA4 traffic_source_metrics from BOTH the CA and US datasets.
-- The two GA4 tables share the same schema: dimensions, shared session/engagement
-- metrics, and the eight funnel-stage columns (`_f1_home` ... `_f8_other`).
-- See bigquery/funnel_buckets_plan.md.
-- A `domain` column ('ca' or 'us') is added so consumers can filter by region.
--
-- Assumes campaign_name values are effectively unique across CA and US in unified_paid_ads
-- (otherwise a paid row could match GA4 rows in both regions and be duplicated;
-- if that ever happens, add a domain column to unified_paid_ads and include it in the join).

CREATE OR REPLACE VIEW `data-dashboard-490713.paid_marketing_data.vw_marketing_performance` AS

WITH paid AS (
  SELECT *
  FROM `data-dashboard-490713.paid_marketing_data.unified_paid_ads`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY report_date, campaign_name, source_platform
    ORDER BY row_updated_at DESC
  ) = 1
),

-- =============================================================================
-- CA traffic_source_metrics. Same column shape as US.
-- =============================================================================
ga_ca AS (
  SELECT
    'ca' AS domain,
    date, source, medium, campaign, platform,

    -- Shared metrics
    unique_users, total_sessions, engaged_sessions, engaged_session_rate,
    total_pageviews, avg_pages_per_session, total_engagement_sec, avg_engagement_time_sec,
    bounced_sessions,

    -- Funnel buckets (same schema on CA and US)
    _f1_home, _f2_info, _f3_service_page, _f4_partner,
    _f5_form_loaded, _f6_form_submitted, _f7_patient_portal, _f8_other

  FROM `data-dashboard-490713.analytics_379142691.traffic_source_metrics`
  WHERE LOWER(medium) IN ('cpc', 'paid', 'ppc', 'paidsocial')
),

-- =============================================================================
-- US traffic_source_metrics. Same column shape as CA.
-- =============================================================================
ga_us AS (
  SELECT
    'us' AS domain,
    date, source, medium, campaign, platform,

    -- Shared metrics
    unique_users, total_sessions, engaged_sessions, engaged_session_rate,
    total_pageviews, avg_pages_per_session, total_engagement_sec, avg_engagement_time_sec,
    bounced_sessions,

    -- Funnel buckets (same schema on CA and US)
    _f1_home, _f2_info, _f3_service_page, _f4_partner,
    _f5_form_loaded, _f6_form_submitted, _f7_patient_portal, _f8_other

  FROM `data-dashboard-490713.analytics_382008010.traffic_source_metrics`
  WHERE LOWER(medium) IN ('cpc', 'paid', 'ppc', 'paidsocial')
),

ga_combined AS (
  SELECT * FROM ga_ca
  UNION ALL
  SELECT * FROM ga_us
),

-- =============================================================================
-- Source platform normalization + dedup (one row per date/campaign/domain/source).
-- =============================================================================
ga_clean AS (
  SELECT
    *,
    CASE
      WHEN LOWER(TRIM(source)) LIKE '%facebook%' OR LOWER(TRIM(source)) LIKE '%fb%' THEN 'facebook'
      WHEN LOWER(TRIM(source)) LIKE '%instagram%' THEN 'instagram'
      WHEN LOWER(TRIM(source)) LIKE '%meta%' THEN 'meta'
      WHEN LOWER(TRIM(source)) LIKE '%google%' OR LOWER(TRIM(source)) LIKE '%adwords%' THEN 'google'
      WHEN LOWER(TRIM(source)) LIKE '%bing%' THEN 'bing'
      WHEN LOWER(TRIM(source)) LIKE '%reddit%' THEN 'reddit'
      WHEN LOWER(TRIM(source)) LIKE '%tiktok%' THEN 'tiktok'
      ELSE REGEXP_REPLACE(LOWER(TRIM(source)), r'[^a-z0-9]+', '')
    END AS ga4_source_clean
  FROM ga_combined
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY
      date,
      campaign,
      domain,
      CASE
        WHEN LOWER(TRIM(source)) LIKE '%facebook%' OR LOWER(TRIM(source)) LIKE '%fb%' THEN 'facebook'
        WHEN LOWER(TRIM(source)) LIKE '%instagram%' THEN 'instagram'
        WHEN LOWER(TRIM(source)) LIKE '%meta%' THEN 'meta'
        WHEN LOWER(TRIM(source)) LIKE '%google%' OR LOWER(TRIM(source)) LIKE '%adwords%' THEN 'google'
        WHEN LOWER(TRIM(source)) LIKE '%bing%' THEN 'bing'
        WHEN LOWER(TRIM(source)) LIKE '%reddit%' THEN 'reddit'
        WHEN LOWER(TRIM(source)) LIKE '%tiktok%' THEN 'tiktok'
        ELSE REGEXP_REPLACE(LOWER(TRIM(source)), r'[^a-z0-9]+', '')
      END
    ORDER BY total_sessions DESC, unique_users DESC
  ) = 1
)

SELECT
  p.*,
  g.domain,
  g.unique_users,
  g.total_sessions,
  g.engaged_sessions,
  g.engaged_session_rate,
  g.total_pageviews,
  g.avg_pages_per_session,
  g.total_engagement_sec,
  g.avg_engagement_time_sec,
  g.bounced_sessions,

  -- Funnel buckets
  g._f1_home,
  g._f2_info,
  g._f3_service_page,
  g._f4_partner,
  g._f5_form_loaded,
  g._f6_form_submitted,
  g._f7_patient_portal,
  g._f8_other

FROM paid p
LEFT JOIN ga_clean g
  ON p.report_date = g.date
  AND p.campaign_name = g.campaign
  AND (
    CASE
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%facebook%' THEN 'facebook'
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%instagram%' THEN 'instagram'
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%meta%' THEN 'meta'
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%google_ads%' THEN 'google'
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%bing_ads%' THEN 'bing'
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%reddit%' THEN 'reddit'
      WHEN LOWER(TRIM(p.source_platform)) LIKE '%tiktok%' THEN 'tiktok'
      ELSE REGEXP_REPLACE(LOWER(TRIM(p.source_platform)), r'[^a-z0-9]+', '')
    END
  ) = g.ga4_source_clean;
