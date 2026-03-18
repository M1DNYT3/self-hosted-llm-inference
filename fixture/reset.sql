-- reset.sql
-- Executed after dump restore on fixture DB init.
-- Clears output tables and nullifies llm_reparsed_at so ALL records are eligible
-- for all three inference tasks (equivalent to running with --force).

-- job_skills task writes here
TRUNCATE market.job_skills_premium;

-- company_enrich task writes here
TRUNCATE market.company_scores_premium;

-- jd_reparse task targets rows where llm_reparsed_at IS NULL
-- Reset so every jobs_derived row is eligible
UPDATE market.jobs_derived
SET llm_reparsed_at = NULL;
