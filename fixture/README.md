# Fixture DB

Self-contained Postgres 16 database with the exact dataset used during the
inference engineering case study. Contains only records that the inference
tasks will actually process — nothing else.

---

## Tables included

### Input (read-only)
| Schema.Table | Filter applied | Purpose |
|---|---|---|
| `market.jobs_raw` | `description IS NOT NULL` | Job descriptions for job_skills + jd_reparse |
| `market.jobs_derived` | joined to included jobs_raw rows | Parsed attributes for jd_reparse |
| `market.companies` | referenced by included jobs_raw | Company registry for company_enrich |
| `market.company_scores` | scores for included companies | Heuristic scores for company_enrich |

### Output (empty — all records are eligible)
| Schema.Table | Task that writes here |
|---|---|
| `market.job_skills_premium` | job_skills |
| `market.company_scores_premium` | company_enrich |

`jd_reparse` writes back to `market.jobs_derived` (sets `llm_reparsed_at`).
`reset.sql` nullifies this column so every row is eligible on startup.

---

## How to generate the dump (from production)

```bash
# 1. Export schema (all schemas needed for FK constraints)
pg_dump --schema-only -Fp jobslicer > fixture/schema.sql

# 2. Export filtered data
psql jobslicer <<'SQL'
\COPY (SELECT * FROM market.jobs_raw WHERE description IS NOT NULL) TO 'jobs_raw.tsv'
\COPY (
  SELECT d.* FROM market.jobs_derived d
  JOIN market.jobs_raw r ON r.id = d.job_fk
  WHERE r.description IS NOT NULL
) TO 'jobs_derived.tsv'
\COPY (
  SELECT c.* FROM market.companies c
  WHERE EXISTS (
    SELECT 1 FROM market.jobs_raw r WHERE r.company_id = c.id AND r.description IS NOT NULL
  )
) TO 'companies.tsv'
\COPY (
  SELECT cs.* FROM market.company_scores cs
  WHERE EXISTS (
    SELECT 1 FROM market.companies c WHERE c.id = cs.company_id
  )
) TO 'company_scores.tsv'
SQL

# 3. Sanitize (removes email/phone from descriptions)
python fixture/sanitize.py fixture/dump_raw.sql | gzip -9 > fixture/dump.sql.gz

# 4. Verify row counts
psql inference_fixture -c "
  SELECT 'jobs_raw' AS t, COUNT(*) FROM market.jobs_raw
  UNION ALL SELECT 'jobs_derived', COUNT(*) FROM market.jobs_derived
  UNION ALL SELECT 'companies', COUNT(*) FROM market.companies
  UNION ALL SELECT 'company_scores', COUNT(*) FROM market.company_scores;
"
```

---

## How to spin up

```bash
# Build and start the fixture DB
docker build -t inference-fixture fixture/
docker run -d --name inference-fixture -p 5433:5432 inference-fixture

# Or via compose
docker compose -f docker/compose.yaml up fixture-db -d

# Connect
psql postgresql://fixture:fixture@localhost:5433/inference_fixture
```

---

## Reset to clean state (re-run any task from scratch)

```bash
psql postgresql://fixture:fixture@localhost:5433/inference_fixture \
  -f fixture/reset.sql
```
