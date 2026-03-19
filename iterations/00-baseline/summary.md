# 00 — Baseline (sequential, 1 worker)

## Change

No parallelism. The LLM pipeline processes records one at a time in a single
thread. `avg_inference_secs` is hardcoded at 30 s/record — a rough estimate
that does not account for per-task variance.

Backend: local RTX 3070 Mobile (8 GB VRAM), LM Studio, Qwen3.5-9B Q4_K_M.

## Why

Establishes the measurement floor. All subsequent iterations are measured
relative to this throughput. Single-threaded execution provides the cleanest
per-record latency signal — no queuing or scheduling noise.

## Metrics

| Task | Workers | Rate (rec/min) | Avg latency |
|---|---|---|---|
| job_skills | 1 | 3.9 | ~15 s |
| company_enrich | 1 | 17.4 | ~3.5 s |
| jd_reparse | 1 | 1.8 | ~33 s/slice × ~3 slices |

> 20 records per task. 1 company_enrich parse failure (19/20 ok) — included in
> rate; non-deterministic at small sample sizes.

## Artifacts

- `benchmark_results.json` — last run (jd_reparse, 20 rec)
- `logs/` — three keeper logs (job_skills, company_enrich, jd_reparse), 20 rec each
