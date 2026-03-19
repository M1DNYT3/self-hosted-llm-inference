# 02 — Bugs fixed (n_parallel + TTL, 4 workers local)

## Change

Two bugs fixed:

**1. n_parallel write-back** — computed value now assigned to `self.n_parallel`
so the `ThreadPoolExecutor` actually spawns the correct number of workers.

**2. TTL bug** — `startup()` was called with a hardcoded `queue_size=0`, so the
instance lifetime formula always produced ~31 min regardless of actual batch size.
Instances were terminated mid-batch on larger runs. Fixed: `_count_pending()` now
queries the DB before startup and passes the real queue size.

Backend: local RTX 3070 Mobile (8 GB VRAM).
Slots: `(8 - 6) / 0.5 = 4`. Workers: **4** (correct).
Dispatch: static interleaved batch split (original production design — worker `i`
gets `rows[i::n]`; idle workers cannot steal from busier ones).

## Why

These were the foundational correctness bugs. Until n_parallel was right, no
parallelism measurement was valid. Until TTL was right, no remote run could
complete reliably. 20-record local runs verify the fix cleanly before going remote.

## Metrics

| Task | Workers | Rate (rec/min) | vs Baseline (1 worker) | Speedup |
|---|---|---|---|---|
| job_skills | 4 | 7.8 | 3.9 | **2.0×** |
| company_enrich | 4 | 36.7 | 17.4 | **2.1×** |
| jd_reparse | 4 | 3.1 | 1.8 | **1.7×** |

> 20 records per task, local GPU, static batch dispatch.
> Near-linear scaling at 4 workers limited by bandwidth-bound decode
> (4× workers → ~2× throughput, not 4×) — expected for LLM inference.

## Artifacts

- `benchmark_results.json` — last run (jd_reparse, 20 rec, local)
- `config.env` — Vast.ai search filters (used by iteration 03's remote runs)
- `logs/` — three keeper logs (job_skills, company_enrich, jd_reparse), 20 rec each, static dispatch
