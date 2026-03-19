# 01 — Parallel design, n_parallel bug (still 1 thread)

## Change

Added `ThreadPoolExecutor` to run LLM requests concurrently. The VRAM formula
was implemented — `n_parallel = (vram_gb - model_vram_gb) / kv_slot_gb` —
but the result was computed locally and **never written back to `self.n_parallel`**.
The executor always spawned 1 worker.

Backend: local RTX 3070 Mobile (8 GB), same as baseline.
Expected slots: `(8 - 6) / 0.5 = 4`. Actual workers: **1** (bug).

## Why

Validates that the n_parallel bug is real and measurable: despite the
parallelism design being complete, throughput is unchanged from the sequential
baseline. One task is sufficient to prove the no-op — company_enrich was chosen
because its per-record latency is short enough to run quickly.

## Metrics

| Task | Workers (designed) | Workers (actual) | Rate (rec/min) | vs Baseline |
|---|---|---|---|---|
| company_enrich | 4 | **1** (bug) | 15.0 | ≈ 17.4 baseline (within noise) |

> 20 records. The ~14% rate difference (15.0 vs 17.4) is measurement noise at
> this sample size, not a real regression. The key signal: no speedup from parallelism.

## Root cause

```python
# inference/backends/vastai.py — startup() before fix
n_parallel = int((vram_gb - model_vram_gb) / kv_slot_gb)
# BUG: result computed but never assigned
# self.n_parallel = n_parallel  ← missing line
```

The executor was created with `max_workers=self.n_parallel`, which was still 1
(its default). The server had capacity for 4 concurrent slots — all unused.

## Artifacts

- `benchmark_results.json` — company_enrich, 20 rec
- `logs/` — one keeper log (company_enrich, 20 rec)
