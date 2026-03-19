# 03 — First Remote Run (1x RTX 3090, 36 slots)

## Change

Validated the full Vast.ai backend against 200 real records per task.
Hardware: 1x RTX 3090 24 GB, simulated via `SINGLE_GPU_MODE=true` on a 2x bundle
(only `CUDA_VISIBLE_DEVICES=0` is exposed to llama-server).

Slot formula: `(24 GB - 6 GB model) / 0.5 GB per slot = 36 parallel slots`.

## Why

The local RTX 3070 Mobile (8 GB) could only support 4 parallel slots.
At 4 workers, jd_reparse ran at 3.1 rec/min — too slow for the production backlog.
Moving to a dedicated cloud GPU with larger VRAM unlocks far more parallelism.

200 records is the warm-up validation sample: enough to measure real rates
without incurring significant cost, and enough to surface any runtime issues
before committing to 1 000-record batches.

## Metrics

| Metric | Local 4x RTX 3070 (20 rec) | 1x RTX 3090 remote (200 rec) |
|---|---|---|
| job_skills compute rate | 7.8 rec/min | **57.2 rec/min** |
| job_skills wall rate | 7.8 rec/min | 32.7 rec/min |
| company_enrich compute rate | 36.7 rec/min | **127.8 rec/min** |
| company_enrich wall rate | 36.5 rec/min | 46.1 rec/min |
| jd_reparse compute rate | 3.1 rec/min | **19.0 rec/min** |
| jd_reparse wall rate | 3.1 rec/min | 16.9 rec/min |
| n_parallel (slots) | 4 | **36** |
| Price | free (local) | ~$0.30/hr |
| Startup overhead | ~0s | ~2m40s (SSH + model load) |

> **Compute rate vs wall rate**: remote runs include a one-time startup cost
> (SSH tunnel + llama-server model load, typically 2–4 min). The compute rate
> measures pure inference throughput; the wall rate includes that overhead.
> For short batches the gap is large; at production scale (1 000+ records)
> the startup is amortised and wall rate converges to compute rate.

## Scale problem

At 19 rec/min for jd_reparse, a 5 000-record backlog takes ~4.4 hours on 1 GPU.
Job_skills at 57.2 rec/min for 8 600 jobs = ~2.5 hours.
Combined daily pipeline: **>6 hours on a single RTX 3090**.

This is the concrete bottleneck that motivates multi-GPU exploration.
The next iteration hardcodes 2 GPUs to validate the SSH-tunnel-per-GPU design.

## Artifacts

- `config.env` — Vast.ai search filters (`NUM_GPUS=2`, `SINGLE_GPU_MODE=true`, `GPU_NAME=3090`)
- `logs/` — three keeper logs (job_skills, company_enrich, jd_reparse), 200 rec each
