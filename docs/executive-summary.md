# Executive Summary

> Status: DRAFT — built iteratively during the interview loop.

---

## Problem and constraints

A production data pipeline required daily LLM inference over batches of up to 10,000 records
across three task types: job skill extraction, company enrichment, and job description
re-parsing. A preliminary benchmark on a local RTX 3070 Mobile estimated API costs of
$10–50 per 10k-record batch — $50–150/month at daily cadence — with no control over
pricing, model deprecation, or data exposure.

**Constraints:**
- Budget: sub-$1/hr for GPU compute; total inference cost must stay under $15/month on
  average, with free days handled by a local CPU fallback on an already-paid VPS.
- Throughput: the pipeline runs on a daily cron; inference must complete within the nightly
  window without becoming the bottleneck.
- Cold start: on-demand GPU instances must be ready to process within 5 minutes typical,
  10 minutes worst case — otherwise startup cost erodes the per-record economics.
- Predictability: API providers charge per token and deprecate models; cost must be bounded
  by compute hours, not data volume.

---

## Approach

Self-hosted inference on rented GPU compute (Vast.ai), routed through a CPU-first fallback
that uses an always-on llama.cpp container for small batches. Large batches rent an on-demand
GPU instance for the duration of the job and destroy it immediately after.

**Model**: Qwen3.5-9B Q4_K_M (6 GB VRAM footprint, `--reasoning-budget 0` to suppress
~300 unnecessary thinking tokens per call).

**Router**: `pick_backend()` estimates wall-clock CPU time from queue size and calibrated
throughput (tokens/sec from `benchmark_results.json`). If estimated time ≤ 2h threshold,
use CPU. Otherwise rent GPU.

**Batch pipeline**: `ThreadPoolExecutor` with `n_parallel` workers (computed from VRAM),
producer-consumer queues, main-thread aggregator that bulk-commits every 50 results.

---

## Key results

| Metric | Baseline | Final |
|---|---|---|
| Throughput (job_skills) | 1 rec/min (sequential) | **369 rec/min** (4x RTX 4070S Ti) |
| Cost per 1k records | ~$0.046 (1x RTX 3090) | **~$0.023** (4x RTX 4070S Ti) |
| API cost alternative | $10–50 per 10k records | — |
| Self-hosted monthly cost | — | **~$15/month worst case** |
| Cold-start overhead | — | 2–4 min non-processing time |

---

## Why it matters

The system inverts the API cost structure: instead of paying per token (volume-driven),
it pays per compute-hour (time-driven). At this workload shape, self-hosted inference
on commodity gaming GPUs is 20–100× cheaper per record than API pricing, with consistent
model quality and no vendor lock-in.

The key hardware insight — that VRAM bandwidth, not TFLOPS or VRAM capacity, predicts
LLM decode throughput — was validated empirically across five GPU configurations, including
a negative result on a high-TFLOPS workstation card (RTX PRO 6000 WS) that matched a
budget gaming card (RTX 3090) at three times the cost.

---

*(Next sections: Architecture, Iteration Log — see iteration-log.md)*
