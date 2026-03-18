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
| Throughput (job_skills) | 1 rec/min (sequential) | **369 rec/min** (4x cards, ~76 workers) |
| Cost per 1k records | ~$0.046 (1x card) | **~$0.023** (4x cards, ~$0.50/hr) |
| API cost alternative | $10–50 per 10k records | — |
| Self-hosted monthly cost | — | **~$15/month worst case** |
| Cold-start overhead | — | 2–4 min non-processing time |

---

## Why it matters

The system inverts the API cost structure: instead of paying per token (volume-driven),
it pays per compute-hour (time-driven). At this workload shape, self-hosted inference
on commodity gaming GPUs is 20–100× cheaper per record than API pricing, with consistent
model quality and no vendor lock-in.

Two hardware insights were validated empirically across five GPU configurations:

1. **TFLOPS does not predict LLM decode throughput.** A TFLOPS filter set at 100/GPU was
   intended as a quality gate, but the high-TFLOPS cards it admitted (workstation-class)
   matched a much cheaper gaming card at 2.8× the cost. Autoregressive decode is
   memory-bandwidth-bound — every token requires loading the full weight matrix from VRAM.
   Relaxing the filter to 20/GPU opened the search to gaming cards, which have better
   bandwidth per dollar.

2. **Per-record latency is nearly flat within a GPU family.** An RTX 3070 Laptop and an RTX
   3090 produce similar per-slot decode speed once slots are correctly sized to VRAM headroom.
   The actual throughput multiplier is **GPU count**, not GPU tier. Going from 1 card (36 slots)
   to 4 cards (~76 slots) — each cheaper than the original — delivered 3.4× the throughput at
   half the cost per record.

---

*(Next sections: Architecture, Iteration Log — see iteration-log.md)*
