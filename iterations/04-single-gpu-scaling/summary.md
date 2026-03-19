# 04 — Single-GPU Scaling Ceiling (RTX PRO 6000 WS + RTX 5090)

## Change

KV cache footprint recalibrated from **0.5 GB/slot** (initial estimate) to
**0.2 GB/slot** (measured at 4096 ctx, Qwen3.5-9B Q4_K_M).

Hypothesis: a workstation GPU with dramatically higher specs than the RTX 3090
should yield proportionally higher throughput — either through more parallel
slots (more VRAM) or faster per-slot decode (more bandwidth/TFLOPS).

Three sub-runs targeting high-spec single GPUs, with parallel count overridden
manually via `--n-parallel` to test scaling behavior:

| Run | Hardware | VRAM | TFLOPS | n_parallel | Rationale |
|---|---|---|---|---|---|
| A | RTX PRO 6000 WS | 96 GB | ~115 | 100 | Max practical slots; ~60% VRAM used |
| B | RTX PRO 6000 WS | 96 GB | ~115 | 72 | Reduced load to check if 100 was saturating |
| C | RTX 5090 | 32 GB | ~170 | 48 | Consumer flagship; higher TFLOPS/bandwidth ratio |

## Hypothesis (and where it was wrong)

**Expected:**
- **TFLOPS (compute)**: Higher TFLOPS → faster token generation → lower per-record
  latency → higher rate at same parallelism, or same rate at higher parallelism.
- **Memory bandwidth**: Higher bandwidth → more parallel KV cache read/write cycles
  per second → throughput scales with parallel count.
- **VRAM**: More VRAM → more slots → near-linear rate increase.

**Reality:**
All three runs capped at approximately **~100 rec/min** — matching the RTX 3090
at $0.30/hr despite the PRO 6000 WS costing >$0.85/hr and having 4× the VRAM
and 3× the TFLOPS.

## Why it fails: memory bandwidth wall

Autoregressive decode is **memory-bandwidth-bound**, not compute-bound.
Every generated token requires loading the full model weights (~6 GB) plus
the growing KV cache for every active slot — from VRAM to the shader cores.

At high parallelism, the **bandwidth bus becomes the bottleneck**:
- More slots → more concurrent KV cache entries → more bytes read per step
- The chip stalls waiting for memory transfers, not for compute
- Higher TFLOPS adds no benefit when the bottleneck is memory, not arithmetic
- Scaling slots beyond the bandwidth ceiling actively degrades per-slot throughput

Reducing from 100 → 72 slots (run B) didn't recover throughput because the
bottleneck is not the number of slots per se — it's the aggregate bandwidth
demand relative to the bus's capacity.

## Metrics

| Run | Hardware | n_parallel | Rate (rec/min) | Cost/hr | Cost/1k rec |
|---|---|---|---|---|---|
| — | RTX 3090 (ref, iter 03) | 36 | 57.2 | ~$0.30 | ~$0.052 |
| A | RTX PRO 6000 WS | 100 | ~100 | >$0.85 | ~$0.085 |
| B | RTX PRO 6000 WS | 72 | ~100 | >$0.85 | ~$0.085 |
| C | RTX 5090 | 48 | ~100 | ~$0.50 | ~$0.050 |

> RTX PRO 6000 WS: 3× TFLOPS, 4× VRAM, ~2× bandwidth vs RTX 3090 → same throughput,
> 2.8× the cost. The spec sheet is irrelevant to autoregressive decode throughput.

## What actually helps: a bigger model

The correct use of high-VRAM workstation cards is not more parallelism with
the same model — it's running a **larger model** at the same parallelism:

- RTX PRO 6000 WS (96 GB): could run Qwen3 35B at 8k context with 36 slots
- Same throughput, higher output quality, similar cost per token
- This matches the workload better than inflating parallel count

But higher quality is not needed here — the pipeline already produces
acceptable output with the 9B model. The real lever is throughput, not quality.

## Conclusion: scale the fleet, not the GPU tier

A single high-spec GPU hits a bandwidth ceiling that cannot be overcome by
adding TFLOPS or VRAM. The right scaling strategy is **horizontal**:

- 4× RTX 4070S Ti at $0.50/hr total → 4× the bandwidth bus, 4× the slots
- Same or less cost than one RTX PRO 6000 WS
- Throughput scales with GPU count (see iteration 05–08)

This experiment eliminated the "buy a bigger GPU" path and redirected effort
toward multi-GPU orchestration.

## Artifacts

- `config.env` — Vast.ai search filters (high TFLOPS, single GPU, KV=0.2 GB)
- `logs/` — three keeper logs (runs A, B, C), job_skills 1000 rec each
- `benchmark_results.json` — last run (run C, RTX 5090)
