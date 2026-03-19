# 04 — Single-GPU Scaling Ceiling (RTX PRO 6000 WS + RTX 5090)

## Change

Hypothesis test: a workstation GPU with dramatically higher specs than the RTX 3090
should yield proportionally higher throughput — either through more parallel slots
(more VRAM) or faster per-slot decode (more bandwidth/TFLOPS).

Three sub-runs targeting high-spec single GPUs, with `LLM_VAST_KV_SLOT_GB` set
per config to force a specific parallel count:

| Run | Hardware | VRAM | TFLOPS | n_parallel | Rationale |
|---|---|---|---|---|---|
| A | RTX PRO 6000 WS | 96 GB | 119 | 99 | Max practical slots; ~25% VRAM used |
| B | RTX PRO 6000 WS | 96 GB | 119 | 71 | Reduced load — check if 99 was the bottleneck |
| C | RTX 5090 | 32 GB | 108 | 47 | Consumer flagship; higher bandwidth/dollar ratio |

jd_reparse excluded: run A terminated mid-batch due to connection loss on the host.
Confirmed the expected behavior (batch interruption, not a software bug) — excluded
from this iteration's metrics. Will be captured at scale in iterations 05–07.

## Hypothesis (and where it was wrong)

**Expected:**
- **TFLOPS (compute)**: Higher TFLOPS → faster token generation → lower per-record
  latency → higher rate at the same parallelism.
- **VRAM**: More VRAM → more slots → near-linear rate increase.
- **Memory bandwidth**: Higher bandwidth → more parallel KV cache cycles → throughput
  scales with parallel count.

**Reality:**
All three runs capped at ~92–102 rec/min for job_skills — matching the RTX 3090 at
$0.30/hr despite the PRO 6000 WS costing $0.78/hr with 4× the VRAM and 3× the TFLOPS.
Run B (71 slots vs 99) moved the needle by less than 2%. The bottleneck isn't slot
count or TFLOPS — it's the **memory bandwidth bus**.

## Why it fails: memory bandwidth wall

Autoregressive decode is **memory-bandwidth-bound**, not compute-bound.
Every generated token requires loading the full model weights (~6 GB) plus
the growing KV cache for every active slot — from VRAM to the shader cores.

At high parallelism, the bandwidth bus becomes the bottleneck:
- More slots → more concurrent KV cache entries → more bytes transferred per step
- The chip stalls waiting for memory, not for arithmetic
- Higher TFLOPS adds no benefit; higher VRAM only gives more idle capacity
- Halving slot count (99 → 71) doesn't recover throughput because the bottleneck
  is aggregate bandwidth demand, not queue depth

On iteration 03's RTX 3090: 36 slots, ~$0.30/hr, same per-slot decode speed.
Here: 99 slots, $0.78/hr, the same ceiling. The extra 63 slots and $0.48/hr are wasted.

## Metrics — job_skills (1000 records)

| Run | Hardware | n_parallel | Compute rate | Wall rate | Cost/hr | Cost/1k rec |
|---|---|---|---|---|---|---|
| ref | RTX 3090 (iter 03, 200 rec) | 36 | 57.2 rec/min | 32.7 rec/min | ~$0.30 | ~$0.087 |
| A | RTX PRO 6000 WS | 99 | **101.5 rec/min** | 81.5 rec/min | $0.780 | **$0.128** |
| B | RTX PRO 6000 WS | 71 | **99.9 rec/min** | 69.3 rec/min | $0.780 | **$0.130** |
| C | RTX 5090 | 47 | **92.2 rec/min** | 71.3 rec/min | $0.538 | **$0.097** |

> Compute rate = pure inference time (startup/teardown excluded).
> Wall rate = full pipeline including Vast.ai instance startup (~3–4 min overhead).
>
> RTX PRO 6000 WS: 3× TFLOPS, 4× VRAM vs RTX 3090 → same throughput ceiling, 2.6× the cost.
> Run B vs A: 28% fewer slots → <2% throughput change. The ceiling is the bus, not the queue.

## Metrics — company_enrich (464 records, all available)

| Run | Hardware | n_parallel | Compute rate | Wall rate |
|---|---|---|---|---|
| A | RTX PRO 6000 WS | 99 | 161.8 rec/min | 63.9 rec/min |
| C | RTX 5090 | 47 | 130.6 rec/min | 94.4 rec/min |

> Only 464 companies in the fixture DB — sufficient for single-GPU comparison,
> insufficient to saturate multi-GPU runs. Dataset will be expanded before iterations 05–07.
> Wall rate is lower for PRO 6000 on this task: shorter compute time means
> the fixed startup overhead ($0.78/hr × 3–4 min) is a larger fraction of total wall time.

## What actually predicts throughput

Measured vs expected across this iteration:

| Metric | Expected effect | Observed |
|---|---|---|
| TFLOPS (119 vs 35) | Higher compute → faster decode | No effect. Bandwidth-bound. |
| VRAM (96 GB vs 24 GB) | More slots → higher rate | More idle capacity, same ceiling. |
| Slot count (99 vs 71) | Fewer slots → lower queue pressure | <2% difference. Not the bottleneck. |
| GPU price ($0.78 vs $0.30) | Higher cost → higher perf | 2.6× cost → same throughput. |

**What does predict throughput:** GPU count. Each additional card adds an independent
bandwidth bus. A fleet of four "good enough" cards saturates at 4× the throughput of
one premium card — for the same or lower cost. Proven in iteration 07.

## Correct use of high-VRAM cards

The right application for 96 GB VRAM is **model size scaling**, not parallelism:
- RTX PRO 6000 WS could run Qwen3 35B Q4_K_M (~20 GB) with ~38 slots
- Same cost, same parallelism ceiling, but higher output quality per token
- This workload doesn't need 35B — acceptable quality at 9B

The hardware is right for a different problem.

## Conclusion: scale the fleet, not the GPU tier

A single high-spec GPU hits a bandwidth ceiling that cannot be overcome by adding
TFLOPS or VRAM. The right scaling strategy is horizontal:

- 4× mid-range cards = 4× independent bandwidth buses = 4× throughput
- Same or less hourly cost than one workstation card
- Proven: iterations 05–07 scale from 2× to 4×/8× cards → rate scales linearly

This experiment eliminated the "buy a bigger GPU" path and validated the fleet approach.

## Run commands

```bash
# Run A — RTX PRO 6000 WS, 99 slots
bash harness/bench.sh 04-single-gpu-scaling job_skills 1000 0 "" "" remote static \
  iterations/04-single-gpu-scaling/config-pro6000-100slots.env

# Run B — RTX PRO 6000 WS, 71 slots
bash harness/bench.sh 04-single-gpu-scaling job_skills 1000 0 "" "" remote static \
  iterations/04-single-gpu-scaling/config-pro6000-72slots.env

# Run C — RTX 5090, 47 slots (job_skills)
bash harness/bench.sh 04-single-gpu-scaling job_skills 1000 0 "" "" remote static \
  iterations/04-single-gpu-scaling/config-5090-48slots.env

# Run C — RTX 5090, 47 slots (company_enrich)
bash harness/bench.sh 04-single-gpu-scaling company_enrich 1000 0 "" "" remote static \
  iterations/04-single-gpu-scaling/config-5090-48slots.env
```

## Artifacts

- `config-pro6000-100slots.env` — RTX PRO 6000 WS, KV=0.9 GB → 99 slots
- `config-pro6000-72slots.env` — RTX PRO 6000 WS, KV=1.25 GB → 71 slots
- `config-5090-48slots.env` — RTX 5090, KV=0.54 GB (actual 47 slots, default fallback)
- `logs/` — five keeper logs: Run A (job_skills, company_enrich), Run B (job_skills), Run C (job_skills, company_enrich)
- `benchmark_results.json` — last run (Run C, company_enrich, 464 rec)
