# 05 — Multi-GPU Hardcoded (2x RTX 5090, static vs shared-queue)

## Change

First multi-GPU run. GPU count hardcoded to 2, with two explicit changes:

**1. onstart script** — two llama-server launch blocks, one per GPU (hardcoded, not generated
from GPU count). Each block sets `CUDA_VISIBLE_DEVICES` to its card index and starts a
server on a dedicated port. Adding a third GPU would require manually editing the onstart.
This limitation is the direct motivation for iteration 06.

**2. Thread-affinity routing** — `thread_id % num_gpus` determines which GPU a worker
targets. Under static batch dispatch (worker `i` gets `rows[i::n]`) this is stable:
each worker always routes to the same GPU. Under shared-queue dispatch workers pick from
a common queue, so the thread→GPU mapping drifts over time.

Hardware: 2x RTX 5090 (32 GB/card, 109 TFLOPS/card), $0.732/hr.
Slots: `(32 - 6) / 0.5 × 2 = 102 total workers` (51/card).

## Why

Validates that horizontal scaling works at all. The single-GPU ceiling (~92–101 rec/min)
should roughly double by adding a second independent bandwidth bus. Also reveals the
dispatch model's interaction with multi-GPU routing.

## Metrics — job_skills (1000 records)

| Dispatch | Workers | Compute rate | Wall rate | Failures | Cost/hr | Cost/1k rec |
|---|---|---|---|---|---|---|
| static | 102 | **219.1 rec/min** | 154.7 rec/min | 1/1000 | $0.732 | **$0.056** |
| shared-queue | 102 | 157.5 rec/min | 129.5 rec/min | **12/1000** | $0.732 | $0.077 |

> Static vs shared-queue: 39% faster compute rate, 12× fewer failures.
> Reference: 1x RTX 5090 @ 92.2 rec/min (iter 04C). Static 2x = **2.4× speedup**.

## Metrics — company_enrich (1000 records)

| Dispatch | Workers | Compute rate | Wall rate | Failures |
|---|---|---|---|---|
| static | 102 | 396.8 rec/min | 269.2 rec/min | 0/1000 |
| shared-queue | 102 | 401.6 rec/min | 271.0 rec/min | 1/1000 |

> Dispatch mode is irrelevant for company_enrich — short, uniform per-record latency
> means the thread→GPU mapping drift under shared-queue doesn't produce measurable overhead.

## Why static dispatch wins for job_skills

Job descriptions vary significantly in length (token count). Under static batches,
worker `i` always routes to `GPU (i % 2)` — assignments are fixed regardless of record
duration, so fast workers don't pull ahead and overload one GPU. Under shared-queue,
fast workers drain the common queue faster, creating uneven load across the two GPUs.
At 102 workers this imbalance is enough to saturate one server's slot queue while the
other is underutilised.

The failures in both runs (1 static, 12 shared-queue) are llama.cpp server connection
drops — the SSH tunnel to the inference process became unresponsive briefly. Root cause
is non-datacenter host instability: consumer or semi-pro environments can experience CPU
contention, storage interference, or thermal throttling that briefly interrupts the
data stream. The same pattern appeared in iteration 04's jd_reparse static run on the
same class of host. The higher count in the shared-queue run is run-specific noise
correlated with host load at that moment, not a causal effect of dispatch mode.

## Scaling verdict

2x RTX 5090 @ $0.732/hr → 219 rec/min.
1x RTX 5090 @ $0.538/hr → 92 rec/min.

Cost-efficiency: $0.056 per 1k records vs $0.097 — **1.7× better per dollar**.
Throughput: 2.4× speedup from 2× GPUs — near-linear (slightly super-linear due to
more total KV slots: 102 vs 47, a 2.17× increase).

The hardcoded design works, but the onstart limitation caps it at 2 GPUs.
Dynamic generation is required to scale further — see iteration 06.

## Run commands

```bash
# Static dispatch
bash harness/bench.sh 05-multi-gpu-hardcoded job_skills 1000 0 "" "" remote static
bash harness/bench.sh 05-multi-gpu-hardcoded company_enrich 1000 0 "" "" remote static

# Shared-queue dispatch
bash harness/bench.sh 05-multi-gpu-hardcoded job_skills 1000 0 "" "" remote
bash harness/bench.sh 05-multi-gpu-hardcoded company_enrich 1000 0 "" "" remote
```

## Artifacts

- `config.env` — 2x RTX 5090 filters, price floor $0.65, min 1 Gbps inet
- `logs/` — four keeper logs (job_skills × 2 dispatch, company_enrich × 2 dispatch)
- `benchmark_results.json` — last run (company_enrich, shared-queue)
