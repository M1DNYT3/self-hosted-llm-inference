# Executive Summary

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
throughput. If estimated time ≤ threshold (configurable per task), use CPU — $0 GPU cost.
Otherwise rent GPU. Light days are effectively free; heavy days scale linearly with rental time.

**Batch pipeline**: `ThreadPoolExecutor` with `n_parallel` workers computed from VRAM:
`(vram_gb - model_vram_gb) / kv_slot_gb`. Workers dispatch to per-GPU llama-servers through
isolated SSH port-forward processes, one per GPU.

---

## Key results

| Metric | Baseline | Final |
|---|---|---|
| Throughput (job_skills, 8x confirmed) | 1 rec/min (sequential bug) | **256 rec/min** (8x RTX 2080 Ti, 256 workers) |
| Throughput (job_skills, 4x confirmed — 3090) | — | **242.9 rec/min** (4x RTX 3090, iter 07) |
| Throughput (job_skills, 4x confirmed — 4070S Ti) | — | **237.5 rec/min** (4x RTX 4070S Ti, 76 workers) |
| Cost per 1k records (4x RTX 3090) | ~$0.046 (1x card, $0.30/hr) | **~$0.050** (4x RTX 3090, $0.61/hr) |
| Cost per 1k records (4x RTX 4070S Ti) | — | **~$0.057** (4x RTX 4070S Ti, $0.703/hr) |
| 10k records — wall time | ~92 min (1x GPU) | **~41–57 min** (4x–8x fleet) |
| 10k records — cost | ~$0.46 (1x GPU) | **~$0.51–0.57** (4x–8x fleet) |
| API cost alternative | $10–50 per 10k records | — |
| Self-hosted monthly cost | — | **~$15/month worst case** |
| Cold-start overhead | — | 2–4 min non-processing time |

---

## Key findings

**1. TFLOPS does not predict LLM decode throughput.**
A filter set at 100 TFLOPS/GPU was intended as a quality gate, but the workstation-class
cards it admitted matched cheaper gaming cards at 2.8× the cost. Autoregressive decode
is memory-bandwidth-bound — every token requires loading the full weight matrix from VRAM.
Relaxing the filter to 20/GPU opened the search to gaming cards, which have better
bandwidth per dollar.

**2. GPU count is the throughput multiplier, not GPU tier.**
Per-record latency is nearly flat within a GPU family once slots are correctly sized to VRAM
headroom. Going from 1 card (36 slots) to 4 cards produced 2.2× throughput (109 → 243 rec/min)
at 2× the cost — confirmed on both RTX 3090 and RTX 4070S Ti. Scaling is sub-linear: 4× GPU
count delivers ~2.2× throughput. At 8x, the gap widens further because the RTX 2080 Ti has
lower bandwidth per slot (501 GB/s) than the RTX 3090 (~800 GB/s) — more workers, but each
slot is slower.

**3. The SSH tunnel was an unexpected infrastructure bottleneck.**
At 256+ concurrent workers, all HTTP connections multiplexed through a single SSH process
saturated the daemon's per-session channel limit. Failures looked like hardware instability
(all workers collapsing simultaneously), but servers were idle and healthy — the client-side
tunnel had silently stopped forwarding. Instrumentation that watches only GPU utilisation
and LLM latency would have missed this entirely. Fix: one SSH process per GPU, bounding
concurrent channels to `slots_per_GPU` regardless of fleet size or batch size.

**4. The CPU router makes low-volume days free.**
The always-on VPS runs llama.cpp at all times. On days where the estimated batch is under
the configured threshold, inference runs on CPU with no GPU rental. GPU is rented only
when the batch exceeds the threshold — the cost model switches from fixed to on-demand
exactly at the inflection point where GPU becomes faster than waiting.

**5. The Blackwell underperformance had a one-line fix.**
RTX 50xx cards (sm_120a) showed throughput at or below the RTX 3090 through iterations
04–09 — unexpected given a 2.2× memory bandwidth advantage. Root cause confirmed in
iteration 10: the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image is built with
CUDA 12.4, which never triggers the sm_120a compilation gate in CMakeLists.txt. At runtime,
Blackwell falls back to PTX JIT — a conservative codegen that misses sm_120a's native
execution paths. Fix: bump `ARG CUDA_VERSION` to 12.9.1 in `.devops/cuda.Dockerfile`.
Validated across four GPU architectures: **+242% on RTX 5090**, zero regression on
sm_75/sm_86/sm_89. RTX 50xx is now the highest-throughput option per slot in the fleet.
PR candidate submitted to ggml-org/llama.cpp, addressing issues #17822 and #18865.

---

Total GPU spend from initial production design through all case study iterations — including failed runs, timeout experiments, and hardware comparisons across five GPU types: **$7.37**.

---

*(See `docs/case-study.md` for full narrative, `docs/iteration-log.md` for per-iteration metrics.)*
