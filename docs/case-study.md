# Inference Engineering Case Study: Self-Hosted LLM at Scale

> **Artifact index**
> - `inference/` — sanitized inference layer (backends, pipeline, router, prompts)
> - `fixture/` — self-contained Postgres fixture DB (Docker)
> - `iterations/00–09/` — one directory per design iteration; each contains config, logs, summary
> - `docs/iteration-log.md` — per-iteration metrics table
> - `docs/measurement-methodology.md` — how rates and costs were measured

---

## 1. Problem Statement

A production data pipeline enriches job market records daily across three LLM tasks:

| Task | Input | Output |
|---|---|---|
| `job_skills` | Job title + description | Required / preferred skill sets |
| `company_enrich` | Company name + description | Enrichment signals and scores |
| `jd_reparse` | Raw job description | Structured section fields (multi-slice) |

At peak load: up to 10,000 records per task per day. A local benchmark on an RTX 3070 Mobile
put the API cost at $10–50 per 10k-record batch — $50–150/month at daily cadence. Beyond cost,
managed API providers introduce three additional risks: per-token pricing that scales with data
volume, model deprecation without warning, and third-party data exposure.

**The decision**: self-host inference on spot GPU rental (Vast.ai) using llama.cpp, routed
through a CPU-first fallback on an already-running VPS. This inverts the cost structure:
instead of paying per token, pay per compute-hour — and only when the batch is large enough
to justify it.

**Target**: sub-$0.50/day at typical load. Expected daily volumes: job_skills ~2–4k
(peaks at ~6–8k on busy scrape days), company_enrich ~2k (degrading toward zero over
time as the company universe saturates), jd_reparse matches daily new records. The 10k
figure is an absolute worst-case planning ceiling, not the expected steady-state.
Worst-case single-task cost: ~$0.57 (8x fleet, ~8k records). Average day: well under $0.50.

---

## 2. Architecture

```
                      ┌─────────────────────────────────┐
  daily cron          │         pick_backend()           │
  ─────────────►      │  estimate CPU wall time          │
                      │  if time ≤ threshold → CPU       │
                      │  else → rent GPU                 │
                      └────────────┬────────────┬────────┘
                                   │            │
                          CPU path │            │ GPU path
                                   ▼            ▼
                         ┌──────────────┐  ┌──────────────────────────┐
                         │ llama.cpp    │  │ Vast.ai GPU instance     │
                         │ port 8085    │  │ tiered fallback 8→4→2→1  │
                         │ always-on    │  │ TTL-bounded, destroyed    │
                         │ VPS ($0 GPU) │  │ after batch completes    │
                         └──────────────┘  └──────────────────────────┘
```

**Model**: Qwen3.5-9B Q4_K_M. 6 GB VRAM footprint. `--reasoning-budget 0` disables chain-of-thought
(suppresses ~300 tokens/call that add latency without improving structured JSON output).

**Slot formula**: `n_parallel = floor((vram_gb - model_vram_gb) / kv_slot_gb)`. Model weights
are a fixed overhead (`model_vram_gb`; 6 GB for Qwen3.5-9B Q4_K_M). The remaining VRAM is
divided into KV-cache slots. `kv_slot_gb` is the configurable variable: ~0.5 GB per slot at
4096-token context for this model. A 24 GB card at `kv_slot_gb=0.5` yields 36 slots; doubling
VRAM to 48 GB yields 84 — super-linear because the model weight overhead is fixed.

`kv_slot_gb` is a deliberate tuning parameter, not a constant. It is increased (fewer slots)
when bandwidth is the bottleneck: adding slots beyond bandwidth headroom causes individual
slots to compete for the same memory bus, degrading per-slot throughput without increasing
aggregate throughput. Different models have different KV-cache footprints at the same context
length, and there is no reliable way to detect the optimal value before a live load test —
so it is measured empirically and set per model/context configuration.

**GPU rental**: tiered fallback searches 8→4→2→1 GPU bundles at configured price/VRAM/bandwidth
filters. The cheapest matching offer is rented, a TTL is computed from queue size and
calibrated `avg_inference_secs`, and the instance is destroyed on completion.

**SSH tunnels**: one `ssh -N` process per GPU, each forwarding one local port to the
corresponding llama-server inside the container. Workers are thread-affinity routed via
`thread_id % num_gpus` to distribute load across servers.

---

## 3. Iteration Journey

### Bug-fix phase (iters 00–02): unblocking existing capacity

The baseline was a single-threaded sequential loop — one request at a time, hardcoded 30s
estimate per record, no parallelism. Two bugs compounded to keep it that way through iter 01:

**Bug 1 — n_parallel write-back** ([01-parallel-1thread](../iterations/01-parallel-1thread/)):
`ThreadPoolExecutor` was added and `n_parallel` was computed correctly from the VRAM formula,
but the result was never written to `self.n_parallel`. The threadpool always spawned 1 worker.
The llama-server sat idle with 35 unused slots.

**Bug 2 — TTL zero** ([02-bugs-fixed](../iterations/02-bugs-fixed/)):
`startup(queue_size=0)` was always called with a hardcoded `0` regardless of actual batch size.
The TTL formula produced ~31 minutes in every case. Instances were destroyed mid-batch on
runs longer than 31 minutes.

Fixing both in iter 02 unlocked the server's full capacity immediately. Throughput jumped
from ~1 rec/min to **109 rec/min** with no hardware change — the same 1x RTX 3090 that had
been doing nothing.

### TFLOPS trap (iters 03–05): what the filter revealed

Single-GPU scaling was explored across three hardware configurations
([04-single-gpu-scaling](../iterations/04-single-gpu-scaling/)):

| Card | VRAM | TFLOPS | Slots | Rate | Cost/hr | Rate/$ |
|---|---|---|---|---|---|---|
| RTX 3090 | 24 GB | 35.6 | 36 | 109 rec/min | $0.30 | 363 |
| RTX PRO 6000 WS | 96 GB | 91.1 | 99 | ~100 rec/min | >$0.85 | <118 |
| RTX 5090 | 32 GB | 209.5 | 47 | ~92 rec/min | ~$0.37 | ~249 |

A TFLOPS filter at `min_tflops_per_gpu=100` was added as a quality gate, reasoning that
higher compute → faster decode. This is the wrong model.

Autoregressive LLM decode is **memory-bandwidth-bound**. Every token generated requires
reading the full model weight tensor from VRAM (~2 FLOPs per byte). The roofline bottleneck
is memory bandwidth, not arithmetic throughput. Workstation cards are TFLOPS-optimised for
rendering and simulation; gaming cards are bandwidth-optimised for rasterisation — a different
axis entirely.

The RTX PRO 6000 WS (91 TFLOPS, 96 GB VRAM) matched the RTX 3090's per-slot throughput at
2.8× the cost. High VRAM was useful for slot count, but per-slot decode speed was identical.
The TFLOPS filter was selecting for the wrong dimension.

### Fleet scaling (iters 05–07): GPU count is the lever

Multi-GPU was first implemented with a hardcoded 2-GPU setup ([05-multi-gpu-hardcoded](../iterations/05-multi-gpu-hardcoded/)),
then generalised to dynamic `num_gpus` from the offer query ([06-multi-gpu-dynamic](../iterations/06-multi-gpu-dynamic/)).

The TFLOPS filter was then relaxed from 100 to 20/GPU ([07-multi-gpu-mvp](../iterations/07-multi-gpu-mvp/)).
This opened the search to the full RTX 3xxx/4xxx/5xxx gaming range. The first fallback hit
4x cards:

| Config | Workers | Rate | Cost/hr | Cost/1k |
|---|---|---|---|---|
| 1x RTX 3090 | 36 | 109 rec/min | $0.30 | $0.046 |
| 4x RTX 3090 (iter 07) | 144 | 242.9 rec/min | $0.61 | ~$0.050 |
| 4x RTX 4070S Ti | 76 | 237.5 rec/min | $0.703 | ~$0.057 |

The 4x RTX 3090 result (iter 07 confirmed run) delivers 2.2× the throughput of 1x at
2× the cost — consistent with the sub-linear fleet-scaling trend from iter 05. The per-slot
decode speed was identical across card tiers; GPU count was the multiplier.

A supplemental run on 4x RTX 4070S Ti (76 workers, 587 GB/s/card, $0.703/hr) confirmed
similar throughput: 237.5 rec/min at ~$0.057/1k — matching the 8x RTX 2080 Ti on cost per
record at half the GPU count. Full results in iter 09 supplemental (instances 33290545,
33290984, 33291148).

Iter 07 also surfaced two new failure modes:
1. **8x model-load timeout**: the first 8x run (RTX 2080 Ti ×8, 22 GB/card) timed out
   at the default 300s — 8 simultaneous llama-server startups take longer than the
   single-GPU default allows. A second run at `TIMEOUT_LOAD=600s` loaded successfully
   and ran for 4m20s before hitting SSH channel exhaustion (460† rec/min, 37% success).
   The initial misdiagnosis was CUDA arch incompatibility; the 22 GB 2080 Ti variant is
   compatible with the image. Fix: `TIMEOUT_LOAD=600` for 8x configs.
2. **jd_reparse failure at 5 min**: all 144 workers disconnected simultaneously. Initially
   attributed to non-DC thermal instability. The real cause was identified in iter 08.

### SSH tunnel bottleneck (iter 08): not all inference failures come from inference

Four runs at 2000 records across different hardware (DC/non-DC, RTX A5000, RTX 4090,
RTX 2080 Ti) all failed the same way: mass `Connection error.` at ~3 minutes,
regardless of GPU tier, price, or datacenter flag.

Post-failure SSH inspection showed servers were alive, idle, and processing nothing.
`ps aux` showed all llama-server processes running. The SSH daemon log contained the
actual error: `connect_to localhost port 8000: failed`.

**Root cause**: the original tunnel design opened one `ssh -N` process carrying all GPU
port-forwards as `-L` arguments on a single TCP connection. With 256–280 workers all
holding open HTTP connections simultaneously, the SSH daemon's per-session channel limit
was saturated. New forwarding requests were rejected. The Python workers received
connection refused on retry, logged `Connection error.`, and stopped sending work.
The servers drained their in-flight slots and went idle. The application had no way to
distinguish "tunnel is gone" from "server is down."

The DC hypothesis (tested at $1.51/hr on RTX A5000 ×8 at 50% utilisation) was disproved
by the data: DC hardware failed identically to non-DC. This was a client-side failure,
not a hardware or network problem. The misleadingly high pre-fix compute rates
(460–705 rec/min) were error inflation: with 30–60% of records failing instantly (~0ms),
they dominated compute time and made the pipeline look faster than it was.

**Fix** ([inference/backends/vastai.py](../inference/backends/vastai.py)):
one SSH process per GPU. Peak concurrent channels per SSH session is bounded to
`slots_per_GPU` — independent of total GPU count and batch size. Adding more GPUs adds
more processes, not more channels per process.

**Two-layer protection**:
- Layer 1 (structural): per-GPU processes prevent channel saturation from occurring
- Layer 2 (runtime): `reconnect_on_error()` hook on `BaseLLMBackend` restarts dead
  tunnels mid-batch and retries the failed request once — transparently to the worker

The hook is a `False`-returning no-op by default; any backend overrides it to opt into
the retry path. CPU and local backends ignore it; only Vast.ai implements reconnection.

### Validation (iter 09): final results

Post-fix runs at 2000 records, all three tasks, two fleet configurations:

**8x RTX 2080 Ti @ $0.598/hr** (instance IDs: 33161914, 33162723, 33164209)

| Task | OK / Total | Compute rate | Wall rate | Cost/1k |
|---|---|---|---|---|
| job_skills | 1992 / 2000 | 255.9 rec/min | 175.6 rec/min | ~$0.057 |
| company_enrich | 2000 / 2000 | 475.1 rec/min | 253.5 rec/min | ~$0.039 |
| jd_reparse | 2000 / 2000 | 94.8 rec/min | 81.6 rec/min | ~$0.122 |

**4x RTX 3090 @ $0.642/hr** (instance IDs: 33165227, 33167859, 33168411)

| Task | OK / Total | Compute rate | Wall rate | Cost/1k |
|---|---|---|---|---|
| job_skills | 1994 / 2000 | 208.8 rec/min | 156.9 rec/min | ~$0.068 |
| company_enrich | 2000 / 2000 | 440.7 rec/min | 328.3 rec/min | ~$0.033 |
| jd_reparse | 1878 / 2000 | 77.5 rec/min | 62.7 rec/min | ~$0.149 |

The 4x jd_reparse failures (6.1%) were caused by a retry-before-confirm race in the
`reconnect_on_error` hook: the hook restarted the tunnel then retried the request
immediately, without confirming the tunnel was ready to forward connections. Under
sustained long-duration load (jd_reparse, 144 workers, ~162s/record), the retry hit
a tunnel that had restarted but not yet completed the SSH handshake. The hook has since
been corrected: after restarting a dead tunnel process, it now probes
`http://localhost:{port}/v1/health` with exponential backoff until the endpoint responds
before returning control to the worker. The 8x run (32 channels/process) had fewer
failures because lower per-process channel pressure reduced how often the hook fired.

**Fleet comparison (job_skills, wall time)**:

| Config | Workers | Wall rate | Est. 10k wall time | Est. 10k cost |
|---|---|---|---|---|
| 1x RTX 3090 | 36 | 73 rec/min | ~92 min | ~$0.46 |
| 4x RTX 3090 | 144 | 156.9 rec/min | ~64 min | ~$0.68 |
| 8x RTX 2080 Ti | 256 | 175.6 rec/min | ~57 min | ~$0.57 |
| 4x RTX 4070S Ti | 76 | 204.0 rec/min | ~49 min | ~$0.58 |

The 4x RTX 4070S Ti is the fastest single-run option at 10k records and matches the 8x
cost per run — at half the GPU count. Scaling from 4x 3090 to 4x 4070S Ti is not explained
by worker count (76 vs 144) but by per-slot speed: the 4070S Ti's shorter avg_latency per
record (~18.5s on job_skills vs ~25s+ on the 3090 instances in iter 09) means each of its
fewer workers finishes faster. 8x 2080 Ti is 14% slower than 4x 4070S Ti despite having
3.4× the workers — because RTX 2080 Ti bandwidth (~501 GB/s) is the lowest in the fleet.

---

## 4. Key Findings

### Finding 1 — TFLOPS is not the bottleneck for autoregressive decode

Every token in autoregressive generation requires a full read of the model weight matrix
from VRAM. This is an $O(\text{model\_params})$ memory load per token. At 6 GB model
size (Q4_K_M), each token touches ~6 GB of data. TFLOPS describes arithmetic capacity;
memory bandwidth describes how fast data moves from storage to compute. For this workload,
the bottleneck is always bandwidth.

**Empirical evidence**: RTX PRO 6000 WS (91 TFLOPS, $0.85+/hr) matched RTX 3090
(36 TFLOPS, $0.30/hr) at identical per-slot throughput. The TFLOPS gap is 2.5×; the
throughput gap is ~0%. The PRO 6000 WS has ~1.75× the memory bandwidth of the 3090
(~1,400 GB/s vs ~800 GB/s) — it should be ~1.75× faster per slot. It isn't. Root cause
confirmed in iteration 10: the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image is
built with CUDA 12.4, which never triggers the `sm_120a` compilation gate in CMakeLists.txt.
At runtime, Blackwell falls back to PTX JIT — a conservative codegen that misses sm_120a's
native tensor core and memory access paths entirely. Fix: bump `ARG CUDA_VERSION` to 12.9.1
in `.devops/cuda.Dockerfile`. Validated: +242% on RTX 5090, zero regression on all other
architectures (see `docs/measurement-methodology.md` and
`iterations/10-blackwell-root-cause/summary.md`).

### Finding 2 — GPU count is the true throughput multiplier

Once the slot formula is correct (`n_parallel = (vram - model_vram) / kv_slot_gb`),
each GPU contributes independently. There is no cross-GPU scheduling overhead for
separate llama-server processes on separate ports. Scaling is sub-linear: 4× the GPU count
delivers approximately 2.2× throughput, not 4×. The gap comes from two sources — queue
management overhead at high worker counts, and per-slot speed differences when cards
from different generations have different memory bandwidth per slot.

**Empirical evidence**: 1x → 4x RTX 3090 (iter 07, confirmed run) produced 2.2× throughput
(109 → 242.9 rec/min) at 2× the cost. A supplemental run on 4x RTX 4070S Ti produced 2.2×
throughput (109 → 237.5 rec/min) at 2.3× the cost — consistent with the scaling law across
different GPU generations at comparable bandwidth-per-dollar.

### Finding 3 — Infrastructure-layer failures can masquerade as inference failures

The SSH channel exhaustion failure was diagnosed as hardware instability across two
iterations, a DC vs non-DC investigation, and four separate runs on three different
GPU models before the real cause was identified.

The diagnostic signal was the mismatch between what the failure looked like and what
the evidence showed: if this were hardware failure, the servers would crash (OOM, process
exit, dmesg errors). Instead they drained and idled. The SSH daemon log — one layer below
the application, not watched by any GPU or LLM monitoring — held the actual error.

At sufficient scale, bottlenecks appear in parts of the system that are not the focus
of optimisation: the tunnel process model, SSH channel multiplexing, per-session daemon
limits. Standard inference instrumentation (GPU utilisation, LLM latency, token throughput)
would not have surfaced this.

### Finding 4 — The CPU router makes low-volume days free

The `pick_backend()` function estimates whether the current batch can be completed by
the always-on CPU server within a per-task threshold (configurable; defaults: 10h for
job_skills, 2h for company_enrich, 1h for jd_reparse). Below the threshold, no GPU
is rented. The CPU server on an already-paid VPS has zero marginal inference cost.

This inverts the API pricing model in the most useful direction: the marginal cost at
low volume is $0, and at high volume it scales with time (GPU hours), not with data
volume (tokens). A day with 200 records costs nothing; a day with 10k records costs
~$0.50–0.70.

---

## 5. Full Cost Model

### API baseline (before)

| Volume | Est. API cost | Monthly (daily cadence) |
|---|---|---|
| 10k records/day | $10–50/run | $50–150/month |
| 2k records/day | $2–10/run | $10–50/month |
| < threshold | $0.20–2/run | $6–30/month |

### Self-hosted (after)

| Volume | Backend | Est. cost | Monthly |
|---|---|---|---|
| < CPU threshold | CPU (VPS) | **~$0** | **~$0** (fixed VPS overhead) |
| ~2k records | 4x GPU | ~$0.15–0.20/run | ~$4–6/month |
| 10k records | 8x GPU | ~$0.57/run | **~$17/month** |
| 10k records (worst) | 4x GPU | ~$0.68/run | **~$20/month** |

Self-hosted cost is **$15–20/month worst case vs $50–150/month API** — a 3–7× reduction,
with fixed monthly cost instead of volume-proportional billing.

---

## 6. What I'd Do Differently

**1. Start with fleet scaling, not single-GPU tuning.**
The single-GPU experiments (iter 04) confirmed that per-slot speed is flat across GPU tiers.
That conclusion was available from the architecture: bandwidth-bound decode means TFLOPS is
irrelevant and per-slot speed equals bandwidth/model_size. The empirical confirmation was
useful but could have come from a single run rather than three hardware configurations.

**2. Instrument the SSH layer from the start.**
Adding process-level health monitoring (poll `_tunnel_procs`, log when a process exits
unexpectedly) would have surfaced the channel exhaustion failure immediately as "tunnel died"
rather than "connection error" — saving two full iterations of misattribution.

**3. Use `KEEP_ON_FAILURE=true` earlier.**
The flag that preserved the instance for SSH inspection was added in iter 08. Adding it
in iter 07 (when jd_reparse first crashed) would have allowed direct examination of the
SSH daemon log on the first occurrence, not the fourth.

**4. Test jd_reparse in isolation at small batch size before fleet runs.**
jd_reparse is structurally different: multi-slice, ~162s per record, sustained full-GPU
load. It should have had its own calibration run before being included in the fleet
scaling tests.

---

## 7. AI Disclosure — How LLM Assistance Was Used

This study was developed with Claude (Anthropic) as an active collaborator. In the interest
of transparency:

**What AI was used for:**
- Translating benchmark observations and logs into written iteration summaries, this case
  study, and supporting docs. Observations were always provided by the engineer; Claude
  shaped them into readable form.
- Hypothesis validation — working through ambiguous failure evidence (SSH vs hardware,
  error inflation vs real throughput drop) and confirming or challenging the engineer's
  initial read. Conclusions were reached together; the evidence and diagnostic intuition
  were the engineer's.
- Generating design alternatives with tradeoffs (tunnel isolation strategy, retry hook
  structure, backend-specific vs generic reconnection). The engineer evaluated options
  and typically chose hybrids.
- Drafting initial code implementations of agreed designs (`reconnect_on_error` hook,
  per-GPU tunnel list, `_call_api` split). AI-generated code is treated as a draft:
  the engineer reviews, corrects, and owns it before it reaches production.

**What AI was not used for:**
- Problem definition, architecture decisions, hardware selection, cost model design.
- Running benchmarks or collecting data.
- Identifying failure modes or generating the diagnostic hypotheses. Those came from the
  engineer observing the logs.
- Final code correctness. One AI draft contained a bug in `reconnect_on_error` — a
  retry-before-confirm race where the hook returned control to the worker before
  confirming the tunnel was ready. The engineer identified the flaw, specified the correct
  flow (restart → probe health endpoint → confirm → retry), and applied the fix directly
  in Python.

The inference engine was developed over approximately one month as part of a production
application. This case study captures and documents one week of optimization work from
that system.
