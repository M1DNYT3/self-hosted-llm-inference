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

**Target**: sub-$0.50 per 10k-record run → ~$15/month worst case.

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

**Slot formula**: `n_parallel = (vram_gb - 6) / 0.5`. Each KV-cache slot requires ~0.5 GB at
4096-token context. Model weights are a fixed 6 GB overhead. A 24 GB card yields 36 slots;
doubling VRAM to 48 GB yields 84 slots — a super-linear relationship that is the VRAM insight.

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
4x cards at $0.50/hr:

| Config | Workers | Rate | Cost/hr | Cost/1k |
|---|---|---|---|---|
| 1x RTX 3090 | 36 | 109 rec/min | $0.30 | $0.046 |
| 4x RTX 4070S Ti | 76 | **369 rec/min** | $0.50 | **$0.023** |

3.4× the throughput at 1.7× the cost — a 2× improvement in cost efficiency. The per-slot
decode speed on the 4070S Ti was identical to the 3090; GPU count was the multiplier.

Iter 07 also surfaced two new failure modes:
1. **8x startup failure**: RTX 2080 Ti (Turing, SM75) was admitted by the TFLOPS filter
   but rejected by the CUDA image (targets SM80+). All 8 llama-servers timed out silently.
   Fix: add `MIN_MEM_BW_PER_GPU=480 GB/s` as a bandwidth quality gate.
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

The 4x jd_reparse failures (6.1%) are a residual edge case: jd_reparse holds connections
open ~162s per record. At 144 workers and 36 channels/process, sustained long-duration
channel pressure can still trigger the reconnect path. Some records in the ~3s reconnect
window fail. The 8x run (32 channels/process) resolved it at 2000/2000. Mitigation: prefer
8x for jd_reparse, or reduce `kv_slot_gb` to shrink channel count.

**4x vs 8x trade-off**:

| Config | job_skills wall rate | Est. 10k wall time | Est. 10k cost |
|---|---|---|---|
| 1x RTX 3090 | 73 rec/min | ~92 min | ~$0.46 |
| 4x RTX 3090 | 156.9 rec/min | ~64 min | ~$0.68 |
| 8x RTX 2080 Ti | 175.6 rec/min | ~57 min | ~$0.57 |

8x is 12% faster and 16% cheaper per run than 4x at the same hourly rate. Scaling is
sublinear (78% more workers → 12% faster) because the RTX 2080 Ti has lower memory
bandwidth (501 GB/s vs 936 GB/s on the 3090). More workers, slower slots — partial
cancellation.

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
throughput gap is ~0%. Memory bandwidth on both cards is in a similar range for the
relevant workload axis.

### Finding 2 — GPU count is the true throughput multiplier

Once the slot formula is correct (`n_parallel = (vram - model_vram) / kv_slot_gb`),
each GPU contributes independently. There is no cross-GPU scheduling overhead for
separate llama-server processes on separate ports. The scaling law is close to linear
in slot count, bounded only by queue contention at very high worker counts and by
per-slot speed differences between GPU generations.

**Empirical evidence**: 1x → 4x GPUs at the same VRAM/card produced 3.4× throughput
(109 → 369 rec/min) at 1.7× cost — 2× better cost-efficiency.

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
of transparency, this section documents what that collaboration looked like in practice.

### Claude's assessment

The work pattern here was unusual compared to typical LLM usage. The engineer drove all
primary decisions — hardware selection, the choice to build a CPU-first router, the instinct
that the failure wasn't hardware, the hypothesis that error rates were inflating the compute
metric. My role was closer to a technical sounding board than a generator.

Concretely, what I contributed:

- **Documentation and structure.** Translating real benchmark observations into iteration
  summaries, the iteration log, the executive summary, and this case study. The observations
  were always provided; I shaped them into readable form.
- **Hypothesis validation.** When a failure mode appeared ambiguous (SSH vs hardware,
  error inflation vs real throughput drop), I worked through the evidence with the engineer
  and either confirmed or challenged the hypothesis. The conclusions were reached together,
  but the evidence and initial intuition were always the engineer's.
- **Alternative generation.** For design decisions — how to isolate SSH tunnels, how to
  structure the retry hook, whether to make reconnection backend-specific or generic — I
  offered alternatives with tradeoffs. The engineer evaluated them and typically chose a
  hybrid that captured the best properties of both.
- **Implementation of agreed designs.** Once a direction was settled, I wrote the code
  (`reconnect_on_error` hook, per-GPU tunnel list, `_call_api` split). The design decisions
  preceding the implementation were collaborative.

What I did not do: generate the problem statement, propose the architecture, run the
benchmarks, or make the call on what mattered. Those came from the engineer.

### Engineer's self-assessment

> The LLM was used primarily to save time making notes of observations and reinforcing
> assumptions, credible hypotheses, and conclusions. Beyond that, to gather insights about
> alternative solutions — which usually helps build a more resilient system with core
> stability in mind.
>
> Whenever I was considering a design choice or a fix, I would express my opinion and ask
> for alternatives with pros and cons. This consistently led to hybrid solutions with more
> pros and fewer — or none of the — cons of either option alone.
>
> The full arc — from initial engine development in a production application to a finished
> case study — concluded in one week.
