# 08 — SSH Throughput Bottleneck (single-tunnel design fails at scale)

> Status: ROOT CAUSE IDENTIFIED — SSH tunnel channel exhaustion under high worker counts.
> Fix implemented: per-GPU isolated SSH processes. Validated in iter 09.

## What happened

Four `job_skills` runs at 2000 records across different hardware and dispatch strategies.
All four failed with mass `Connection error.` around the 3–5 minute mark, regardless of:
- Hardware (DC vs non-DC, RTX A5000 vs RTX 4090 vs RTX 2080 Ti)
- Dispatch strategy (shared-queue vs static-batches)
- GPU count (4x vs 8x)
- Price tier ($0.60/hr vs $1.50/hr)

The servers survived every run. Post-failure `ps aux` via SSH showed all llama-server
processes running normally. The servers drained their in-flight slots and went idle —
they never OOM'd, never crashed, and never logged errors. The instance itself remained
reachable via SSH throughout.

**The error was on the client side, not the GPU side.**

## Root cause — SSH tunnel channel exhaustion

The original tunnel design opened one `ssh -N` process carrying all GPU port forwards
as `-L` arguments on a single TCP connection:

```
ssh -N \
  -L port0:localhost:8000 \  # GPU 0
  -L port1:localhost:8001 \  # GPU 1
  ...
  -L portN:localhost:800N \  # GPU N
  root@ssh_host
```

Every Python worker holds one open HTTP connection to the server while waiting for
inference to complete. With 256–280 workers, all 256–280 TCP connections are
multiplexed as forwarding channels through that single SSH session. The SSH daemon's
per-session channel limit is reached; it begins rejecting new port-forward requests
with `connect_to localhost port 8000: failed`. The Python workers receive connection
refused on their next retry, log `Connection error.`, and stop sending requests.
The servers, seeing no more incoming requests, drain gracefully and go idle.

The application has no awareness that the tunnel killed itself — it continues
dispatching work to URLs that are no longer reachable.

**Evidence:**
- All workers fail simultaneously at the same wall-clock point (~3 min)
- SSH daemon logs: `connect_to localhost port 8000: failed`
- `ps aux` post-failure: all 8 llama-server processes still running, 0% CPU (idle)
- Servers log graceful slot releases, no panics, no OOM
- Failure reproduces across DC and non-DC hardware → hardware is not the variable
- Failure reproduces across shared-queue and static-batches → dispatch is not the variable

## Why iter 07 (4x runs) did not show this

4x RTX 3090 with 24 GB VRAM: `(24 - 6) / 0.5 × 4 = 144 workers`. Under the SSH
daemon's per-session channel limit. 8x pushed it to 256–280 workers, over the limit.

The iter 07 jd_reparse crash (5 min, 4x, 144 workers) was attributed to non-DC
thermal instability — that hypothesis was wrong. jd_reparse keeps workers connected
for ~162s per record, so 144 workers hold 144 SSH channels open for much longer
continuous stretches than job_skills (~20s) or company_enrich (~50s), which
complete their runs before channel pressure accumulates.
**Duration under load, not hardware class, determined when the limit was reached.**

## Prior misattribution — DC hypothesis discarded

Iter 07 and the initial iter 08 hypothesis attributed failures to non-DC host
thermal/power collapse. That was wrong:
- DC hardware (RTX A5000 ×8, $1.51/hr, datacenter=True) failed identically
- DC hardware was at ~50% VRAM and TFLOPS utilization — nowhere near thermal stress
- The failure mode (graceful drain, not crash) is inconsistent with host collapse
- Connection errors appear in SSH daemon logs, not in llama-server or dmesg

The DC config produced the most expensive run ($1.51/hr) with the worst outcome
(725/2000 = 36% completion) — disproving the DC-stability hypothesis by counterexample.

## Runs in this iteration

| Run | Hardware | Dispatch | Workers | OK / Total | Compute rate | Wall time | Cost/hr | Notes |
|---|---|---|---|---|---|---|---|---|
| 20260319_183037 | RTX 4090 ×4 (DC) | shared-queue | 140 | 885 / 2000 | 634.3 rec/min | 7m14s | $1.34 | Old RTX filter; 4x instead of 8x |
| 20260319_185036 | — | — | — | — | — | — | — | Aborted before startup |
| 20260319_201240 | RTX A5000 ×8 (DC) | static-batches | 280 | 725 / 2000 | 705.2 rec/min | 7m06s | $1.51 | DC + highest workers → worst failure rate |
| 20260319_202310 | RTX 2080 Ti ×8 (non-DC) | shared-queue | 256 | 674 / 2000 | 414.6 rec/min | 8m13s | $0.60 | Non-DC; same failure pattern |

All four runs fail at the same phase for the same reason. DC flag, GPU tier, dispatch
strategy, and price bracket are orthogonal to this failure.

## The fix — per-GPU isolated SSH processes

One SSH process per GPU, each forwarding one port. Peak concurrent channels per
SSH session = `slots_per_GPU = (VRAM - model_VRAM) / kv_slot_gb` — independent of
GPU count and batch size:

| Fleet | Channels/process (old) | Channels/process (fixed) |
|---|---|---|
| 1×24GB | 36 | 36 (no change) |
| 4×24GB | 144 | 36 |
| 8×24GB | 288 → over limit | 36 |
| 8×22GB | 256 → over limit | 32 |

Adding more GPUs adds SSH processes, not channels per process. Batch size (2k → 10k)
extends duration without increasing peak concurrent channel count. The fix scales
correctly across all fleet/VRAM/batch combinations.

Implemented in `inference/backends/vastai.py`: `_tunnel_proc` → `_tunnel_procs: list`.
All per-GPU processes start in parallel, are checked after 5s, and are terminated
independently on shutdown.

## Two-layer protection

The fix is applied at two levels, each independent of the other:

**Layer 1 — structural fix (per-GPU SSH processes):** Eliminates the failure mode
entirely under normal operating conditions. Channel count per session is bounded by
`slots_per_GPU`, which does not grow with fleet size or batch size. This is the
primary fix and should prevent the bottleneck from ever being reached again.

**Layer 2 — runtime fallback (`reconnect_on_error` hook):** If a tunnel dies mid-batch
for any reason (transient network event, SSH daemon restart, OS signal), `complete()`
detects the failure, calls `reconnect_on_error()`, restarts the dead process(es), and
retries the request once — transparently to the worker and the pipeline. The record
is not lost; no manual intervention is needed.

The two layers are intentionally decoupled. Layer 1 prevents the problem; Layer 2
recovers from it if it somehow occurs anyway. Neither depends on the other.

**Extensibility:** `reconnect_on_error` is a hook on `BaseLLMBackend` that returns
`False` by default. Any future backend that uses a reconnectable transport (a different
cloud provider, a VPN tunnel, a proxy) opts into the retry behavior by overriding the
hook — no changes to `complete()`, the pipeline, or the worker threads required.

## Why this matters

The natural assumption when an LLM inference pipeline fails is to look at the model,
the GPU, or the network between client and host. This failure did none of those things.

The root cause was a local port-forwarding channel limit inside the SSH session on the
client machine — a layer below the application, invisible to the inference code, and
unrelated to GPU utilization, model behavior, or external network quality. The instance
was healthy. The servers were healthy. The model was loaded and responding. From the
application's perspective it was simply making HTTP requests to localhost.

The failure was unintuitive precisely because every observable signal pointed away from
it: the servers drained gracefully (not an OOM or crash), the instance stayed up
(not a thermal collapse), the network was fine (SSH itself remained connected). The only
artifact was a daemon-level log line: `connect_to localhost port 8000: failed`.

It illustrates that at sufficient scale, bottlenecks can appear in the infrastructure
scaffolding — the tunnel, the process model, the channel multiplexing — rather than
in the compute layer that is the focus of tuning. Instrumentation that only watches
GPU utilization and LLM latency would have missed this entirely.

## Artifacts

- `config.env` — non-DC config used for final two runs (after DC hypothesis discarded)
- `logs/` — four job_skills runs showing the bottleneck across hardware and dispatch variants
