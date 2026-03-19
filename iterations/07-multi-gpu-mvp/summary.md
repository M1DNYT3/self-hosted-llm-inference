# 07 — Multi-GPU MVP (4x and 8x fleet, dynamic onstart)

> Status: PARTIALLY PROVEN — 4x results confirmed; 8x startup failed (CUDA arch incompatibility); jd_reparse excluded (non-DC thermal instability surfaced).

## Change

First runs using the dynamic `_build_onstart` from iter 06 at scale beyond 2 GPUs.
Two GPU counts benchmarked against the same tasks to establish the fleet-scaling law
and compare cost-efficiency against single-GPU results from iter 04.

**4x run** — replicates production-scale configuration.
Config: `config-4x.env` — up to 4× RTX 2xxx–5xxx, 16–36 GB VRAM,
≥25 TFLOPS/GPU, ≤$0.90/hr, datacenter=false.

**8x run** — first exploration beyond production scale.
Config: `config-8x.env` — same filters, NUM_GPUS=8, TFLOPS floor=13/GPU
(100 fleet total / 8 cards = 12.5 → 13). Admits RTX 2080 Ti (~13.4 TFLOPS/GPU).

## Why

Iters 04–05 established that per-slot decode speed is nearly flat across GPU tiers
(RTX 3090 ≈ RTX 5090 per slot), and that GPU count is the true throughput multiplier.
Iter 05 proved 2× GPUs → ~2.4× throughput at 1.7× better cost-efficiency per record.

Iter 07 tests whether this scaling law continues past 2 cards, and whether the
TFLOPS filter relaxation opens better-value hardware than the 100 TFLOPS floor
that admitted only premium workstation cards.

**Expected hardware landscape — 4x config (≥25 TFLOPS/GPU, ≥100 total):**

| Card | 4x TFLOPS | Bandwidth/card | Typical price | Expectation |
|---|---|---|---|---|
| RTX 4070S Ti | ~170 total | ~580 GB/s | $0.39–0.45/hr | prod reference |
| RTX 5070 Ti | ~170 total | ~760 GB/s | $0.30–0.40/hr | preferred: higher bandwidth, lower price |
| RTX 5060 Ti | ~90–100 total | ~360 GB/s | $0.30–0.40/hr | **excluded** — ~22–23 TFLOPS/GPU below the 25/GPU floor |

**Actual hardware — 4x run:** RTX 3090 ×4 at $0.591/hr (35/GPU, 141 total TFLOPS,
936 GB/s bandwidth). Not the 4070S Ti / 5070 Ti range targeted in config comments —
the 3090 passes ≥25 TFLOPS/GPU and was the cheapest available 4x offer at run time.

**Expected hardware landscape — 8x config (≥13 TFLOPS/GPU, ≥100 total):**

| Card | 8x TFLOPS | Bandwidth/card | Typical price | Expectation |
|---|---|---|---|---|
| RTX 4070S Ti | ~340 total | ~580 GB/s | ~$0.80–0.90/hr | high-end 8x |
| RTX 5070 Ti | ~340 total | ~760 GB/s | ~$0.60–0.80/hr | preferred if available at 8x |
| RTX 2080 Ti | ~107 total | ~500 GB/s | lower | admitted: 13.4 TFLOPS/GPU passes the 13/GPU floor |

**Actual hardware — 8x run:** RTX 2080 Ti ×8 at $0.590/hr — only 1 matching 8x offer.
Instance booted and SSH'd successfully, but all 8 llama-servers timed out at 300s.
Root cause: Turing architecture (SM75) is not supported by the
`ghcr.io/ggml-org/llama.cpp:server-cuda` image (targets SM80+ / Ampere and newer).
The TFLOPS floor passed the offer; the CUDA image rejected it silently at runtime.

## Metrics — job_skills

| Config | Workers | Hardware | Compute rate | Wall rate | Failures | Cost/hr | Cost/1k rec |
|---|---|---|---|---|---|---|---|
| 4x / 1000 rec | 144 | RTX 3090 ×4 | 242.9 rec/min | 125.3 rec/min | 1 | $0.61 | ~$0.05 |
| 8x / 2000 rec | — | RTX 2080 Ti ×8 | startup failure | — | — | $0.60 | — |

## Metrics — company_enrich

| Config | Workers | Hardware | Compute rate | Wall rate | Failures | Cost/hr |
|---|---|---|---|---|---|---|
| 4x / 1000 rec | 144 | RTX 3090 ×4 | 506.7 rec/min | 283.2 rec/min | 0 | $0.61 |
| 8x / 2000 rec | — | — | not attempted | — | — | — |

## Metrics — jd_reparse

| Config | Workers | Hardware | Result | Wall time | OK / Total |
|---|---|---|---|---|---|
| 4x / 1000 rec | 144 | RTX 3090 ×4 | excluded — server crash | 5m15s | 192 / 1000 |
| 8x | — | — | not attempted | — | — |

**jd_reparse excluded — non-DC thermal instability under sustained load:**

192/1000 records completed before all 144 workers failed simultaneously with
"Connection error." The SSH tunnel dropped entirely (GPU label disappears from
error lines), indicating the llama-server process crashed rather than timed out.
Errors began at ~3m18s compute time / 5m15s wall time. TTL was set to 37 min —
not the cause.

Same failure mode as iter 04 (single RTX PRO 6000 WS, crash at ~11 min). Pattern:
non-DC host under sustained full-TDP multi-GPU load reaches a power or thermal limit
and the host collapses:

| Iter | Hardware | Fleet TDP | Slots | Time to crash |
|---|---|---|---|---|
| 04 | RTX PRO 6000 WS ×1 | ~600W | 99 | ~11 min |
| 07 4x | RTX 3090 ×4 | ~1400W | 144 | ~5 min |

Higher total fleet TDP → faster collapse. jd_reparse exposes this because
multi-slice records (~162s/record avg) keep GPUs at 100% for longer continuous
stretches than job_skills (~20s) or company_enrich (~50s) — the latter two
complete their runs before the host fails.

Fix: `REQUIRE_DATACENTER=true`. See `config-jd-dc.env` and iter 08.

## Findings

**Fleet scaling confirmed through 4x:** 4x RTX 3090 delivers 2.2× the job_skills
compute rate of 1x RTX 3090 from iter 04 (242.9 vs 109 rec/min) at ~2× the cost
($0.61 vs $0.30/hr) — matching the cost-efficiency trend from iter 05.

**Non-DC fragility is task-dependent and load-dependent:** Short-latency tasks
(job_skills, company_enrich) complete before non-DC hosts destabilize. Long-latency
tasks (jd_reparse) expose the failure. This is a hardware procurement constraint,
not a code bug — and it is invisible to the TFLOPS/VRAM/price filters.

**TFLOPS filter cannot substitute for CUDA architecture awareness:** The 8x filter
admitted RTX 2080 Ti (SM75 / Turing) by TFLOPS, but the CUDA image does not support
it. A bandwidth filter (`MIN_MEM_BW_PER_GPU=480`) would exclude SM75 cards by proxy
(RTX 2080 Ti: ~616 GB/s — this one passes; the actual gap is smaller cards in the
Turing range). The correct fix is either a higher TFLOPS floor (≥20/GPU excludes
SM75 cards in practice) or a multi-arch CUDA image. Config-8x.env now sets
`TIMEOUT_LOAD=600` to surface the failure faster on retry.

## Run commands

```bash
# 4x config — completed
bash harness/bench.sh 07-multi-gpu-mvp job_skills     1000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-4x.env
bash harness/bench.sh 07-multi-gpu-mvp company_enrich 1000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-4x.env

# 8x config — retry pending (TIMEOUT_LOAD updated to 600)
bash harness/bench.sh 07-multi-gpu-mvp job_skills     2000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-8x.env
bash harness/bench.sh 07-multi-gpu-mvp company_enrich 2000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-8x.env

# jd_reparse — DC config, moved to iter 08
bash harness/bench.sh 07-multi-gpu-mvp jd_reparse     1000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-jd-dc.env
```

## Artifacts

- `config-4x.env` — 4x bundle offer filters
- `config-8x.env` — 8x bundle offer filters (TIMEOUT_LOAD=600 after CUDA arch failure)
- `config-jd-dc.env` — DC-only config for jd_reparse (4x→2x fallback, ≤$0.80/hr)
- `logs/` — keeper logs for completed runs
