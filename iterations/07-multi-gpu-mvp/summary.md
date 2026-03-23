# 07 — Multi-GPU MVP (4x and 8x fleet, dynamic onstart)

> Status: PARTIALLY PROVEN — 4x results confirmed; 8x ran partially (SSH channel exhaustion at 4m20s); jd_reparse excluded (non-DC thermal instability surfaced).

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

**Actual hardware — 8x run:** RTX 2080 Ti ×8 at $0.590/hr (22 GB/card) — only 1 matching 8x offer.

Two runs attempted:

- **First run**: default `TIMEOUT_LOAD=300s`. All 8 llama-servers timed out before model
  finished loading. 8 simultaneous llama-server startups take longer than the single-GPU
  default allows. No inference reached.

- **Second run**: `TIMEOUT_LOAD=600s`. All 8 servers loaded successfully. Ran for 4m20s,
  processing 747/2000 records, before mass "Connection error." at all workers — the same
  SSH channel exhaustion failure later diagnosed in iter 08. 460.4 rec/min compute rate
  is error-inflated (37% success rate; near-instant failures dominated compute time).

Root cause of second run failure: SSH channel exhaustion, not hardware. The 22 GB RTX 2080 Ti
is compatible with the `ghcr.io/ggml-org/llama.cpp:server-cuda` image (confirmed by
successful model load and inference in both this run and the final iter 09 validation).
The CUDA arch hypothesis was incorrect. See iter 08 for full SSH diagnosis and fix.

## Metrics — job_skills

| Config | Workers | Hardware | Compute rate | Wall rate | Failures | Cost/hr | Cost/1k rec |
|---|---|---|---|---|---|---|---|
| 4x / 1000 rec | 144 | RTX 3090 ×4 | 242.9 rec/min | 125.3 rec/min | 1 | $0.61 | ~$0.05 |
| 8x / 2000 rec (run 1) | — | RTX 2080 Ti ×8 | timeout (300s) | — | — | $0.60 | — |
| 8x / 2000 rec (run 2) | 256 | RTX 2080 Ti ×8 | 460.4 rec/min† | — | 1253/2000 | $0.60 | — |

> † Error-inflated: 37% success rate. Near-instant failures compressed compute time. Real throughput confirmed in iter 09 at 255.9 rec/min after SSH fix.

## Metrics — company_enrich

| Config | Workers | Hardware | Compute rate | Wall rate | Failures | Cost/hr |
|---|---|---|---|---|---|---|
| 4x / 1000 rec | 144 | RTX 3090 ×4 | 506.7 rec/min† | 283.2 rec/min | 0 | $0.61 |
| 8x / 2000 rec | — | — | not attempted | — | — | — |

> † 1000-record sample (records 1–1000 by id). Rate reflects average latency for that
> specific sample. Post-fix 2000-record runs on comparable hardware measured 440–473 rec/min —
> different records, different data distribution, longer sustained load. Not a performance
> regression; different inputs at different scale.

## Metrics — jd_reparse

| Config | Workers | Hardware | Result | Wall time | OK / Total |
|---|---|---|---|---|---|
| 4x / 1000 rec | 144 | RTX 3090 ×4 | excluded — server crash | 5m15s | 192 / 1000 |
| 8x | — | — | not attempted | — | — |

**jd_reparse excluded — connection failure at ~5 min:**

192/1000 records completed before all 144 workers failed simultaneously with
"Connection error." Errors began at ~3m18s compute time / 5m15s wall time.
TTL was set to 37 min — not the cause.

**Initial hypothesis (iter 07): non-DC thermal instability.**
The failure pattern — correlated with fleet TDP and run duration — looked consistent
with thermal collapse on a non-DC host:

| Iter | Hardware | Fleet TDP | Slots | Time to crash |
|---|---|---|---|---|
| 04 | RTX PRO 6000 WS ×1 | ~600W | 99 | ~11 min |
| 07 4x | RTX 3090 ×4 | ~1400W | 144 | ~5 min |

**Correction (iter 08/09): hypothesis falsified.**
Iter 08 ran the same workload on DC hardware (RTX A5000 ×8, $1.51/hr) and failed
identically. Datacenter enforcement was not the fix. Iter 09 ran the same
non-DC hardware with per-GPU isolated SSH tunnels and succeeded at 2000/2000.
The real cause was SSH channel exhaustion — identical to the iter 07 8x job_skills
failure. See iter 08 for full diagnosis. The thermal/TDP correlation was coincidental:
jd_reparse's ~162s/record connection hold-time is what saturates channels, not GPU power.

`config-jd-dc.env` was created under the thermal hypothesis and is retained as an
artifact. `REQUIRE_DATACENTER=true` is not required.

## Findings

**Fleet scaling confirmed through 4x:** 4x RTX 3090 delivers 2.2× the job_skills
compute rate of 1x RTX 3090 from iter 04 (242.9 vs 109 rec/min) at ~2× the cost
($0.61 vs $0.30/hr) — matching the cost-efficiency trend from iter 05.

**Non-DC thermal hypothesis — not confirmed:** The iter 07 hypothesis (non-DC hosts
collapse under sustained multi-GPU load) was later falsified by iter 08 (DC hardware
failed identically) and iter 09 (non-DC hardware succeeded with SSH fix). The real
failure was SSH channel exhaustion. See iter 08.

**Model load timeout must scale with GPU count:** 8 simultaneous llama-server startups
require more time than the default `TIMEOUT_LOAD=300s` allows. The 22 GB RTX 2080 Ti
variant is compatible with `ghcr.io/ggml-org/llama.cpp:server-cuda`; the initial
misdiagnosis (CUDA arch incompatibility) was incorrect. The correct fix: set
`TIMEOUT_LOAD=600` for 8x fleet configs. `config-8x.env` now reflects this.

## Run commands

```bash
# 4x config — completed
bash harness/bench.sh 07-multi-gpu-mvp job_skills     1000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-4x.env
bash harness/bench.sh 07-multi-gpu-mvp company_enrich 1000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-4x.env

# 8x config — two runs captured (timeout failure + SSH failure); SSH fix in iter 08/09
bash harness/bench.sh 07-multi-gpu-mvp job_skills     2000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-8x.env

# jd_reparse — DC config, moved to iter 08
bash harness/bench.sh 07-multi-gpu-mvp jd_reparse     1000 0 "" "" remote "" iterations/07-multi-gpu-mvp/config-jd-dc.env
```

## Artifacts

- `config-4x.env` — 4x bundle offer filters
- `config-8x.env` — 8x bundle offer filters (TIMEOUT_LOAD=600 after model-load timeout on run 1)
- `config-jd-dc.env` — DC-only config for jd_reparse (4x→2x fallback, ≤$0.80/hr)
- `logs/` — keeper logs for completed runs
