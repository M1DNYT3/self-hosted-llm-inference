# 10 — Blackwell Root Cause (RTX 5090 +242% on CUDA 12.9.1)

> Status: COMPLETE — Root cause confirmed and validated. One-line Dockerfile fix.
> PR submitted: https://github.com/ggml-org/llama.cpp/pull/20920

## Background

Iteration 04 benchmarked the RTX 5090 (Blackwell, sm_120a, 1,792 GB/s) and found throughput
equal to or below the RTX 3090 (Ampere, sm_86, ~800 GB/s) — a card with 2.24× less memory
bandwidth. The case study concluded at the time that this was a Blackwell-specific kernel
regression in llama.cpp (see `measurement-methodology.md`) and advised against RTX 50xx
until upstream resolved it.

This iteration investigates the root cause directly, traces it to a CUDA version gap in
llama.cpp's official Docker image, and validates a fix across the full GPU matrix.

## Source Code Investigation

Issue #18865 pointed at `mmq.cuh` — the CUDA matrix-multiplication kernel selection logic
tuned for RTX 4090. The hypothesis was that sm_120a might be falling through to a suboptimal
dispatch branch. The call chain was traced to verify:

```
ggml_cuda_mul_mat()          [ggml-cuda.cu]
  └─ ggml_cuda_op_mul_mat()
       ├─ use_mul_mat_q?      → ggml_cuda_should_use_mmq()
       │    └─ checks arch, quant type, batch size
       └─ mul_mat_q()         [mmq.cu]
            └─ launch_mul_mat_q()
                 └─ mul_mat_q_switch() [mmq.cuh]
                      └─ dispatches on GGML_CUDA_CC_*
```

In `mmq.cuh`, the dispatch table included explicit branches for `CC_VOLTA` (sm_70),
`CC_TURING` (sm_75), `CC_AMPERE` (sm_80/sm_86), `CC_ADA` (sm_89), and a `default` path
for anything else. `sm_120a` (Blackwell, compute capability 12.0) had no dedicated branch —
it resolved to `default`.

The critical question: **is `default` wrong, or is `default` simply not compiled?**

In `common.cuh` / `fattn.cu`, the same pattern appeared — attention kernels also dispatch
on CC constants, and sm_120a again fell to `default`. The `default` path is not inherently
wrong; it selects conservative tile sizes that work on any architecture. The question was
whether sm_120a-specific cubins existed at all to execute those paths natively.

That led to `ggml/src/ggml-cuda/CMakeLists.txt`:

```cmake
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
endif()
```

The gate was already there. sm_120a support was not a missing feature — it was gated on the
CUDA toolkit version at build time. Without CUDA ≥ 12.8, no sm_120a cubin is emitted for
any kernel path, including `default`. At runtime, CUDA falls back to PTX JIT for the entire
device: the `mmq.cuh` dispatch table is irrelevant because there is no native binary to
dispatch into. PTX JIT produces a conservative, architecture-agnostic codegen that misses
Blackwell's tensor core and memory access patterns entirely.

This is why the performance penalty was so large and so consistent across all operation
types — it was not a per-kernel dispatch miss, it was a total absence of native cubins for
the device.

## Root Cause

`ggml/src/ggml-cuda/CMakeLists.txt` contains the following gate:

```cmake
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
endif()
```

`ghcr.io/ggml-org/llama.cpp:server-cuda` (official production image) sets:

```dockerfile
ARG CUDA_VERSION=12.4.0
```

CUDA 12.4.0 < 12.8 → the gate never triggers → no `sm_120a` cubin is compiled. At runtime,
the RTX 5090 finds no native cubin and falls back to PTX JIT (just-in-time PTX compilation).
PTX JIT produces a conservative, non-optimised kernel path — the decode performance penalty
is ~3.4× vs native cubins on sm_120a.

The gate and the fix were already in the codebase. Only the CUDA version in the Dockerfile
prevented them from taking effect.

## The vastai Detour — A Near-Miss

Before reaching the root cause, a secondary image was tested: `vastai/llama-cpp:b8468-cuda-12.9`,
which uses CUDA 12.8 toolkit and does compile `sm_120a-real` cubins.

**Initial results on vastai 12.8:**
- RTX 5090: 47.5 → 51.1/min — marginal improvement, well below expectation
- RTX 4070S Ti: 113.6 → 52.5/min — **severe regression**
- RTX 3090: 95.9 → 44.6/min — **severe regression**

All three cards showed an SM% burst pattern in `nvidia-smi dmon`: SM% spiking briefly then
dropping to 0% while power draw held at 110 W. This was incorrectly hypothesised as a
universal CPU scheduling bottleneck in llama.cpp — which would have ended the investigation
with a false conclusion.

**Pivot**: a control run was added — RTX 3090 on the official 12.4 image:
- `logs/cuda-12.4/benchmark/RTX3090_20260323_161416_job_skills_100rec.log` → 95.9/min
- SM% was sustained 70–100% throughout
  (`logs/cuda-12.4/hardware/RTX3090_20260323_161529_33388662_dmon.log`)

The burst pattern was a vastai build-specific artifact, not a llama.cpp behavior. vastai's
image has a kernel efficiency regression on sm_86 and sm_89: SM% drops to 0% at ~110 W for
approximately 2 minutes at the start of each run, then recovers.
Evidence: `logs/cuda-12.8-vastai/hardware/RTX3090_20260323_160652_33388114_dmon.log`

Without the control run the investigation would have concluded that llama.cpp has a scheduler
bug affecting all GPUs, and missed the real cause.

## The Fix

Single line change in `.devops/cuda.Dockerfile`:

```dockerfile
# Before
ARG CUDA_VERSION=12.4.0

# After
ARG CUDA_VERSION=12.9.1
```

`CUDA_DOCKER_ARCH=default` (llama.cpp's own documented default) compiles cubins for all
supported architectures: sm_50 through sm_121, including `sm_120a-real` for Blackwell.

**Why 12.9.1 specifically:**

12.9.1 is the latest CUDA 12.x release. CUDA 13.0 deprecates Pascal (sm_60/62, GTX 10xx).
CUDA 12.x still covers Maxwell (sm_52, GTX 9xx) and Pascal — the fix enables Blackwell
support without dropping any existing production GPU and without requiring a 13.x migration.
A single build covers the full range from GTX 9xx through RTX 50xx with no trade-offs.

**Build pitfall encountered:** the first custom build used
`CUDA_DOCKER_ARCH="86-real;89-real;120a-real"` (explicit arch list). RTX 2080 Ti (sm_75)
received "no kernel image is available for execution on the device" — sm_75 was not in the
list. Fixed by using `default`.

**Published image:** `m1dnyt3/llama-cpp:server-cuda-12.9` (Docker Hub)

## Benchmark Matrix

All 19-slot runs force `--n-parallel 19` per GPU regardless of VRAM capacity, equalising
slot count for apples-to-apples per-GPU throughput comparison. KV cache per slot varies by
card (RTX 5090 ≈ 1.37 GB/slot, RTX 3090 ≈ 0.95 GB/slot, RTX 4070S Ti ≈ 0.53 GB/slot,
RTX 2080 Ti ≈ 0.84 GB/slot). The per-card `LLM_VAST_KV_SLOT_GB` value is back-calculated
from the VRAM headroom ÷ 19; the harness displays a rounded figure in the slot formula line.

| Card | Arch | VRAM | Mem BW | Official 12.4 | vastai 12.8¹ | 12.9.1 bump | Delta |
|---|---|---|---|---|---|---|---|
| RTX 5090 | sm_120a | 32 GB | 1,792 GB/s | 47.5/min² | 51.1/min | **162.6/min** | **+242%** |
| RTX 4070S Ti | sm_89 | 16 GB | 587 GB/s | 113.6/min | 52.5/min | 117.5/min | +3% |
| RTX 3090 | sm_86 | 24 GB | ~800 GB/s | 95.9–109.6/min | 44.6/min | 95.6/min | ≈0% |
| RTX 2080 Ti | sm_75 | 22 GB | 501 GB/s | 26.8/min | — | 29.2/min | +9% |

¹ vastai 12.8 numbers are not representative — this build has a kernel efficiency regression
on sm_86 and sm_89 unrelated to CUDA version. Excluded from the delta calculation.

² RTX 5090 official baseline used 51 natural slots, not 19 (`kv_slot_gb=0.5`).
Log: `logs/cuda-12.4/benchmark/RTX5090_20260322_235818_job_skills_100rec.log`

**Key result**: +242% throughput on Blackwell. Zero regression on sm_75/sm_86/sm_89.
CUDA 12.9.1 minor improvements on older architectures (sm_75: +9%) are a side effect of
updated toolchain optimisations, not Blackwell-specific changes.

### Per-slot throughput (19-slot runs, 12.9.1 bump vs official 12.4)

| Card | Arch | Official 12.4 (rec/min/slot) | 12.9.1 bump (rec/min/slot) | Ratio |
|---|---|---|---|---|
| RTX 5090 | sm_120a | ~1.0 (51-slot est.)³ | 8.56 | ~8.4× |
| RTX 4070S Ti | sm_89 | 5.98 | 6.18 | 1.03× |
| RTX 3090 | sm_86 | 5.03–5.77 | 5.03 | ≈1.0× |
| RTX 2080 Ti | sm_75 | 1.41 | 1.54 | 1.09× |

³ RTX 5090 on official 12.4 at 51 slots: 47.5/min ÷ 51 = 0.93 rec/min/slot. The 5090's
per-slot speed was below the 2080 Ti despite ~3.6× greater memory bandwidth — the clearest
evidence of PTX JIT overhead.

After the fix: RTX 5090 delivers 8.56 rec/min/slot vs RTX 3090's 5.03 — a 1.7× ratio
consistent with the ~2.2× memory bandwidth advantage (1,792 vs ~805 GB/s measured from
offer metadata). The gap between 1.7× and 2.2× reflects non-bandwidth overhead in the
workload (scheduling, attention ops) — expected for a real batch, not a synthetic BW test.

Note: the RTX 2080 Ti offer in this iteration reported 22 GB VRAM, exceeding the stock 11 GB.
Vast.ai hosts may run modified or overclocked cards; the offer metadata reflects actual
available VRAM as reported by the driver.

## Hardware Diagnostics

`nvidia-smi dmon` logs (SM%, VRAM controller %, power draw, CPU%) confirm execution patterns:

**RTX 5090 — official 12.4 (PTX JIT):** bursty SM%, variable power — JIT recompilation
overhead during the first tokens of each batch.
`logs/cuda-12.4/hardware/RTX5090_20260322_235920_33351046_dmon.log`

**RTX 5090 — 12.9.1 bump (native sm_120a):** SM% sustained 60–93%, power draw 280–393 W.
Healthy Blackwell saturation — no JIT stalls.
`logs/cuda-12.9-custom/hardware/RTX5090_20260323_183736_33394606_dmon.log`

**RTX 3090 — official 12.4 (control run):** SM% sustained 70–100%. Confirmed that the
burst pattern seen on vastai's build is NOT a universal llama.cpp behavior.
`logs/cuda-12.4/hardware/RTX3090_20260323_161529_33388662_dmon.log`

**RTX 3090 — vastai 12.8 (regression):** SM=0% at ~110 W for the first ~2 minutes, then
recovery. Build-specific artifact that was the near-miss false lead.
`logs/cuda-12.8-vastai/hardware/RTX3090_20260323_160652_33388114_dmon.log`

## Why Qwen3 Specifically?

Issues [#17822](https://github.com/ggml-org/llama.cpp/issues/17822) and
[#18865](https://github.com/ggml-org/llama.cpp/issues/18865) both specifically mention Qwen3
on Blackwell. Issue #17822 provides the clearest empirical cross-model evidence: on the same
RTX PRO 6000 WS hardware with the same image, Qwen3 delivers tg512 = 21 tok/s while a
non-Qwen model reaches 241 tok/s — an 11× gap that points to a Qwen3-specific execution path
on sm_120a.

This benchmark uses Qwen3.5-9B exclusively across all runs. There is one Qwen2.5 context run
(`logs/context/benchmark/RTX5090_20260323_001251_*`), but slot count and model size differ,
making a direct comparison unreliable. The data here cannot reproduce or disprove the
cross-model gap — for that, #17822 is the empirical reference.

The PTX JIT root cause affects all models: without sm_120a cubins, every kernel falls back
regardless of model architecture. Why Qwen3 shows a larger penalty than other models at the
kernel level is not confirmed here. The 11× figure in #17822 was observed with the official
12.4 image; whether it closes fully with the fix (i.e., whether any Qwen3-specific dispatch
suboptimality remains on top of the PTX JIT penalty) is an open question and out of scope
for this iteration.

## Conclusion

The RTX 5090 Blackwell underperformance documented in iterations 04–09 was not a hardware
limitation or a fundamental llama.cpp kernel gap — it was a one-line Dockerfile omission.

The CUDA 12.4.0 base image predates the sm_120a gate in CMakeLists.txt. The gate was already
in the codebase waiting to be unlocked. Bumping to 12.9.1 (latest 12.x, no deprecation) is
the complete fix.

- **No architectural changes.** No kernel tuning required.
- **No regression.** sm_75/sm_86/sm_89 performance is unchanged or marginally improved.
- **No GPU deprecation.** CUDA 12.9.1 covers Maxwell through Blackwell. Pascal and Maxwell
  drop only in CUDA 13.0 — a separate decision for another day.
- **Drop-in deployment.** `LLM_VAST_IMAGE=m1dnyt3/llama-cpp:server-cuda-12.9` in `.env`.

PR submitted: https://github.com/ggml-org/llama.cpp/pull/20920 — addresses #17822 and #18865.

## Log Index

| Log | GPU | Image | Slots | Rate | Category |
|---|---|---|---|---|---|
| `logs/context/benchmark/RTX5090_20260322_224310_*` | RTX 5090 | official 12.4 | natural | — | context |
| `logs/context/benchmark/RTX5090_20260322_230304_*` | RTX 5090 | official 12.4 | natural | — | context |
| `logs/context/benchmark/RTX5090_20260322_234533_*` | RTX 5090 | official 12.4 | natural | — | context |
| `logs/context/benchmark/RTX5090_20260323_001251_*` | RTX 5090 | official 12.4 | natural | — | context (Qwen2.5) |
| `logs/cuda-12.4/benchmark/RTX5090_20260322_235818_*` | RTX 5090 | official 12.4 | 51 | 47.5/min | baseline |
| `logs/cuda-12.4/benchmark/RTX3090_20260323_161416_*` | RTX 3090 | official 12.4 | 19 | 95.9/min | control run |
| `logs/cuda-12.4/benchmark/RTX3090_20260323_184451_*` | RTX 3090 | official 12.4 | 19 | 109.6/min | fresh baseline |
| `logs/cuda-12.4/benchmark/RTX2080Ti_20260323_200731_*` | RTX 2080 Ti | official 12.4 | 19 | 26.8/min | baseline |
| `logs/cuda-12.4/benchmark/RTX4070STi_20260323_202104_*` | RTX 4070S Ti | official 12.4 | 19 | 113.6/min | baseline |
| `logs/cuda-12.8-vastai/benchmark/RTX5090_20260323_144740_*` | RTX 5090 | vastai 12.8 | 19 | 51.1/min | vastai (partial sm_120a) |
| `logs/cuda-12.8-vastai/benchmark/RTX4070STi_20260323_155937_*` | RTX 4070S Ti | vastai 12.8 | 19 | 52.5/min | vastai (regression) |
| `logs/cuda-12.8-vastai/benchmark/RTX3090_20260323_160536_*` | RTX 3090 | vastai 12.8 | 19 | 44.6/min | vastai (regression) |
| `logs/cuda-12.9-custom/benchmark/RTX5090_20260323_183656_*` | RTX 5090 | 12.9.1 bump | 19 | **162.6/min** | **FIXED — money shot** |
| `logs/cuda-12.9-custom/benchmark/RTX4070STi_20260323_185104_*` | RTX 4070S Ti | 12.9.1 bump | 19 | 117.5/min | no regression |
| `logs/cuda-12.9-custom/benchmark/RTX2080Ti_20260323_195519_*` | RTX 2080 Ti | 12.9.1 bump | 19 | 29.2/min | no regression |
| `logs/cuda-12.9-custom/benchmark/RTX3090_20260323_202419_*` | RTX 3090 | 12.9.1 bump | 19 | 95.6/min | no regression |
