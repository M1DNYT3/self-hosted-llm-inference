# Measurement Methodology

## What was measured

### Per-record metrics (emitted inline during every batch run)
```
job {id}  {latency_ms}ms  in={input_tokens} out={output_tokens}  [ok|parse_fail|failed]
```
- **latency_ms**: wall-clock time from HTTP request sent to response received (includes queuing inside llama-server)
- **input_tokens / output_tokens**: reported by llama-server via OpenAI-compatible `/chat/completions` usage field
- **parse_success**: whether the LLM output was valid JSON matching the expected schema (set by caller)

### Per-batch summary (emitted at end of each run)
```
[job_skills] Done. {N}/{total} parsed  time={elapsed}s  rate={rec/min} rec/min
```
- **rate**: `N / (elapsed / 60)` — records successfully parsed per minute
- **elapsed**: wall-clock seconds from first worker dispatch to last result committed

### Per-benchmark (from `benchmark.py`)
Samples N records, runs job_skills task, saves to `benchmark_results.json`:
```json
{
  "timestamp": "...",
  "results": [
    {"job_id": 123, "latency_ms": 2500, "input_tokens": 450, "output_tokens": 150,
     "tokens_per_sec": 12.3, "parse_success": true},
    ...
  ]
}
```
Router reads `benchmark_results.json` to calibrate CPU routing thresholds.

---

## What was NOT measured (gaps)

- **GPU utilization %**: not instrumented in this harness. The case-study harness does not
  include an nvidia-smi monitor. In iteration 10, SM% and power draw were captured by enabling
  the production app's gated diagnostic logger (disabled by default; only used for hardware
  investigations), with output saved into `iterations/10-blackwell-root-cause/logs/*/hardware/`.
  Utilization in iterations 00–09 is inferred indirectly from throughput vs theoretical.
- **VRAM bandwidth utilization**: not directly measured. VRAM bandwidth was used as a
  selection criterion (via `inet_down` proxy for model download and GPU spec sheets for
  inference), not a live metric.
- **p50/p95/p99 latency**: raw `latency_ms` per record is logged but not aggregated into
  percentiles in the current tooling. Can be computed post-hoc from `benchmark_results.json`.
- **Token throughput (tok/s) under load**: `benchmark.py` measures tok/s per record, but
  this is single-threaded (1 worker). Effective tok/s at `n_parallel` workers ≈
  `tok/s_single × n_parallel` (approximately — limited by server-side KV cache contention
  at high slot counts).

---

## How to reproduce

### CPU emulation (no GPU cost)
```bash
docker compose -f docker/compose.yaml up fixture-db -d
bash harness/run_local.sh          # runs job_skills on CPU backend, seeds from workload-contract
```
Expected output: ~12 rec/min (RTX 3070 Mobile equivalent; CPU is ~4 effective slots at 12 tok/s).

### Remote GPU (Vast.ai — real cost)
```bash
# Review first — this rents a real instance
bash harness/run_remote.sh
```
See `harness/workload-contract.yaml` for the exact task, limit, seed, and expected rate.

---

## Hardware specs (reference)

| GPU | VRAM | Memory BW | TFLOPS (FP16) |
|---|---|---|---|
| RTX 3070 Mobile | 8 GB GDDR6 | ~336 GB/s | ~29 |
| RTX 3090 | 24 GB GDDR6X | **~800 GB/s** | ~71 |
| RTX PRO 6000 WS | 96 GB GDDR6X | **~1,400 GB/s** | ~91 (Blackwell; effective BW limited by llama.cpp kernel) |
| RTX 5090 | 32 GB GDDR7 | ~1,792 GB/s | ~209 (Blackwell; bandwidth unrealised — see anomaly below) |
| RTX 4070S Ti | 16 GB GDDR6X | **587 GB/s** | ~44 |

> Memory BW values for RTX 3090 and RTX 4070S Ti are from Vast.ai offer metadata logged
> during benchmark runs (805–827 GB/s across two RTX 3090 instances; 587 GB/s on RTX 4070S Ti).
> PRO 6000 WS and RTX 5090 values are from spec sheets; Mem BW was not logged for those runs
> (feature added after iter 04).

**Key observation**: RTX PRO 6000 WS and RTX 3090 have near-identical measured throughput,
despite the PRO 6000 WS having ~1.75× greater memory bandwidth (~1,400 vs ~800 GB/s). If
throughput were bandwidth-limited and kernels were optimal, the PRO 6000 WS should deliver
~1.75× the per-slot speed. It doesn't. The Blackwell kernel regression below is the only
consistent explanation.

**Confirmed — RTX 5090 / Blackwell kernel regression**: The RTX 5090 (Blackwell, 1,792 GB/s)
and RTX PRO 6000 WS (Blackwell, ~1,400 GB/s) both showed throughput at or below the RTX 3090
(Ampere, ~800 GB/s), despite the 5090's ~2.24× bandwidth advantage over the 3090. Wall time
on the 5090 was fixed regardless of worker count, which rules out queue contention.

**Confirmed root cause (see [iteration 10](../iterations/10-blackwell-root-cause/summary.md))**:
`ghcr.io/ggml-org/llama.cpp:server-cuda` uses `CUDA_VERSION=12.4.0`. A gate in
`ggml/src/ggml-cuda/CMakeLists.txt` adds `sm_120a-real` cubins only when `CUDA_VERSION >= 12.8`. The
official image never triggers the gate — RTX 5090 falls back to PTX JIT at runtime, which
produces a ~3.4× throughput penalty on Blackwell. Two related open issues:

- [#18865](https://github.com/ggml-org/llama.cpp/issues/18865) (JohannesGaessler, Jan 2026):
  CUDA kernel selection in llama.cpp is tuned for RTX 4090 (high-frequency consumer GPU).
  Blackwell GPUs run at lower clock frequencies and require different kernel selection
  (`mmq.cuh` vs `dequant + cuBLAS`). The issue is open; optimization for Blackwell
  architectures is pending.

- [#17822](https://github.com/ggml-org/llama.cpp/issues/17822) (Dec 2025): RTX PRO 6000
  Blackwell (cc 12.0) + **Qwen3** model shows tg512 = 21 tok/s vs 241 tok/s on the same
  hardware for a non-Qwen model — 11× slower token generation, with elevated CPU usage
  despite `gpu-layers=99`. Caused by PTX JIT fallback on sm_120 in images built with CUDA < 12.8.

**Fix**: change `ARG CUDA_VERSION=12.4.0` to `12.9.1` in `.devops/cuda.Dockerfile`.
Validated across sm_75/sm_86/sm_89/sm_120a: +242% on RTX 5090, zero regression elsewhere.
Iteration 10 documents the full investigation, control runs, the vastai detour that nearly
caused a false conclusion, and the final validated fix. [PR #20920](https://github.com/ggml-org/llama.cpp/pull/20920) submitted upstream, addressing #17822 and #18865.

**Implication for hardware selection**: with the fixed image (`m1dnyt3/llama-cpp:server-cuda-12.9`
or equivalent CUDA 12.8+ build), RTX 50xx delivers ~1.7× per-slot throughput vs RTX 3090,
consistent with its ~2.2× memory bandwidth advantage. The hardware selection guidance from
iterations 04–09 (avoid RTX 50xx) is superseded by the fix.
