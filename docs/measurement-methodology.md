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

- **GPU utilization %**: not instrumented. Vast.ai instances don't expose nvtop/nvidia-smi
  output to the client process. Utilization inferred indirectly from throughput vs theoretical.
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
| RTX 3090 | 24 GB GDDR6X | **936 GB/s** | ~71 |
| RTX PRO 6000 WS | 96 GB GDDR6 | ~960 GB/s | ~91 (but lower effective BW/slot) |
| RTX 5090 | 32 GB GDDR7 | ~1,792 GB/s | ~209 |
| RTX 4070S Ti | 16 GB GDDR6X | **672 GB/s** | ~44 |

**Key observation**: RTX PRO 6000 WS and RTX 3090 have similar memory bandwidth
despite a 3× VRAM and TFLOPS gap — consistent with their near-identical measured throughput.
The 5090's anomalous wall time (fixed regardless of worker count) may indicate scheduling
overhead or memory controller saturation at the 47-slot level that bandwidth alone doesn't explain.
