# Iteration Log

> Status: DRAFT — benchmark results filled in as real runs complete.
> Each row corresponds to a directory in `iterations/`.

| # | Name | Change made | Why | Rate before | Rate after | Cost delta | Artifact |
|---|---|---|---|---|---|---|---|
| 00 | Baseline | Sequential single-threaded; hardcoded `avg_inference_secs=30s` | Starting point | 1 rec/min | — | — | [00-baseline/](../iterations/00-baseline/) |
| 01 | Parallel (bug) | Added `ThreadPoolExecutor`; computed `n_parallel` from VRAM but never wrote it back | Enable concurrency | 1 rec/min | ~1 rec/min (bug: still 1 thread) | 0 | [01-parallel-1thread/](../iterations/01-parallel-1thread/) |
| 02 | Bugs fixed | Fixed `n_parallel` writeback + passed `queue_size` to TTL calc | Server had 36 slots, Python used 1; TTL was always 31min | ~1 rec/min | **109 rec/min** | baseline | [02-bugs-fixed/](../iterations/02-bugs-fixed/) |
| 03 | Multi-GPU hardcoded | Hardcoded 2 GPUs; SSH tunnel per GPU; thread-affinity routing | Single GPU VRAM limits parallelism | 109 rec/min | TBD | TBD | [03-multi-gpu-hardcoded/](../iterations/03-multi-gpu-hardcoded/) |
| 04 | Multi-GPU dynamic | Extended VRAM formula: `n_parallel = slots_per_gpu × num_gpus` | Hardcoded count doesn't adapt to actual hardware | TBD | TBD | TBD | [04-multi-gpu-dynamic/](../iterations/04-multi-gpu-dynamic/) |
| 05 | Tiered fallback | Offer search: 8→4→2→1 GPU tiers; RTX filter; BW cost cap | No single tier always has availability | TBD | TBD | TBD | [05-multi-gpu-fallback/](../iterations/05-multi-gpu-fallback/) |
| 06 | 2x RTX 5090 | Tested 2x5090 32GB @ $0.80/hr; wave ramp attempted | High VRAM + high TFLOPS — expected best result | 109 rec/min | 210–275 rec/min (inconsistent) | +$0.50/hr | [06-2x5090/](../iterations/06-2x5090/) |
| 07 | 4x RTX 4070S Ti | 4x16GB bundle @ $0.50/hr — hit via tiered fallback | Fallback found better bandwidth/$ ratio | 210 rec/min | **369 rec/min** | −$0.30/hr vs 5090 | [07-4x4070sti/](../iterations/07-4x4070sti/) |

---

## Hardware comparison (job_skills task)

| GPU | Config | Price/hr | Workers | Rate | Cost/1k records | Verdict |
|---|---|---|---|---|---|---|
| RTX 3070 Mobile | 1x local | free | 4 | ~12 rec/min | $0 | Dev only |
| RTX 3090 | 1x Vast.ai | $0.30 | 36 | 109 rec/min | $0.046 | Good baseline |
| RTX PRO 6000 WS | 1x Vast.ai | >$0.85 | ~180 (theoretical) | ~100 rec/min | >$0.14 | Ruled out — TFLOPS ≠ throughput |
| RTX 5090 | 2x Vast.ai | $0.80 | 94 | 210–275 rec/min | $0.10–0.13 | Inconsistent; discarded |
| **RTX 4070S Ti** | **4x Vast.ai** | **$0.50** | **76** | **369 rec/min** | **$0.023** | **MVP** |
