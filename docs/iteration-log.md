# Iteration Log

> All iterations have completed. Rates marked † are error-inflated (high failure rate causes
> near-instant errors to dominate compute time — not real throughput). See iter 08 for analysis.
> Each row corresponds to a directory in `iterations/`.

---

## Design iterations

| # | Dir | Change | Why | Rate before | Rate after | Notes |
|---|---|---|---|---|---|---|
| 00 | [00-baseline](../iterations/00-baseline/) | Sequential single-threaded; `avg_inference_secs` hardcoded to 30s; 1 worker | Starting point — establish measurement baseline | — | ~1 rec/min | Rate limited by sequential execution, not GPU |
| 01 | [01-parallel-1thread](../iterations/01-parallel-1thread/) | `ThreadPoolExecutor` added; `n_parallel` computed from VRAM but never written to `self.n_parallel` | Parallelism designed but not functional | ~1 rec/min | ~1 rec/min | Bug: llama-server had 36 slots; Python spawned 1 worker |
| 02 | [02-bugs-fixed](../iterations/02-bugs-fixed/) | Two bugs fixed: (1) `n_parallel` write-back; (2) `queue_size` passed to TTL formula | Bugs masked all parallel capacity; TTL always 31min → killed mid-batch | ~1 rec/min | **109 rec/min** | 36 workers, 1x GPU. Jump is entirely from unblocking existing server capacity |
| 03 | [03-first-remote-run](../iterations/03-first-remote-run/) | First real Vast.ai run: 200 records, 1x RTX 3090, 36 slots | Validate end-to-end pipeline at production hardware scale | 109 rec/min | 109 rec/min | Scale problem confirmed: 1 GPU saturates regardless of card tier |
| 04 | [04-single-gpu-scaling](../iterations/04-single-gpu-scaling/) | Multi-config exploration: RTX PRO 6000 WS (96GB, 99-slot), RTX 5090 (32GB, 47-slot) | Test whether higher TFLOPS or more VRAM improves per-slot speed | 109 rec/min | ~100 rec/min | Per-slot perf flat across all cards. PRO 6000: same rate at 2.8× cost. TFLOPS ≠ throughput. |
| 05 | [05-multi-gpu-hardcoded](../iterations/05-multi-gpu-hardcoded/) | 2 GPUs hardcoded; SSH tunnel per port; thread-affinity routing (`thread_id % num_gpus`) | Single GPU VRAM caps slot count; multi-GPU is the scaling path | 109 rec/min | ~219 rec/min | First multi-GPU run (2x RTX 5090). Static dispatch beats shared-queue at this scale. |
| 06 | [06-multi-gpu-dynamic](../iterations/06-multi-gpu-dynamic/) | `onstart` script generated dynamically from `num_gpus`; VRAM formula: `n_parallel = slots_per_gpu × num_gpus` | Hardcoded GPU count doesn't adapt to rented hardware | ~219 rec/min | ~219 rec/min | Code-only refactor; no new benchmarks. Enables arbitrary GPU count at runtime. |
| 07 | [07-multi-gpu-mvp](../iterations/07-multi-gpu-mvp/) | TFLOPS filter 100→20/GPU; RTX family + bandwidth filter; 4x→8x fallback | TFLOPS=100 too restrictive; bandwidth is the real bottleneck; fleet count is the throughput multiplier | ~219 rec/min | **243 rec/min (4x)** | 4x RTX 3090 confirmed; 8x startup failure (CUDA arch); jd_reparse SSH failure surfaced |
| 08 | [08-ssh-throughput-bottleneck](../iterations/08-ssh-throughput-bottleneck/) | Diagnose mass connection failures; switch from 1 shared SSH process to per-GPU isolated processes; add `reconnect_on_error` hook | Single SSH tunnel exhausted under 256+ concurrent workers; failure looked like hardware but was client-side | 460† rec/min | — | Root cause: SSH daemon channel limit. All iter 08 runs failed. Fix implemented, validated in 09. |
| 09 | [09-conclusion](../iterations/09-conclusion/) | Validate SSH fix at 2000 records across all 3 tasks; 3 extra 4x runs for fleet comparison | Confirm SSH fix resolves all failure modes; characterise 4x vs 8x trade-off | — | **256 rec/min (8x)** | 8x RTX 2080 Ti: all tasks stable. 4x RTX 3090: jd_reparse 6% failures (long-duration channel pressure). |

---

## The TFLOPS filter — what it reveals

The jump in iter 07 is a filter change (`LLM_VAST_MIN_TFLOPS_PER_GPU=100 → 20`).

The rationale for 100 TFLOPS/GPU was: higher compute → faster decode. This is the wrong model.
Autoregressive LLM decode is **memory-bandwidth-bound**, not compute-bound. Each token requires
loading the full model weight matrix from VRAM (~2 FLOPs per byte). TFLOPS is headroom for
batched prefill, not for sequential decode.

**What the filter inadvertently showed:**
- At 100 TFLOPS/GPU: only workstation-class cards pass. Expensive. No better per-slot throughput.
- At 20 TFLOPS/GPU: full RTX 3xxx/4xxx/5xxx gaming range is eligible.
- Per-record latency within a GPU family is nearly flat regardless of tier.
  RTX 3070 Laptop 8GB ≈ RTX 3090 24GB per slot — same bandwidth-bound bottleneck.
- The actual throughput lever is **GPU count**: more cards = more parallel slots = higher rec/min.
  4 × any reasonable gaming GPU > 1 × premium card, at lower cost.

---

## The SSH tunnel — an infrastructure-layer failure

The failure in iters 07–08 looked like a hardware problem (all workers failing simultaneously
at ~3 min) but was a client-side SSH channel exhaustion. Key evidence:

- Servers drained gracefully and went idle — not an OOM or crash
- Instance remained SSH-accessible throughout
- SSH daemon logs: `connect_to localhost port 8000: failed`
- Failure reproduced on DC hardware at 50% GPU utilisation — hardware was not the variable
- Error-inflated compute rates: iter 07 8x shows 460 rec/min at 33% success; post-fix shows
  255 rec/min at 99.6% success. The pre-fix rate was not real throughput.

**Fix**: one SSH process per GPU. Peak concurrent channels per process = `slots_per_GPU`
(independent of total GPU count and batch size). The `reconnect_on_error` hook on
`BaseLLMBackend` provides a runtime recovery path for transient failures.

---

## Hardware observed (job_skills, all confirmed runs)

| Config | Hardware | Workers | Compute rate | Wall rate | Cost/hr | Cost/1k rec |
|---|---|---|---|---|---|---|
| 1x local | RTX 3070 Mobile | 4 | ~12 rec/min | ~12 rec/min | $0 | $0 |
| 1x remote | RTX 3090 | 36 | 109 rec/min | ~73 rec/min | ~$0.30 | ~$0.041 |
| 1x remote | RTX PRO 6000 WS | ~99 | ~100 rec/min | — | >$0.85 | >$0.14 |
| 2x remote | RTX 5090 | 94 | 219 rec/min | — | ~$0.73 | ~$0.056 |
| 4x remote | RTX 4070S Ti | 76 | **369 rec/min** | ~228 rec/min | ~$0.50 | **~$0.023** |
| 4x remote | RTX 3090 | 144 | 208.8 rec/min | 156.9 rec/min | $0.642 | ~$0.068 |
| 8x remote | RTX 2080 Ti | 256 | 255.9 rec/min | 175.6 rec/min | $0.598 | ~$0.057 |

---

## Key numbers summary

| Metric | 1-worker bug era | After fix (1x GPU) | Best 4x (4070S Ti) | Final 8x (iter 09) |
|---|---|---|---|---|
| job_skills rate | ~1 rec/min | 109 rec/min | **369 rec/min** | 256 rec/min |
| Cost per 1k records | ~$5 | ~$0.046 | **~$0.023** | ~$0.057 |
| 10k records wall time | ~16h | ~92 min | ~44 min | ~57 min |
| Cost per 10k records | ~$50 | ~$0.46 | ~$0.37 | ~$0.57 |
