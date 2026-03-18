# Iteration Log

> Status: DRAFT — benchmark rates for iterations 00–05 are from calibration runs on 1x RTX 3090.
> Iterations 06–07 require real remote runs (see `harness/run_remote.sh`).
> Each row corresponds to a directory in `iterations/`.

---

## Design iterations

| # | Dir | Change | Why | Rate before | Rate after | Notes |
|---|---|---|---|---|---|---|
| 00 | [00-baseline](../iterations/00-baseline/) | Sequential single-threaded; `avg_inference_secs` hardcoded to 30s; 1 worker | Starting point — establish measurement baseline | — | ~1 rec/min | Rate limited by sequential execution, not GPU |
| 01 | [01-parallel-1thread](../iterations/01-parallel-1thread/) | `ThreadPoolExecutor` added; `n_parallel` computed from VRAM formula but never written back to `self.n_parallel` | Parallelism designed but not functional | ~1 rec/min | ~1 rec/min | Bug: llama-server had 36 slots; Python spawned 1 worker |
| 02 | [02-bugs-fixed](../iterations/02-bugs-fixed/) | Two bugs fixed: (1) `n_parallel` write-back; (2) `queue_size` passed to TTL calculator | Bugs masked all gains; TTL always 31min → killed mid-batch | ~1 rec/min | **109 rec/min** | 36 workers, 1x GPU. Jump is entirely from unblocking the existing server capacity |
| 03 | [03-multi-gpu-hardcoded](../iterations/03-multi-gpu-hardcoded/) | 2 GPUs hardcoded; SSH tunnel per GPU port; thread-affinity routing (`thread_id % num_gpus`) | Single GPU VRAM caps parallel slots | 109 rec/min | TBD | Proof-of-concept: multi-GPU routing works |
| 04 | [04-multi-gpu-dynamic](../iterations/04-multi-gpu-dynamic/) | VRAM formula extended: `n_parallel = slots_per_gpu × num_gpus`; `num_gpus` from offer query | Hardcoded GPU count doesn't adapt to actual hardware | TBD | TBD | Slot count now automatically scales to rented card |
| 05 | [05-multi-gpu-fallback](../iterations/05-multi-gpu-fallback/) | Tiered fallback search: 8→4→2→1 GPUs; `LLM_VAST_MIN_TFLOPS_PER_GPU=100`; RTX filter; BW cost cap | No single GPU tier always has availability; wanted high-TFLOPS "quality" cards | TBD | TBD | TFLOPS=100 filter admits only 5090/PRO-6000WS class |
| 06 | [06-first-remote-run](../iterations/06-first-remote-run/) | Same code as 05. First real production run under the strict TFLOPS filter | Validate end-to-end with actual data | 109 rec/min | 210–275 rec/min (inconsistent) | High-TFLOPS card (2x bundle); wall time inconsistent regardless of worker count. Discarded. |
| 07 | [07-four-card-mvp](../iterations/07-four-card-mvp/) | Config change only: `LLM_VAST_MIN_TFLOPS_PER_GPU` 100 → 20. Fallback hit 4x cards. | Per-record perf is near-identical within a GPU family; TFLOPS is the wrong filter. GPU count is the multiplier. | ~210 rec/min | **369 rec/min** | 4x cards, ~76 workers. Best cost-efficiency: ~$0.50/hr total. |

---

## The TFLOPS filter — what it reveals

The jump from iteration 06 to 07 is a single env var change (`LLM_VAST_MIN_TFLOPS_PER_GPU=100 → 20`).

The rationale for 100 TFLOPS/GPU was: higher compute → faster decode. This is the wrong model.
Autoregressive LLM decode is **memory-bandwidth-bound**, not compute-bound. Each token requires
loading the full model weight matrix from VRAM (~2 FLOPs per byte). TFLOPS is headroom for
batched prefill (prompt processing), not for sequential decode.

**What the filter inadvertently showed:**
- At 100 TFLOPS/GPU: only workstation-class and flagship cards pass. Expensive. No better per-slot.
- At 20 TFLOPS/GPU: the full RTX 3xxx/4xxx/5xxx gaming range is eligible.
- Per-record latency within a GPU family (e.g., 3xxx) is nearly flat regardless of tier.
  RTX 3070 Laptop 8GB ≈ RTX 3090 24GB per slot, because the memory controller bottleneck
  is similar across the family once slots are correctly sized to available VRAM headroom.
- The actual throughput lever is **GPU count**: more cards = more parallel slots = higher rec/min.
  4 × any reasonable gaming GPU > 1 × premium card, at lower cost.

Relaxing the filter to 20 TFLOPS/GPU opened the fallback search to 4-card bundles. The 4-card
result (369 rec/min @ ~$0.50/hr) beat the 2-card high-TFLOPS result (≤275 rec/min @ $0.80/hr)
on both throughput and cost.

---

## Hardware observed (job_skills task, 1000 records)

These are the cards that Vast.ai returned for the configured filters — not deliberate targets.
The filter (price, VRAM range, RTX family, download speed, reliability) determines the pool;
the market determines what's in it.

| Config | Price/hr | Workers | Rate | Cost/1k rec | Notes |
|---|---|---|---|---|---|
| 1x (local laptop GPU) | free | 4 | ~12 rec/min | $0 | Dev-only baseline; correct shape, not speed |
| 1x (calibration, ~24GB card) | ~$0.30 | 36 | 109 rec/min | ~$0.046 | Calibration hardware for iterations 00–05 |
| 1x (workstation GPU, high TFLOPS) | >$0.85 | ~100 | ~100 rec/min | >$0.14 | TFLOPS=100 filter. Expensive, no throughput gain. TFLOPS ≠ rec/min. |
| 2x (high-TFLOPS bundle) | ~$0.80 | ~94 | 210–275 rec/min | $0.10–0.13 | Inconsistent wall time. Wave-ramp strategy no help. Discarded. |
| **4x (mid-range bundle)** | **~$0.50** | **~76** | **369 rec/min** | **~$0.023** | **MVP. TFLOPS filter relaxed → opened 4-card range.** |

---

## Key numbers (job_skills task, 1000 records, seed=42)

| Metric | 1-worker bug era | After fix (1x GPU) | After fix (4x GPUs) |
|---|---|---|---|
| Wall time | ~16h | ~9min | ~2m42s |
| Rate | ~1 rec/min | 109 rec/min | 369 rec/min |
| Workers | 1 | 36 | 76 |
| Cost/run | ~$5 | ~$0.046 | ~$0.023 |
