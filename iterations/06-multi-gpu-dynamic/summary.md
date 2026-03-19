# 06 — Multi-GPU Dynamic (generated onstart, any GPU count)

## Change

Replaced the two hardcoded `llama-server` launch blocks in the `onstart` script with a
loop generated at rent time from `num_gpus`.

**Before (iter 05 — hardcoded):**
The `onstart` script contained two explicit server launch blocks written as literal strings:

```bash
env CUDA_VISIBLE_DEVICES=0 "$LLAMA_BIN" --port 8000 --parallel 51 --ctx-size 208896 ...
env CUDA_VISIBLE_DEVICES=1 "$LLAMA_BIN" --port 8001 --parallel 51 --ctx-size 208896 ...
```

Adding a third GPU required editing two places in the source: the `onstart` string and
the `num_gpus` constant. The two were coupled by hand.

**After (iter 06 — dynamic):**
`_build_onstart(n_parallel_per_gpu)` loops over `range(self._num_gpus)`, emitting one
server block per GPU:

```python
for i in range(self._num_gpus):
    port = 8000 + i
    cmd = (
        f'env CUDA_VISIBLE_DEVICES={i} "$LLAMA_BIN" \\\n'
        f'    --port {port} --parallel {n_parallel_per_gpu} ...'
    )
```

The VRAM formula was already multi-GPU-aware from iter 05
(`n_parallel = slots_per_gpu × num_gpus`), so no slot-count logic changed.

## Why

Iter 05 proved that 2× GPUs produce near-linear throughput scaling (~2.4×) at better
cost-efficiency per record ($0.056 vs $0.097). The bottleneck now is the hardcoded
GPU count: renting a 4x or 8x bundle would require a code edit before deployment.

The dynamic `_build_onstart` removes that ceiling. The `_find_offer` tiered-fallback
(8→4→2→1) already existed; now it can actually use whatever count it lands on —
the correct `onstart` is generated at rent time, not at code-write time.

## Metrics

No benchmark run for this iteration — the generated `onstart` output for
`num_gpus=2` is byte-identical to iter 05's hardcoded script. All numbers carry
forward unchanged from iter 05.

## Artifacts

- No `benchmark_results.json` — design change only
- No `config.env` — no new hardware configuration introduced

## What this unblocks

Iteration 07 rents 4× and 8× bundles without any code change. The only inputs that
vary per run are `LLM_VAST_NUM_GPUS` and the matching offer filters in `config.env`.
