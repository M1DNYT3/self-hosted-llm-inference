#!/usr/bin/env python3
# harness/workload_driver.py
"""Deterministic benchmark workload driver.

Connects to the fixture DB, runs one task against the inference layer,
and writes a benchmark_results.json with per-record metrics + batch summary.

Usage:
  python harness/workload_driver.py \\
      --task job_skills \\
      --limit 1000 \\
      --db-url postgresql://fixture:fixture@localhost:5433/inference_fixture \\
      --backend remote \\
      --seed 42 \\
      --output iterations/07-4x4070sti/benchmark_results.json

  # Local CPU emulation (no GPU cost):
  python harness/workload_driver.py \\
      --task job_skills \\
      --limit 100 \\
      --db-url postgresql://fixture:fixture@localhost:5433/inference_fixture \\
      --backend-url http://localhost:8085/v1 \\
      --model Qwen3.5-9B-Q4_K_M

Reproducibility:
  The fixture DB always contains the same records in the same order (ORDER BY id ASC).
  Given the same --limit and --seed, the workload is identical across runs.
  --seed is recorded in the output JSON but does not currently affect DB query order;
  it is reserved for future random-sampling extensions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Allow running from case-study/ or case-study/harness/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load_env(path: str = ".env", override: bool = False) -> None:
    """Load key=value pairs from .env into os.environ (minimal dotenv).

    override=False (default): setdefault — existing env vars win.
    override=True: force-set — file values win (used for per-iteration config.env).
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if override:
                    os.environ[key] = val
                else:
                    os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


def _build_backend(args: argparse.Namespace) -> "BaseLLMBackend":  # type: ignore[name-defined]
    """Instantiate the appropriate backend from CLI args."""
    import inference.backends  # noqa: F401 — trigger @register() side-effects
    from inference.base import BaseLLMBackend
    from inference.config import (
        LLM_API_KEY,
        LLM_CONTEXT_WINDOW_COMPANY_ENRICH,
        LLM_CONTEXT_WINDOW_JD_REPARSE,
        LLM_CONTEXT_WINDOW_JOB_SKILLS,
        LLM_MODEL,
        LLM_VAST_AVG_INFERENCE_SECS_COMPANY_ENRICH,
        LLM_VAST_AVG_INFERENCE_SECS_JD_REPARSE,
        LLM_VAST_AVG_INFERENCE_SECS_JD_VALIDATE,
        LLM_VAST_AVG_INFERENCE_SECS_JOB_SKILLS,
    )
    from inference.registry import get_backend

    backend_name = args.backend
    model = args.model or LLM_MODEL
    api_key = args.api_key or LLM_API_KEY
    base_url = args.backend_url or ""

    if backend_name == "cpu" or (backend_name is None and base_url):
        # Local CPU/llama.cpp endpoint — no lifecycle management
        if not base_url:
            base_url = "http://localhost:8085/v1"
        backend = get_backend("cpu", base_url=base_url, model=model, api_key=api_key)

    elif backend_name == "local":
        if not base_url:
            base_url = "http://localhost:8000/v1"
        backend = get_backend("local", base_url=base_url, model=model, api_key=api_key)

    elif backend_name == "remote":
        # Vast.ai — task-specific avg_inference_secs and ctx_per_slot
        task = args.task
        if task == "job_skills":
            avg_secs = LLM_VAST_AVG_INFERENCE_SECS_JOB_SKILLS
            ctx = LLM_CONTEXT_WINDOW_JOB_SKILLS
        elif task == "jd_reparse":
            avg_secs = LLM_VAST_AVG_INFERENCE_SECS_JD_REPARSE
            ctx = LLM_CONTEXT_WINDOW_JD_REPARSE
        elif task == "jd_validate":
            avg_secs = LLM_VAST_AVG_INFERENCE_SECS_JD_VALIDATE
            ctx = LLM_CONTEXT_WINDOW_JD_REPARSE
        else:
            avg_secs = LLM_VAST_AVG_INFERENCE_SECS_COMPANY_ENRICH
            ctx = LLM_CONTEXT_WINDOW_COMPANY_ENRICH

        if not base_url:
            base_url = "http://localhost:8000/v1"

        backend = get_backend(
            "remote",
            base_url=base_url,
            model=model,
            api_key=api_key,
            avg_inference_secs=avg_secs,
            ctx_per_slot=ctx,
        )
    else:
        raise ValueError(
            f"Unknown backend: '{backend_name}'. Choose: cpu, local, remote"
        )

    backend.n_parallel = args.n_parallel or 1
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark workload driver for the inference case study."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["job_skills", "jd_reparse", "jd_validate", "company_enrich"],
        help="Inference task to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum records to process (default: 1000).",
    )
    parser.add_argument(
        "--db-url",
        default="postgresql://fixture:fixture@localhost:5433/inference_fixture",
        help="PostgreSQL connection URL for the fixture DB.",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "local", "remote"],
        default="remote",
        help="Backend to use: cpu (local llama.cpp), local (any HTTP endpoint), "
        "remote (Vast.ai on-demand GPU).",
    )
    parser.add_argument(
        "--backend-url",
        default="",
        help="Base URL override for cpu/local backends "
        "(e.g. http://localhost:8085/v1).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name to pass to the inference server.",
    )
    parser.add_argument(
        "--api-key",
        default="none",
        help="API key for the inference server (default: 'none' for local).",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=0,
        help="Number of parallel workers. 0 = auto (set by backend startup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (recorded in output; fixture uses ORDER BY id ASC).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process already-enriched records.",
    )
    parser.add_argument(
        "--static-batches",
        action="store_true",
        help="Use static interleaved batch split (worker i gets rows[i::n]). "
        "Replicates the original pre-unified-queue dispatch design. "
        "Idle workers cannot steal from busier ones once their slice is exhausted.",
    )
    parser.add_argument(
        "--env-file",
        default="",
        help="Per-iteration config file (e.g. iterations/02-bugs-fixed/config.env). "
        "Loaded after the root .env with override semantics — iteration values win.",
    )
    parser.add_argument(
        "--output",
        default="iterations/local-emulation/benchmark_results.json",
        help="Output path for benchmark_results.json.",
    )
    args = parser.parse_args()

    # Load .env from the case-study root (base defaults, setdefault — existing env wins)
    _load_env(str(_root / ".env"))
    _load_env(".env")
    # Per-iteration config overrides base values (force-set — iteration wins)
    if args.env_file:
        _load_env(args.env_file, override=True)

    backend = _build_backend(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dispatch = "static-batches" if args.static_batches else "shared-queue"
    print(f"=== Benchmark: {args.task} | limit={args.limit} | backend={args.backend} | dispatch={dispatch} ===")
    print(f"DB:     {args.db_url}")
    print(f"Output: {output_path}")
    print()

    from inference.pipeline import run_batch

    t_wall_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    summaries, compute_secs = run_batch(
        backend=backend,
        task=args.task,
        limit=args.limit,
        db_url=args.db_url,
        force=args.force,
        static_batches=args.static_batches,
    )
    t_wall_end = time.monotonic()
    wall_secs = t_wall_end - t_wall_start

    ok = sum(1 for s in summaries if s.get("status") == "ok")
    failed = sum(1 for s in summaries if s.get("status") not in ("ok",))
    wall_rate = len(summaries) / (wall_secs / 60) if wall_secs > 0 else 0
    compute_rate = len(summaries) / (compute_secs / 60) if compute_secs > 0 else 0

    # Per-record token metrics (job_skills / company_enrich only)
    latencies = [
        s["latency_ms"] for s in summaries if "latency_ms" in s
    ]
    in_tokens = [s["input_tokens"] for s in summaries if "input_tokens" in s]
    out_tokens = [s["output_tokens"] for s in summaries if "output_tokens" in s]

    def _avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "meta": {
            "task": args.task,
            "limit": args.limit,
            "backend": args.backend,
            "dispatch": "static-batches" if args.static_batches else "shared-queue",
            "seed": args.seed,
            "model": args.model or os.getenv("LLM_MODEL", ""),
            "started_at": started_at,
            "wall_secs": round(wall_secs, 1),
            "compute_secs": round(compute_secs, 1),
        },
        "summary": {
            "total": len(summaries),
            "ok": ok,
            "failed": failed,
            "wall_rate_rec_per_min": round(wall_rate, 1),
            "compute_rate_rec_per_min": round(compute_rate, 1),
            "avg_latency_ms": round(_avg(latencies)),
            "avg_input_tokens": round(_avg(in_tokens)),
            "avg_output_tokens": round(_avg(out_tokens)),
            "total_input_tokens": sum(in_tokens),
            "total_output_tokens": sum(out_tokens),
        },
        "records": summaries,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    def _fmt_time(secs: float) -> str:
        return f"{int(secs // 60)}m{int(secs % 60):02d}s"

    print()
    print(f"=== Results ===")
    print(f"  Total:        {len(summaries)}")
    print(f"  OK:           {ok}")
    print(f"  Failed:       {failed}")
    print(f"  Compute time: {_fmt_time(compute_secs)}  ({compute_rate:.1f} rec/min)")
    print(f"  Wall time:    {_fmt_time(wall_secs)}  ({wall_rate:.1f} rec/min)")
    if latencies:
        print(f"  Avg latency:  {_avg(latencies):.0f}ms")
    print(f"  Output:       {output_path}")


if __name__ == "__main__":
    main()
