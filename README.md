# LLM Inference Case Study

Self-contained artifact documenting an iterative on-demand GPU inference system
built to replace per-volume API pricing with predictable, self-hosted compute.

---

## What's here

| Directory | Contents |
|---|---|
| `docs/` | Executive summary, iteration log, measurement methodology |
| `inference/` | Sanitized inference layer (router, backends, batch pipeline) — no product code |
| `fixture/` | Self-contained Postgres fixture DB with the exact dataset used during development |
| `iterations/` | One directory per design iteration with real benchmark artifacts |
| `harness/` | Workload driver and local emulation scripts |
| `docker/` | Docker Compose stack, pinned image digests |

---

## Quick start (local emulation — no GPU cost)

```bash
# 1. Start the fixture database
docker compose -f docker/compose.yaml up fixture-db -d

# 2. Run the CPU emulation harness (requires llama.cpp or a local inference server)
bash harness/run_local.sh
```

## Full remote run (Vast.ai — incurs real cost)

> **COST WARNING**: Running on Vast.ai rents a real GPU instance. Expect $0.50–0.80/hr.
> Always review `harness/run_remote.sh` before executing.

```bash
bash harness/run_remote.sh
```

---

## Reproducibility contract

- **Dataset**: `fixture/dump.sql.gz` — filtered snapshot of real production data.
  Output tables are empty; all records are eligible (equivalent to `--force`).
- **Workload shape**: `harness/workload-contract.yaml` — request size distribution,
  concurrency, burst profile, seeds.
- **What is fixed**: workload shape, model (Qwen3.5-9B Q4_K_M), inference server
  (llama.cpp), parallelism formula, timeout policies.
- **What is replaceable**: the dataset (substitute any Postgres dump matching the schema).
- **What NOT to run**: `run_remote.sh` unless you have a Vast.ai account and accept the cost.

---

## Iteration log

See [docs/iteration-log.md](docs/iteration-log.md) for the full table of design changes
and measured deltas. Each iteration has a corresponding directory in `iterations/` with
the raw benchmark output.
