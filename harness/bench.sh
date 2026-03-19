#!/usr/bin/env bash
# harness/bench.sh — Parameterized benchmark runner with structured log capture.
#
# Usage:
#   bash harness/bench.sh ITER TASK LIMIT [N_PARALLEL] [BACKEND_URL] [MODEL] [BACKEND] [STATIC_BATCHES] [ENV_FILE]
#
# Arguments:
#   ITER           Iteration directory name under iterations/ (e.g. 00-baseline)
#   TASK           Inference task: job_skills | jd_reparse | jd_validate | company_enrich
#   LIMIT          Number of records to process
#   N_PARALLEL     Worker count (0 = auto from backend; use explicit value for local runs)
#   BACKEND_URL    Base URL of the inference server
#   MODEL          Model name passed to the server (default: Qwen3.5-9B-Q4_K_M)
#   BACKEND        Backend type: local (LAN GPU / LM Studio) | cpu (Docker llama.cpp on localhost:8085)
#                                | remote (Vast.ai). Default: local.
#   STATIC_BATCHES Pass "static" to use interleaved pre-split dispatch (historical 02-bugs-fixed design).
#                  Omit or pass "" for the default shared-queue dispatch.
#   ENV_FILE       Optional explicit path to a config .env file. Overrides the default
#                  auto-detected iterations/{ITER}/config.env. Use for sub-run variants
#                  within the same iteration (e.g. different parallel counts).
#
# Output:
#   iterations/{ITER}/logs/{YYYYMMDD_HHMMSS}_{TASK}_{LIMIT}rec.log  — full console log
#   iterations/{ITER}/benchmark_results.json                         — structured metrics
#
# Logs are excluded from git by default (.gitignore: iterations/*/logs/).
# Commit a log manually when it's a keeper:
#   git add -f iterations/ITER/logs/filename.log
#
# Examples:
#   # Sequential baseline (1 worker, LAN GPU via LM Studio)
#   bash harness/bench.sh 00-baseline job_skills 200 1 http://192.168.1.x:PORT/v1 qwen/qwen3.5-9b local
#
#   # Parallel with bug (designed 4, but n_parallel not written back → pass 1 to simulate)
#   bash harness/bench.sh 01-parallel-1thread job_skills 200 1 http://192.168.1.x:PORT/v1 qwen/qwen3.5-9b local
#
#   # Bug fixed: 4 workers, static batch split (historical dispatch design for 02)
#   bash harness/bench.sh 02-bugs-fixed job_skills 200 4 http://192.168.1.x:PORT/v1 qwen/qwen3.5-9b local static
#
#   # CPU Docker container (always-on, localhost:8085)
#   bash harness/bench.sh 00-baseline job_skills 50 1 http://localhost:8085/v1 Qwen3.5-9B-Q4_K_M cpu
#
#   # Sub-run with explicit env file (e.g. 04 PRO 6000 72-slot variant)
#   bash harness/bench.sh 04-single-gpu-scaling job_skills 1000 0 "" "" remote static \
#     iterations/04-single-gpu-scaling/config-pro6000-72slots.env

set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────

ITER="${1:?Usage: bench.sh ITER TASK LIMIT [N_PARALLEL] [BACKEND_URL] [MODEL]}"
TASK="${2:?TASK required: job_skills|jd_reparse|jd_validate|company_enrich}"
LIMIT="${3:?LIMIT required: number of records}"
N_PARALLEL="${4:-0}"
BACKEND_URL="${5:-http://localhost:8085/v1}"
MODEL="${6:-Qwen3.5-9B-Q4_K_M}"
BACKEND="${7:-local}"
STATIC_BATCHES="${8:-}"
CUSTOM_ENV_FILE="${9:-}"

FIXTURE_URL="postgresql://fixture:fixture@localhost:5433/inference_fixture"

# ── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_STUDY_ROOT="$(dirname "$SCRIPT_DIR")"
ITER_DIR="$CASE_STUDY_ROOT/iterations/$ITER"
LOG_DIR="$ITER_DIR/logs"
OUTPUT_JSON="$ITER_DIR/benchmark_results.json"

if [[ ! -d "$ITER_DIR" ]]; then
    echo "ERROR: Iteration directory not found: $ITER_DIR"
    exit 1
fi

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/${TIMESTAMP}_${TASK}_${LIMIT}rec.log"

# ── Health check ─────────────────────────────────────────────────────────────

if [[ "$BACKEND" != "remote" ]]; then
    if ! curl -sf "$BACKEND_URL/models" > /dev/null 2>&1; then
        echo "ERROR: Inference server not responding at $BACKEND_URL"
        exit 1
    fi
fi

if ! psql "$FIXTURE_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo "ERROR: Fixture DB not responding at $FIXTURE_URL"
    echo "Start it with: docker compose -f docker/compose.yaml up fixture-db -d"
    exit 1
fi

# ── Run ──────────────────────────────────────────────────────────────────────

echo "=== bench.sh ==="
echo "Iteration:  $ITER"
echo "Task:       $TASK"
echo "Limit:      $LIMIT records"
echo "Workers:    $N_PARALLEL (0=auto)"
echo "Dispatch:   ${STATIC_BATCHES:-shared-queue}"
echo "Backend:    $BACKEND  ($BACKEND_URL)"
echo "Model:      $MODEL"
echo "Env file:   ${CUSTOM_ENV_FILE:-iterations/$ITER/config.env (auto)}"
echo "Log:        $LOG_FILE"
echo "Output:     $OUTPUT_JSON"
echo ""

(
    cd "$CASE_STUDY_ROOT"
    STATIC_FLAG=""
    [[ "$STATIC_BATCHES" == "static" ]] && STATIC_FLAG="--static-batches"
    ENV_FILE_FLAG=""
    if [[ -n "$CUSTOM_ENV_FILE" ]]; then
        ENV_FILE_FLAG="--env-file $CUSTOM_ENV_FILE"
    else
        ITER_CONFIG="$ITER_DIR/config.env"
        [[ -f "$ITER_CONFIG" ]] && ENV_FILE_FLAG="--env-file $ITER_CONFIG"
    fi
    python3 harness/workload_driver.py \
        --task "$TASK" \
        --limit "$LIMIT" \
        --db-url "$FIXTURE_URL" \
        --backend "$BACKEND" \
        --backend-url "$BACKEND_URL" \
        --model "$MODEL" \
        --n-parallel "$N_PARALLEL" \
        --seed 42 \
        --force \
        --output "$OUTPUT_JSON" \
        $STATIC_FLAG \
        $ENV_FILE_FLAG
) 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved: $LOG_FILE"
