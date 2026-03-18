#!/usr/bin/env bash
# run_local.sh — CPU emulation of the inference workload (no GPU cost).
#
# Uses the fixture DB and a local llama.cpp container on port 8085.
# Results will NOT match GPU benchmark rates — this validates timing shapes
# and request distribution only.
#
# Prerequisites:
#   - Docker running with fixture-db container (see docker/compose.yaml)
#   - llama.cpp server container running on port 8085 with Qwen3.5-9B Q4_K_M
#
# Usage:
#   bash harness/run_local.sh [task] [limit]
#   bash harness/run_local.sh job_skills 100
#   bash harness/run_local.sh company_enrich 50

set -euo pipefail

TASK="${1:-job_skills}"
LIMIT="${2:-100}"
FIXTURE_URL="postgresql://fixture:fixture@localhost:5433/inference_fixture"
CPU_URL="http://localhost:8085/v1"
CPU_MODEL="Qwen3.5-9B-Q4_K_M"

echo "=== Local CPU emulation ==="
echo "Task:    $TASK"
echo "Limit:   $LIMIT records"
echo "DB:      $FIXTURE_URL"
echo "Backend: $CPU_URL"
echo ""

# Health check
if ! curl -sf "$CPU_URL/models" > /dev/null 2>&1; then
    echo "ERROR: llama.cpp not responding at $CPU_URL"
    echo "Start it with: docker compose -f docker/compose.yaml up llama-cpu -d"
    exit 1
fi

# Check fixture DB
if ! psql "$FIXTURE_URL" -c "SELECT 1" > /dev/null 2>&1; then
    echo "ERROR: Fixture DB not responding at $FIXTURE_URL"
    echo "Start it with: docker compose -f docker/compose.yaml up fixture-db -d"
    exit 1
fi

echo "Both services ready. Running workload..."
echo ""

python harness/workload_driver.py \
    --task "$TASK" \
    --limit "$LIMIT" \
    --db-url "$FIXTURE_URL" \
    --backend-url "$CPU_URL" \
    --model "$CPU_MODEL" \
    --seed 42 \
    --output "iterations/local-emulation/benchmark_results.json"
