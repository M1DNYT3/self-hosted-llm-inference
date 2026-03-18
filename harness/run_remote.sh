#!/usr/bin/env bash
# run_remote.sh — Run the benchmark against a live Vast.ai GPU instance.
#
# !! COST WARNING !!
# This script rents a real GPU instance on Vast.ai.
# Typical cost: $0.30–0.80/hr + model download overhead.
# The instance is destroyed automatically after the run (via backend.shutdown()).
# Always verify your Vast.ai account balance before running.
#
# Prerequisites:
#   - LLM_PROVIDER_KEY set in .env (your Vast.ai API key)
#   - LLM_VAST_SSH_KEY set to your SSH private key path
#   - Fixture DB running (docker compose -f docker/compose.yaml up fixture-db -d)
#
# Usage:
#   cp harness/.env.example .env  # fill in your keys
#   bash harness/run_remote.sh [task] [limit] [iteration_name]
#   bash harness/run_remote.sh job_skills 1000 07-4x4070sti

set -euo pipefail

TASK="${1:-job_skills}"
LIMIT="${2:-1000}"
ITERATION="${3:-remote-run}"
FIXTURE_URL="postgresql://fixture:fixture@localhost:5433/inference_fixture"

echo "=== COST WARNING: This rents a real GPU instance on Vast.ai ==="
echo "Task:      $TASK"
echo "Limit:     $LIMIT records"
echo "Iteration: $ITERATION"
echo ""
read -p "Proceed? (yes/no): " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

# Load env
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

mkdir -p "iterations/$ITERATION"

python harness/workload_driver.py \
    --task "$TASK" \
    --limit "$LIMIT" \
    --db-url "$FIXTURE_URL" \
    --backend "remote" \
    --seed 42 \
    --output "iterations/$ITERATION/benchmark_results.json"

echo ""
echo "Results written to iterations/$ITERATION/benchmark_results.json"
echo "Run: python harness/plot.py iterations/$ITERATION/ to generate plots."
