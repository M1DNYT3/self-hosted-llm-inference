# inference/config.py
"""All configuration via environment variables.

Copy harness/.env.example to case-study/.env and fill in your keys.
All values have sensible defaults matching the workload contract.
"""

import os

# ---------------------------------------------------------------------------
# Inference server (shared)
# ---------------------------------------------------------------------------
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "none")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen3.5-9B-Q4_K_M")
LLM_GPU_BACKEND: str = os.getenv("LLM_GPU_BACKEND", "remote")  # "cpu" | "local" | "remote"
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "800"))

# ---------------------------------------------------------------------------
# CPU always-on container (port 8085)
# ---------------------------------------------------------------------------
CPU_LLM_BASE_URL: str = os.getenv("CPU_LLM_BASE_URL", "http://localhost:8085/v1")
CPU_LLM_MODEL: str = os.getenv("CPU_LLM_MODEL", "Qwen3.5-9B-Q4_K_M")

# CPU threshold: if estimated batch duration < threshold → use CPU, else GPU
CPU_THRESHOLD_JOB_SKILLS_HOURS: float = float(
    os.getenv("CPU_THRESHOLD_JOB_SKILLS_HOURS", "10")
)
CPU_THRESHOLD_COMPANY_ENRICH_HOURS: float = float(
    os.getenv("CPU_THRESHOLD_COMPANY_ENRICH_HOURS", "2")
)
CPU_THRESHOLD_JD_REPARSE_HOURS: float = float(
    os.getenv("CPU_THRESHOLD_JD_REPARSE_HOURS", "1")
)
CPU_THRESHOLD_JD_VALIDATE_HOURS: float = float(
    os.getenv("CPU_THRESHOLD_JD_VALIDATE_HOURS", "1")
)

# ---------------------------------------------------------------------------
# Context windows (tokens) per task
# ---------------------------------------------------------------------------
LLM_CONTEXT_WINDOW_JOB_SKILLS: int = int(
    os.getenv("LLM_CONTEXT_WINDOW_JOB_SKILLS", "4096")
)
LLM_CONTEXT_WINDOW_JD_REPARSE: int = int(
    os.getenv("LLM_CONTEXT_WINDOW_JD_REPARSE", "4096")
)
LLM_CONTEXT_WINDOW_COMPANY_ENRICH: int = int(
    os.getenv("LLM_CONTEXT_WINDOW_COMPANY_ENRICH", "4096")
)

# ---------------------------------------------------------------------------
# Per-task average inference time calibration (seconds/record)
# Calibrated on 1x RTX 3090 24 GB, Qwen3.5-9B Q4_K_M, --reasoning-budget 0
# Used by TTL formula: startup + ceil(queue/n_parallel) * avg_secs + 30m buffer
# ---------------------------------------------------------------------------
LLM_VAST_AVG_INFERENCE_SECS_JOB_SKILLS: int = int(
    os.getenv("LLM_VAST_AVG_INFERENCE_SECS_JOB_SKILLS", "20")
)
LLM_VAST_AVG_INFERENCE_SECS_JD_REPARSE: int = int(
    # 36s/slice × 4.5 avg slices/record = 162s/record.
    # Using per-slice time (36) here caused TTL underestimation — the instance
    # was destroyed mid-batch because the budget was 4.5× too short.
    os.getenv("LLM_VAST_AVG_INFERENCE_SECS_JD_REPARSE", "162")
)
LLM_VAST_AVG_INFERENCE_SECS_JD_VALIDATE: int = int(
    os.getenv("LLM_VAST_AVG_INFERENCE_SECS_JD_VALIDATE", "33")
)
LLM_VAST_AVG_INFERENCE_SECS_COMPANY_ENRICH: int = int(
    os.getenv("LLM_VAST_AVG_INFERENCE_SECS_COMPANY_ENRICH", "50")
)

# Average slices per jd_reparse record (used by router to estimate wall time)
JD_REPARSE_AVG_SLICES_PER_RECORD: float = float(
    os.getenv("JD_REPARSE_AVG_SLICES_PER_RECORD", "4.5")
)

# ---------------------------------------------------------------------------
# Vast.ai — instance selection
# ---------------------------------------------------------------------------
LLM_PROVIDER_KEY: str = os.getenv("LLM_PROVIDER_KEY", "")  # Vast.ai API key
LLM_OFFER_ID: str = os.getenv("LLM_OFFER_ID", "")  # pin a specific offer; empty = auto-select

LLM_VAST_IMAGE: str = os.getenv(
    "LLM_VAST_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda"
)
LLM_VAST_HF_REPO: str = os.getenv(
    "LLM_VAST_HF_REPO", "bartowski/Qwen2.5-7B-Instruct-GGUF"
)
LLM_VAST_HF_FILE: str = os.getenv("LLM_VAST_HF_FILE", "Qwen3.5-9B-Q4_K_M.gguf")
LLM_VAST_DISK_GB: int = int(os.getenv("LLM_VAST_DISK_GB", "20"))
LLM_VAST_SSH_KEY: str = os.getenv("LLM_VAST_SSH_KEY", "~/.ssh/id_rsa")

# ---------------------------------------------------------------------------
# Vast.ai — VRAM / parallel slot formula
# n_parallel = (vram_gb - model_vram_gb) / kv_slot_gb
# Model weights (6 GB) are a fixed cost; remaining VRAM = KV cache budget.
# ---------------------------------------------------------------------------
LLM_VAST_MODEL_VRAM_GB: float = float(os.getenv("LLM_VAST_MODEL_VRAM_GB", "6.0"))
LLM_VAST_KV_SLOT_GB: float = float(os.getenv("LLM_VAST_KV_SLOT_GB", "0.5"))

# ---------------------------------------------------------------------------
# Vast.ai — offer search filters
# ---------------------------------------------------------------------------
LLM_VAST_NUM_GPUS: int = int(os.getenv("LLM_VAST_NUM_GPUS", "4"))  # start of tiered search
LLM_VAST_SINGLE_GPU_MODE: bool = (
    os.getenv("LLM_VAST_SINGLE_GPU_MODE", "false").lower() == "true"
)  # rent the offer as-is, but start/use only GPU 0 (simulate single-GPU on a multi-GPU bundle)
LLM_VAST_GPU_NAME: str = os.getenv(
    "LLM_VAST_GPU_NAME", ""
)  # optional substring filter on gpu_name (e.g. "3090"); empty = no filter
LLM_VAST_MIN_VRAM_GB: int = int(os.getenv("LLM_VAST_MIN_VRAM_GB", "8"))
LLM_VAST_MAX_VRAM_GB: int = int(os.getenv("LLM_VAST_MAX_VRAM_GB", "96"))
LLM_VAST_MIN_PRICE: float = float(os.getenv("LLM_VAST_MIN_PRICE", "0.0"))  # USD/hr floor
LLM_VAST_MAX_PRICE: float = float(os.getenv("LLM_VAST_MAX_PRICE", "1.0"))  # USD/hr
LLM_VAST_MAX_BW_COST_USD: float = float(
    os.getenv("LLM_VAST_MAX_BW_COST_USD", "0.50")
)  # max model-download cost
LLM_VAST_MIN_RELIABILITY: float = float(os.getenv("LLM_VAST_MIN_RELIABILITY", "0.95"))
LLM_VAST_MIN_TFLOPS_PER_GPU: float = float(
    os.getenv("LLM_VAST_MIN_TFLOPS_PER_GPU", "20.0")
)
LLM_VAST_MIN_MEM_BW_PER_GPU: float = float(
    os.getenv("LLM_VAST_MIN_MEM_BW_PER_GPU", "0.0")
)  # GB/s per card; 0 = disabled. LLM decode is bandwidth-bound — this is a
   # more direct quality filter than TFLOPS for autoregressive workloads.
LLM_VAST_MIN_INET_DOWN_MBPS: int = int(
    os.getenv("LLM_VAST_MIN_INET_DOWN_MBPS", "1000")
)  # ≥1 Gbps — guards against slow model downloads
LLM_VAST_REQUIRE_DATACENTER: bool = (
    os.getenv("LLM_VAST_REQUIRE_DATACENTER", "true").lower() == "true"
)
LLM_VAST_REQUIRE_VERIFIED: bool = (
    os.getenv("LLM_VAST_REQUIRE_VERIFIED", "true").lower() == "true"
)

# ---------------------------------------------------------------------------
# Vast.ai — timeouts and duration
# ---------------------------------------------------------------------------
LLM_VAST_TIMEOUT_RUNNING: int = int(
    os.getenv("LLM_VAST_TIMEOUT_RUNNING", "300")
)  # seconds to wait for "running" status
LLM_VAST_TIMEOUT_SSH: int = int(
    os.getenv("LLM_VAST_TIMEOUT_SSH", "120")
)  # seconds to wait for SSH ready
LLM_VAST_TIMEOUT_LOAD: int = int(
    os.getenv("LLM_VAST_TIMEOUT_LOAD", "600")
)  # seconds to wait for llama-server model load
LLM_VAST_MAX_DURATION_HOURS: int = int(
    os.getenv("LLM_VAST_MAX_DURATION_HOURS", "4")
)  # hard TTL cap
LLM_VAST_KEEP_ON_FAILURE: bool = (
    os.getenv("LLM_VAST_KEEP_ON_FAILURE", "false").lower() == "true"
)  # skip instance destruction on exit — preserves instance for SSH log inspection

# ---------------------------------------------------------------------------
# Pipeline limits (used by workload_driver; overridden by CLI --limit)
# ---------------------------------------------------------------------------
PIPELINE_JD_REPARSE_LIMIT: int = int(os.getenv("PIPELINE_JD_REPARSE_LIMIT", "5000"))
PIPELINE_JD_VALIDATE_LIMIT: int = int(os.getenv("PIPELINE_JD_VALIDATE_LIMIT", "5000"))
