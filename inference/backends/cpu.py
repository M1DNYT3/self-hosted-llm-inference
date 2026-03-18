# inference/backends/cpu.py
"""CPU-local LLM backend — always-on llama.cpp Docker container.

This backend is not a user-selectable LLM_GPU_BACKEND value. It is
instantiated by the routing layer and used as the first-choice backend
for small batches whose estimated duration falls within the CPU threshold.

Lifecycle:
  startup()    — health-check only (container is always-on; raises if unreachable)
  shutdown()   — no-op (container lifecycle managed by Docker / docker compose)
  health_check() — GET {CPU_LLM_BASE_URL}/models → HTTP 200
"""

from inference.base import BaseLLMBackend
from inference.registry import register


@register("cpu")
class CpuLLMBackend(BaseLLMBackend):
    """Backend for the always-on CPU llama.cpp Docker container (port 8085)."""

    def startup(self, queue_size: int = 0) -> None:
        if not self.health_check():
            raise RuntimeError(
                f"CPU LLM container not reachable at {self.base_url}.\n"
                "Start it with:\n"
                "  docker compose -f docker/compose.yaml --profile cpu up -d llama-cpu"
            )
        print(f"CPU LLM container online at {self.base_url}")

    def shutdown(self) -> None:
        pass  # always-on — Docker manages the container lifecycle
