# inference/backends/local.py
"""Local LLM backend — health check only, no VM lifecycle management.

The user runs the inference server (llama-cpp or vLLM) on a separate machine.
LLM_BASE_URL points at it. This backend just verifies the server is up
before the batch begins.
"""

from inference.base import BaseLLMBackend
from inference.registry import register


@register("local")
class LocalLLMBackend(BaseLLMBackend):
    """Backend for a locally-managed (user-controlled) LLM server."""

    def startup(self, queue_size: int = 0) -> None:
        if not self.health_check():
            raise RuntimeError(
                f"LLM server not reachable at {self.base_url}.\n"
                "Start the server before running the benchmark.\n"
                "  llama.cpp: docker compose -f docker/compose.yaml up llama-cpu -d\n"
                "  vLLM:      vllm serve <model> --port 8080"
            )
        print(f"LLM server online at {self.base_url}")

    def shutdown(self) -> None:
        pass  # user manages the server lifecycle
