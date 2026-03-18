# inference/base.py
"""Base interface for all LLM backends.

Any new backend must subclass BaseLLMBackend, implement health_check(),
and optionally override startup() / shutdown() for lifecycle management.

The complete() method is implemented once here using the openai SDK and is
never overridden by subclasses — all backends share identical inference logic.
This works with any OpenAI-compatible server: llama-cpp, vLLM, Ollama, etc.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI


@dataclass
class LLMRequest:
    """Input to a single LLM inference call."""

    task: str  # "job_skills" | "jd_reparse" | "jd_validate" | "company_enrich"
    system_prompt: str
    user_prompt: str
    max_tokens: int = 800
    temperature: float = 0.1


@dataclass
class LLMResponse:
    """Output from a single LLM inference call."""

    content: str  # raw model output
    input_tokens: int
    output_tokens: int
    latency_ms: int
    parse_success: bool = False  # set by caller after JSON validation
    parsed: dict[str, Any] | None = None  # set by caller after JSON validation
    worker_label: str = ""  # which GPU/worker handled the request (e.g. "gpu1")


class BaseLLMBackend(ABC):
    """Abstract base class every LLM backend must implement.

    How to add a new backend
    ------------------------
    1. Create inference/backends/mybackend.py
    2. Subclass BaseLLMBackend, implement health_check()
    3. Override startup() / shutdown() if your backend needs lifecycle management
    4. Decorate the class with @register("mybackend")
    5. Import the module in inference/backends/__init__.py
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        # Populated by startup() for remote backends (computed from VRAM);
        # set manually for local/cpu backends via the --n-parallel CLI flag.
        self.n_parallel: int = 1

    def startup(self, queue_size: int = 0) -> None:
        """Called before the batch begins.

        Args:
            queue_size: Number of records about to be processed. Passed through
                so backends like Vast.ai can size instance TTL accurately.
                Bug note: an early version always passed 0 here, causing the TTL
                to always default to 31 min — instances were destroyed mid-batch.

        Default: health-check and raise if the server is not reachable.
        Override to add lifecycle management (e.g. power on a remote GPU instance).
        """
        if not self.health_check():
            raise RuntimeError(
                f"LLM server not reachable at {self.base_url}. "
                "Ensure the server is running before starting a batch."
            )

    def shutdown(self) -> None:
        """Called in finally after the batch finishes (or errors).

        Default: no-op. Override to power off a remote VM.
        """

    def health_check(self) -> bool:
        """Return True if the inference server is up and responding.

        Default: GET {base_url}/models -> HTTP 200.
        """
        base = self.base_url.rstrip("/")
        url = f"{base}/models"
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one chat completion request and return a structured response.

        Implemented once in the base class — subclasses must NOT override this.
        Uses the openai SDK with a configurable base_url so it works with any
        OpenAI-compatible server.
        """
        import time

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]

        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        usage = response.usage
        content = response.choices[0].message.content or ""

        worker_label_fn = getattr(self, "get_worker_label", None)
        worker_label = str(worker_label_fn()) if callable(worker_label_fn) else ""

        return LLMResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            worker_label=worker_label,
        )


__all__ = ["LLMRequest", "LLMResponse", "BaseLLMBackend"]
