# inference/backends/vastai.py
"""Vast.ai on-demand GPU backend.

Lifecycle:
  startup()  → validate HF model exists (HEAD request — free, no billing)
             → auto-select cheapest matching GPU offer (GET /bundles/ — free)
             → compute n_parallel from VRAM and instance TTL from queue size
             → POST /asks/{offer_id}/ to rent instance
             → poll until actual_status == "running"
             → open SSH tunnel per GPU: localhost:{free_port} → container:800N
             → poll llama-server health until all GPUs ready (GGUF download ~2-5 min)
  shutdown() → terminate SSH tunnel subprocess
             → DELETE /instances/{instance_id}/ (always executes via finally)

Vast.ai enforces the instance TTL server-side: even if our process crashes,
the instance is destroyed after duration_secs — preventing runaway billing.

Auth: Authorization: Bearer {LLM_PROVIDER_KEY}
Uses requests directly against the Vast.ai REST API — no SDK dependency.
"""

import json
import math
import re
import socket
import subprocess
import time

import requests

from inference.base import BaseLLMBackend
from inference.registry import register

_VASTAI_API = "https://console.vast.ai/api/v0"
_POLL_INTERVAL = 10  # seconds between status polls

# Post-filter: RTX 20xx–50xx gaming cards only.
# Matches any NVIDIA RTX card (gaming 2xxx–5xxx, workstation A-series, PRO).
# Quality is enforced downstream by MIN_MEM_BW_PER_GPU and TFLOPS filters —
# non-RTX cards (A100, H100, L40) are excluded by not carrying the RTX name.
_RTX_PATTERN = re.compile(r"\bRTX\b", re.IGNORECASE)


def _read_proc_stderr(proc: "subprocess.Popen[bytes]") -> str:
    raw = proc.stderr.read() if proc.stderr else b""
    return (raw or b"").decode(errors="replace")


@register("remote")
class VastaiLLMBackend(BaseLLMBackend):
    """Backend that rents a Vast.ai GPU instance on demand."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        avg_inference_secs: int,
        ctx_per_slot: int,
    ) -> None:
        super().__init__(base_url=base_url, model=model, api_key=api_key)
        self._instance_id: int | None = None
        self._tunnel_procs: list[subprocess.Popen] = []  # type: ignore[type-arg]
        self._base_urls: tuple[str, ...] | None = None

        # SSH connection details stored during startup for mid-batch reconnection
        self._ssh_host: str = ""
        self._ssh_port: int = 0
        self._local_ports: list[int] = []

        import threading
        self._thread_map: dict[int, int] = {}
        self._thread_lock = threading.Lock()
        self._tunnel_reconnect_lock = threading.Lock()

        from inference.config import (
            LLM_OFFER_ID,
            LLM_PROVIDER_KEY,
            LLM_VAST_DISK_GB,
            LLM_VAST_GPU_NAME,
            LLM_VAST_HF_FILE,
            LLM_VAST_HF_REPO,
            LLM_VAST_IMAGE,
            LLM_VAST_KV_SLOT_GB,
            LLM_VAST_MAX_DURATION_HOURS,
            LLM_VAST_MIN_PRICE,
            LLM_VAST_MAX_PRICE,
            LLM_VAST_MAX_BW_COST_USD,
            LLM_VAST_MAX_VRAM_GB,
            LLM_VAST_MIN_RELIABILITY,
            LLM_VAST_MIN_TFLOPS_PER_GPU,
            LLM_VAST_MIN_MEM_BW_PER_GPU,
            LLM_VAST_MIN_INET_DOWN_MBPS,
            LLM_VAST_MIN_VRAM_GB,
            LLM_VAST_MODEL_VRAM_GB,
            LLM_VAST_NUM_GPUS,
            LLM_VAST_REQUIRE_DATACENTER,
            LLM_VAST_REQUIRE_VERIFIED,
            LLM_VAST_SINGLE_GPU_MODE,
            LLM_VAST_SSH_KEY,
            LLM_VAST_KEEP_ON_FAILURE,
            LLM_VAST_TIMEOUT_LOAD,
            LLM_VAST_TIMEOUT_RUNNING,
            LLM_VAST_TIMEOUT_SSH,
        )

        self._offer_id: str = LLM_OFFER_ID
        self._vast_image: str = LLM_VAST_IMAGE
        self._ctx_per_slot: int = ctx_per_slot
        self._provider_key: str = LLM_PROVIDER_KEY
        self._ssh_key: str = LLM_VAST_SSH_KEY

        self._min_price: float = LLM_VAST_MIN_PRICE
        self._max_price: float = LLM_VAST_MAX_PRICE
        self._max_bw_cost: float = LLM_VAST_MAX_BW_COST_USD
        self._min_vram_gb: int = LLM_VAST_MIN_VRAM_GB
        self._max_vram_gb: int = LLM_VAST_MAX_VRAM_GB
        self._num_gpus: int = LLM_VAST_NUM_GPUS
        self._require_datacenter: bool = LLM_VAST_REQUIRE_DATACENTER
        self._require_verified: bool = LLM_VAST_REQUIRE_VERIFIED
        self._min_reliability: float = LLM_VAST_MIN_RELIABILITY
        self._min_tflops_per_gpu: float = LLM_VAST_MIN_TFLOPS_PER_GPU
        self._min_mem_bw_per_gpu: float = LLM_VAST_MIN_MEM_BW_PER_GPU
        self._min_inet_down_mbps: int = LLM_VAST_MIN_INET_DOWN_MBPS
        self._single_gpu_mode: bool = LLM_VAST_SINGLE_GPU_MODE
        self._gpu_name_filter: str = LLM_VAST_GPU_NAME

        self._hf_repo: str = LLM_VAST_HF_REPO
        self._hf_file: str = LLM_VAST_HF_FILE
        self._disk_gb: int = LLM_VAST_DISK_GB

        self._timeout_running: int = LLM_VAST_TIMEOUT_RUNNING
        self._timeout_ssh: int = LLM_VAST_TIMEOUT_SSH
        self._timeout_load: int = LLM_VAST_TIMEOUT_LOAD
        self._keep_on_failure: bool = LLM_VAST_KEEP_ON_FAILURE

        self._model_vram_gb: float = LLM_VAST_MODEL_VRAM_GB
        self._kv_slot_gb: float = LLM_VAST_KV_SLOT_GB
        self._avg_inference_secs: int = avg_inference_secs
        self._max_duration_hours: int = LLM_VAST_MAX_DURATION_HOURS

    # ------------------------------------------------------------------
    # Thread-affinity routing for multi-GPU instances
    # Each thread is pinned to a specific GPU via thread_id % num_gpus.
    # Workers call backend.complete() which reads self.base_url — which
    # returns the URL for the GPU assigned to this thread.
    # ------------------------------------------------------------------

    def _get_thread_index(self) -> int:
        import threading
        tid = threading.get_ident()
        with self._thread_lock:
            if tid not in self._thread_map:
                self._thread_map[tid] = len(self._thread_map) % self._num_gpus
            return self._thread_map[tid]

    @property
    def base_url(self) -> str:
        if not getattr(self, "_base_urls", None):
            return getattr(self, "_default_base_url", "http://localhost:8000/v1")
        urls = self._base_urls
        assert urls is not None
        idx = self._get_thread_index()
        return urls[idx]

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._default_base_url = value

    def get_worker_label(self) -> str:
        """Returns 'gpu1'...'gpuN' based on the thread ID."""
        if not getattr(self, "_base_urls", None):
            return ""
        idx = self._get_thread_index()
        return f"gpu{idx + 1}"

    def health_check(self) -> bool:
        """Return True if all GPU llama-servers are up and responding."""
        if not getattr(self, "_base_urls", None):
            return super().health_check()
        urls = self._base_urls
        assert urls is not None
        for url in urls:
            base = url.rstrip("/")
            try:
                resp = requests.get(f"{base}/models", timeout=5)
                if resp.status_code != 200:
                    return False
            except requests.RequestException:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._provider_key}"}

    def _check_hf_model(self) -> None:
        """Validate HF repo/file exist on HuggingFace.

        Uses a HEAD request — no download, no billing risk.
        Must be called before any GPU is rented.
        """
        if not self._hf_repo or not self._hf_file:
            raise RuntimeError(
                "LLM_VAST_HF_REPO and LLM_VAST_HF_FILE must both be set. "
                "Example: LLM_VAST_HF_REPO=bartowski/Qwen2.5-7B-Instruct-GGUF"
            )
        url = (
            f"https://huggingface.co/{self._hf_repo}/resolve/main/{self._hf_file}"
        )
        try:
            resp = requests.head(url, allow_redirects=True, timeout=15)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"[vastai] HuggingFace check failed: {exc}"
            ) from exc
        if not resp.ok:
            raise RuntimeError(
                f"[vastai] Model not found on HuggingFace (HTTP {resp.status_code}): "
                f"{self._hf_repo}/{self._hf_file}"
            )
        print(
            f"[vastai] Model verified on HuggingFace: "
            f"{self._hf_repo}/{self._hf_file}"
        )

    def _find_offer(self) -> tuple[int, float, float]:
        """Query Vast.ai for the cheapest matching GPU offer.

        Uses a tiered fallback: starts at the configured GPU count (snapped to
        8→4→2→1) and falls back to lower tiers if no matches found.

        This is the strategy that discovered 4x RTX 4070S Ti as the MVP:
        the 8-GPU and 4-GPU tiers had no matches within price/VRAM constraints,
        and the fallback to 4x found the winning configuration.

        Returns (offer_id, vram_gb_per_card, inet_down_mbps).
        Raises RuntimeError if no offers match at any tier.
        """
        # Snap to standard Vast.ai GPU bundle tiers
        target = self._num_gpus
        if target >= 8:
            target = 8
        elif target >= 4:
            target = 4
        elif target >= 2:
            target = 2
        else:
            target = 1

        if target != self._num_gpus:
            print(
                f"[vastai] Normalizing GPU count {self._num_gpus} -> {target}"
            )
            self._num_gpus = target

        path = [p for p in [8, 4, 2, 1] if p <= target]
        path.sort(reverse=True)

        gpu_name_filter_str = f", gpu_name⊇'{self._gpu_name_filter}'" if self._gpu_name_filter else ""
        mem_bw_filter_str = f", mem_bw≥{self._min_mem_bw_per_gpu:.0f}GB/s" if self._min_mem_bw_per_gpu > 0 else ""
        print(
            f"[vastai] Offer search — filters: "
            f"x{target} GPUs (fallback path {path}), "
            f"price ${self._min_price:.2f}–${self._max_price:.2f}/hr, "
            f"VRAM {self._min_vram_gb}–{self._max_vram_gb}GB, "
            f"tflops≥{self._min_tflops_per_gpu}/GPU, "
            f"inet_down≥{self._min_inet_down_mbps}Mbps, "
            f"bw_cost≤${self._max_bw_cost:.2f}, "
            f"datacenter={self._require_datacenter}, "
            f"verified={self._require_verified}, "
            f"reliability≥{self._min_reliability}"
            f"{mem_bw_filter_str}"
            f"{gpu_name_filter_str}"
        )

        best_offer = None
        best_gpu_count = target

        for gpu_count in path:
            q = {
                "rentable": {"eq": True},
                "verified": {"eq": self._require_verified},
                "dph_total": {"gte": self._min_price, "lte": self._max_price},
                "gpu_ram": {
                    "gte": self._min_vram_gb * 1024,
                    "lte": self._max_vram_gb * 1024,
                },
                "datacenter": {"eq": self._require_datacenter},
                "reliability2": {"gte": self._min_reliability},
                "inet_down": {"gte": self._min_inet_down_mbps},
                "num_gpus": {"eq": gpu_count},
                "total_flops": {"gte": self._min_tflops_per_gpu * gpu_count},
            }
            try:
                resp = requests.get(
                    f"{_VASTAI_API}/bundles/",
                    params={"q": json.dumps(q)},
                    headers=self._headers(),
                    timeout=20,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(
                    f"Warning: [vastai] Offer search for x{gpu_count} failed: {exc}"
                )
                continue

            offers = resp.json().get("offers", [])

            # Post-filter: gaming GPUs only (RTX 30xx/40xx/50xx).
            # LLM decode is memory-bandwidth-bound, not TFLOPS-bound.
            # Workstation cards (A100, RTX PRO 6000) have higher TFLOPS but
            # lower bandwidth per dollar → same throughput at 2-3× the price.
            matching = [
                o for o in offers if _RTX_PATTERN.search(o.get("gpu_name", ""))
            ]

            # Optional GPU name substring filter (e.g. "3090" to target a specific card).
            if self._gpu_name_filter:
                matching = [
                    o for o in matching
                    if self._gpu_name_filter.lower() in o.get("gpu_name", "").lower()
                ]

            # Exclude China-region hosts: HuggingFace is blocked there,
            # causing model download to stall silently at instance startup.
            matching = [
                o for o in matching
                if "CN" not in (o.get("geolocation", "") or "").upper()
            ]

            # Filter by minimum GPU memory bandwidth (GB/s per card).
            # LLM autoregressive decode is memory-bandwidth-bound, not TFLOPS-bound.
            # A bandwidth floor is a more direct quality gate than TFLOPS for this workload.
            # Disabled when min_mem_bw_per_gpu == 0 (default).
            if self._min_mem_bw_per_gpu > 0:
                matching = [
                    o for o in matching
                    if float(o.get("gpu_mem_bw", 0.0)) >= self._min_mem_bw_per_gpu
                ]

            # Filter out hosts with expensive bandwidth (slow model downloads
            # are billed at full GPU rate — a $0.05 download that takes 10 min
            # on a $0.50/hr host costs $0.08 in wasted startup time).
            matching = [
                o for o in matching
                if (o.get("inet_down_cost", 0.0) * self._model_vram_gb)
                <= self._max_bw_cost
            ]

            if matching:
                def get_estimated_price(o: dict) -> float:
                    dph = o.get("dph_total", 999.0)
                    bw_cost = o.get("inet_down_cost", 0.0)
                    return dph + (bw_cost * self._model_vram_gb)

                matching.sort(key=get_estimated_price)
                best_offer = matching[0]
                best_gpu_count = gpu_count
                print(
                    f"[vastai] Found {len(matching)} matching offers "
                    f"for x{gpu_count} GPUs."
                )
                break
            else:
                print(
                    f"[vastai] No matching x{gpu_count} GPU offers. "
                    f"Falling back..."
                )

        if not best_offer:
            raise RuntimeError(
                f"[vastai] No matching GPU offers found after tiered search {path}. "
                f"Filters: price≤{self._max_price}, "
                f"VRAM {self._min_vram_gb}–{self._max_vram_gb}GB, "
                f"datacenter={self._require_datacenter}, "
                f"reliability≥{self._min_reliability}, "
                f"RTX 3xxx/4xxx/5xxx, MaxBWCost≤${self._max_bw_cost}"
            )

        if best_gpu_count != self._num_gpus:
            print(
                f"[vastai] DOWNGRADED: Target was {self._num_gpus}x, "
                f"using {best_gpu_count}x GPUs."
            )
            self._num_gpus = best_gpu_count

        offer_id: int = best_offer["id"]
        vram_gb: float = best_offer["gpu_ram"] / 1024
        inet_down: float = float(best_offer.get("inet_down", 500))
        inet_down_cost: float = float(best_offer.get("inet_down_cost", 0.0))
        bw_cost_estimate: float = inet_down_cost * self._model_vram_gb
        tflops: float = float(best_offer.get("total_flops", 0.0))
        tflops_per_gpu: float = tflops / best_gpu_count if best_gpu_count else 0.0
        mem_bw: float = float(best_offer.get("gpu_mem_bw", 0.0))
        reliability: float = float(best_offer.get("reliability2", 0.0))
        # Vast.ai bundles API returns datacenter as int (0/1), not bool.
        # bool(0) == False even for genuine DC hosts, so cast via int explicitly.
        datacenter: bool = bool(int(best_offer.get("datacenter") or 0))

        def get_final_price(o: dict) -> float:
            return o.get("dph_total", 0.0) + (
                o.get("inet_down_cost", 0.0) * self._model_vram_gb
            )

        dph: float = float(best_offer.get("dph_total", 0.0))
        price: float = get_final_price(best_offer)
        gpu_name: str = best_offer.get("gpu_name", "unknown")
        print(
            f"[vastai] Selected offer:\n"
            f"  GPU:         {gpu_name} x{best_gpu_count}\n"
            f"  VRAM:        {vram_gb:.0f} GB/card  ({vram_gb * best_gpu_count:.0f} GB total)\n"
            f"  TFLOPS:      {tflops_per_gpu:.0f}/GPU  ({tflops:.0f} total)\n"
            f"  Mem BW:      {mem_bw:.0f} GB/s/card\n"
            f"  Price:       ${dph:.3f}/hr base + ${bw_cost_estimate:.3f} BW est = ${price:.3f}/hr\n"
            f"  inet_down:   {inet_down:.0f} Mbps  (cost ${inet_down_cost:.4f}/GB)\n"
            f"  Reliability: {reliability:.3f}  datacenter={datacenter}\n"
            f"  Offer ID:    {offer_id}"
        )
        return offer_id, vram_gb, inet_down

    def _calc_parallel_slots(self, vram_gb: float) -> int:
        """Compute safe n_parallel from available VRAM per card.

        Formula: (total_vram - model_vram) / kv_slot_gb

        Model weights (6 GB for Qwen3.5-9B Q4_K_M) are a fixed cost loaded once.
        Remaining VRAM is divided into KV-cache slots — one slot per concurrent request.

        This asymmetry means doubling VRAM more than doubles available parallel slots:
          16 GB card: (16 - 6) / 0.5 = 20 slots
          24 GB card: (24 - 6) / 0.5 = 36 slots  (not 2×, but 1.8× the slots)
          32 GB card: (32 - 6) / 0.5 = 52 slots
        """
        available = vram_gb - self._model_vram_gb
        if available <= 0:
            return 1
        return max(1, int(available / self._kv_slot_gb))

    def _calc_duration(
        self, n_parallel: int, inet_down_mbps: float, queue_size: int
    ) -> int:
        """Compute instance TTL in seconds.

        Vast.ai enforces this server-side — the instance is destroyed at TTL
        even if our process has crashed, guarding against runaway billing.

        Components:
          startup_secs   = model download time + server init (60s)
          inference_secs = ceil(queue / n_parallel) batches × avg_secs/record
          overhead_secs  = 30 min safety buffer

        Bug note: an early version always received queue_size=0 here (the caller
        passed a hard-coded 0 instead of the actual pending count). This caused
        inference_secs = 0 → TTL = ~31 min always → instances were destroyed
        mid-batch when processing 1000+ records.
        """
        model_mb = self._model_vram_gb * 1024
        download_secs = (model_mb * 8) / inet_down_mbps
        startup_secs = download_secs + 60

        batches = max(1, math.ceil(queue_size / n_parallel))
        inference_secs = batches * self._avg_inference_secs

        overhead_secs = 30 * 60

        estimated = int(startup_secs + inference_secs + overhead_secs)
        hard_cap = self._max_duration_hours * 3600
        return min(estimated, hard_cap)

    def _build_onstart(
        self, n_parallel_per_gpu: int, ctx_per_slot: int = 4096
    ) -> str:
        """Return the bash onstart script that downloads the GGUF and starts N llama-servers.

        Uses aria2c (multi-connection download) and runs one llama-server process
        per GPU with CUDA_VISIBLE_DEVICES={i}. All output goes to /var/log/onstart.log.
        """
        total_ctx = max(4096, ctx_per_slot) * n_parallel_per_gpu
        hf_url = (
            f"https://huggingface.co/{self._hf_repo}/resolve/main/{self._hf_file}"
        )

        server_cmds = []
        for i in range(self._num_gpus):
            port = 8000 + i
            log_file = f"/var/log/llama_gpu{i}.log"
            cmd = (
                f'env CUDA_VISIBLE_DEVICES={i} "$LLAMA_BIN" \\\n'
                f"    --model /models/{self._hf_file} \\\n"
                f"    --host 0.0.0.0 \\\n"
                f"    --port {port} \\\n"
                "    --n-gpu-layers 99 \\\n"
                f"    --ctx-size {total_ctx} \\\n"
                "    --n-predict 800 \\\n"
                f"    --parallel {n_parallel_per_gpu} \\\n"
                # --reasoning-budget 0 suppresses thinking tokens (~300 tokens
                # per call on Qwen3 series) for a significant throughput gain.
                f"    --reasoning-budget 0 > {log_file} 2>&1 &"
            )
            server_cmds.append(cmd)
            server_cmds.append(
                f'echo "[onstart] llama-server GPU {i} started on port {port} (PID $!)"'
            )

        onstart_lines = [
            "#!/bin/bash",
            "exec > /var/log/onstart.log 2>&1",
            "set -ex",
            "mkdir -p /models",
            "apt-get update && apt-get install -y aria2",
            f'echo "[onstart] Downloading {self._hf_file} with aria2c ..."',
            f"aria2c -x 16 -s 16 -k 1M -c \\",
            f'    "{hf_url}" \\',
            f'    -d /models -o "{self._hf_file}"',
            'echo "[onstart] Download complete. Locating llama-server..."',
            "LLAMA_BIN=$(command -v llama-server 2>/dev/null \\",
            "    || find /usr /app /opt /root -name 'llama-server' -type f 2>/dev/null | head -1 \\",
            "    || echo /llama-server)",
            'echo "[onstart] llama-server binary: $LLAMA_BIN"',
        ] + server_cmds

        return "\n".join(onstart_lines) + "\n"

    def _create_instance(
        self, n_parallel_per_gpu: int, duration_secs: int
    ) -> int:
        """Rent the selected offer. Returns the new instance ID."""
        url = f"{_VASTAI_API}/asks/{self._offer_id}/"
        payload = {
            "image": self._vast_image,
            "disk": self._disk_gb,
            "onstart": self._build_onstart(
                n_parallel_per_gpu, ctx_per_slot=self._ctx_per_slot
            ),
            "duration": duration_secs,
        }
        print(f"[vastai] Using image: {self._vast_image}")
        resp = requests.put(
            url, json=payload, headers=self._headers(), timeout=30
        )
        if not resp.ok:
            raise RuntimeError(
                f"Vast.ai create instance failed (HTTP {resp.status_code}): "
                f"{resp.text}"
            )
        data = resp.json()
        instance_id: int = data["new_contract"]
        print(
            f"Vast.ai instance created: id={instance_id} "
            f"(offer={self._offer_id})"
        )
        return instance_id

    def _get_instance(self, instance_id: int) -> dict:
        """Fetch current instance state from Vast.ai.

        Uses GET /instances/ (list all) filtered by id — Vast.ai has no
        single-instance GET endpoint.
        """
        resp = requests.get(
            f"{_VASTAI_API}/instances/",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        for inst in resp.json().get("instances", []):
            if isinstance(inst, dict) and inst.get("id") == instance_id:
                return inst
        return {}  # not yet visible — caller will retry

    def _destroy_instance(self, instance_id: int) -> None:
        """Destroy (terminate) the rented instance."""
        url = f"{_VASTAI_API}/instances/{instance_id}/"
        try:
            resp = requests.delete(url, headers=self._headers(), timeout=10)
            if resp.ok:
                print(f"Vast.ai instance {instance_id} destroyed.")
            else:
                print(
                    f"Warning: Vast.ai destroy returned HTTP {resp.status_code}: "
                    f"{resp.text}"
                )
        except requests.RequestException as exc:
            # Non-fatal — the instance may already be gone or expired via TTL
            print(f"Warning: Vast.ai destroy request failed: {exc}")

    def _free_port(self) -> int:
        """Return an available local TCP port."""
        with socket.socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_ssh_tunnel(self, instance: dict) -> tuple[str, ...]:
        """Open one SSH tunnel process per GPU port on the instance.

        Each GPU gets an isolated ssh -N process forwarding one local port to
        localhost:800N inside the container.  Splitting per-GPU bounds the
        concurrent channel count per SSH session to slots_per_GPU rather than
        total_workers, which prevents SSH daemon channel exhaustion under high
        worker counts (see iter 08 post-mortem).

        Returns tuple of local base_urls:
          ("http://localhost:{port0}/v1", "http://localhost:{port1}/v1", ...)
        """
        import os as _os
        import time as _time

        ssh_host = instance.get("ssh_host", "")
        ssh_port = instance.get("ssh_port")
        if not ssh_host or not ssh_port:
            raise RuntimeError(
                f"[vastai] Missing SSH details: "
                f"ssh_host={ssh_host!r}, ssh_port={ssh_port!r}"
            )

        key_path = _os.path.expanduser(self._ssh_key) if self._ssh_key else ""
        if key_path and not _os.path.exists(key_path):
            raise RuntimeError(
                f"[vastai] SSH key not found: {key_path!r}. "
                "Set LLM_VAST_SSH_KEY to your private key path."
            )

        local_ports = [self._free_port() for _ in range(self._num_gpus)]
        # Store for mid-batch reconnection via reconnect_on_error()
        self._ssh_host = ssh_host
        self._ssh_port = int(ssh_port)
        self._local_ports = local_ports

        # Start one process per GPU — all in parallel before checking
        for i, lp in enumerate(local_ports):
            cmd = [
                "ssh", "-N",
                "-L", f"{lp}:localhost:{8000 + i}",
                "-p", str(ssh_port),
                f"root@{ssh_host}",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=3",
            ]
            if key_path:
                cmd += ["-i", key_path]
            self._tunnel_procs.append(
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            )

        # Give all SSH processes ~5s to connect; any that exit immediately failed
        _time.sleep(5)
        failed = []
        for i, proc in enumerate(self._tunnel_procs):
            if proc.poll() is not None:
                stderr = _read_proc_stderr(proc)
                failed.append(f"GPU{i} exit={proc.returncode}: {stderr.strip()}")
        if failed:
            raise RuntimeError(
                f"[vastai] SSH tunnel(s) exited immediately: {'; '.join(failed)}"
            )

        port_info = ", ".join(
            f"localhost:{lp} → container:{8000+i} (GPU{i})"
            for i, lp in enumerate(local_ports)
        )
        print(f"[vastai] SSH tunnels opened ({self._num_gpus}× isolated): {port_info}")
        return tuple(f"http://localhost:{lp}/v1" for lp in local_ports)

    def reconnect_on_error(self, exc: Exception) -> bool:
        """Restart any dead SSH tunnel processes and return True if reconnected.

        Called by BaseLLMBackend.complete() on request failure. Only one thread
        performs the reconnect; others wait on the lock and skip if the tunnels
        are already back up by the time they acquire it.
        """
        import os as _os
        import time as _time

        if self._instance_id is None or not self._ssh_host:
            return False

        with self._tunnel_reconnect_lock:
            dead = [i for i, p in enumerate(self._tunnel_procs) if p.poll() is not None]
            if not dead:
                # Another thread already reconnected — tunnels are alive again
                return True

            print(
                f"[vastai] {len(dead)} SSH tunnel(s) dead (GPU {dead}). Reconnecting ..."
            )
            key_path = _os.path.expanduser(self._ssh_key) if self._ssh_key else ""
            for i in dead:
                lp = self._local_ports[i]
                cmd = [
                    "ssh", "-N",
                    "-L", f"{lp}:localhost:{8000 + i}",
                    "-p", str(self._ssh_port),
                    f"root@{self._ssh_host}",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ServerAliveInterval=10",
                    "-o", "ServerAliveCountMax=3",
                ]
                if key_path:
                    cmd += ["-i", key_path]
                self._tunnel_procs[i] = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )

            _time.sleep(3)
            still_dead = [i for i in dead if self._tunnel_procs[i].poll() is not None]
            if still_dead:
                print(f"[vastai] SSH reconnect failed for GPU(s): {still_dead}")
                return False

            print(f"[vastai] SSH tunnel(s) reconnected for GPU(s): {dead}")
            return True

    def _stop_ssh_tunnel(self) -> None:
        """Terminate all per-GPU SSH tunnel subprocesses."""
        for proc in self._tunnel_procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._tunnel_procs.clear()
        print("[vastai] SSH tunnels closed.")

    # ------------------------------------------------------------------
    # BaseLLMBackend lifecycle
    # ------------------------------------------------------------------

    def startup(self, queue_size: int = 0) -> None:
        """Provision a GPU instance and make it ready to serve requests.

        Steps:
          0. Validate HF model exists (free HEAD request — no billing risk)
          1. Auto-select offer (tiered fallback 8→4→2→1 GPUs) or use fixed offer_id
          2. Compute n_parallel from VRAM + instance TTL from queue_size
          3. Create (rent) the instance
          4. Poll until actual_status == "running"
          5. Open SSH tunnel per GPU port
          6. Poll llama-server health until model is loaded (GGUF download)
        """
        # Step 0
        self._check_hf_model()

        # Step 1 + 2
        if not self._offer_id:
            offer_id, vram_gb, inet_down_mbps = self._find_offer()
            self._offer_id = str(offer_id)

            if self._single_gpu_mode and self._num_gpus > 1:
                print(
                    f"[vastai] SINGLE_GPU_MODE: rented bundle has {self._num_gpus}x GPUs "
                    f"but only CUDA_VISIBLE_DEVICES=0 will be started. "
                    f"Remaining cards idle (still billed as part of the bundle)."
                )
                self._num_gpus = 1

            n_parallel_per_gpu = self._calc_parallel_slots(vram_gb)
            n_parallel = n_parallel_per_gpu * self._num_gpus
            duration_secs = self._calc_duration(
                n_parallel, inet_down_mbps, queue_size
            )
            available_vram = vram_gb - self._model_vram_gb
            print(
                f"[vastai] Parallel slot formula:\n"
                f"  ({vram_gb:.0f}GB VRAM - {self._model_vram_gb:.0f}GB model) "
                f"/ {self._kv_slot_gb:.1f}GB per KV slot "
                f"= {n_parallel_per_gpu} slots/GPU × {self._num_gpus} GPU(s) "
                f"= {n_parallel} total workers\n"
                f"  available_vram={available_vram:.0f}GB"
            )
            print(
                f"[vastai] Instance TTL: {duration_secs // 60}min "
                f"(queue={queue_size}, n_parallel={n_parallel})"
            )
        else:
            n_parallel = self.n_parallel if self.n_parallel > 1 else 1
            n_parallel_per_gpu = max(1, n_parallel // self._num_gpus)
            duration_secs = self._max_duration_hours * 3600
            print(
                f"[vastai] Using fixed offer_id={self._offer_id}, "
                f"TTL={duration_secs // 60}min"
            )

        # Write back so the batch runner spawns the right number of workers.
        # Bug note: an early version computed n_parallel_per_gpu but never
        # wrote it back to self.n_parallel. ThreadPoolExecutor always spawned
        # 1 worker despite the llama-server having 36 slots available.
        self.n_parallel = n_parallel

        # Step 3
        total_ctx = max(4096, self._ctx_per_slot) * n_parallel_per_gpu
        print(
            f"[vastai] llama-server config per GPU:\n"
            f"  image:        {self._vast_image}\n"
            f"  model:        {self._hf_repo}/{self._hf_file}\n"
            f"  GPUs started: {self._num_gpus} (CUDA_VISIBLE_DEVICES 0..{self._num_gpus - 1})\n"
            f"  ports:        {', '.join(str(8000 + i) for i in range(self._num_gpus))}\n"
            f"  --parallel:   {n_parallel_per_gpu}\n"
            f"  --ctx-size:   {total_ctx}  ({self._ctx_per_slot} tokens/slot × {n_parallel_per_gpu} slots)\n"
            f"  --reasoning-budget: 0  (thinking disabled)\n"
            f"  disk:         {self._disk_gb}GB"
        )
        self._instance_id = self._create_instance(n_parallel_per_gpu, duration_secs)

        # Step 4: poll until running
        deadline = time.monotonic() + self._timeout_running
        print(
            f"Waiting for Vast.ai instance to reach 'running' "
            f"(timeout {self._timeout_running}s) ..."
        )
        while time.monotonic() < deadline:
            instance = self._get_instance(self._instance_id)
            status = instance.get("actual_status", "")
            print(f"  status={status}")
            if status == "running":
                break
            time.sleep(_POLL_INTERVAL)
        else:
            self._destroy_instance(self._instance_id)
            raise RuntimeError(
                f"Vast.ai instance {self._instance_id} did not reach 'running' "
                f"within {self._timeout_running}s. Instance destroyed."
            )

        # Step 5: SSH tunnel (retry until sshd is accepting connections)
        deadline = time.monotonic() + self._timeout_ssh
        print(f"Waiting for SSH (timeout {self._timeout_ssh}s) ...")
        while time.monotonic() < deadline:
            try:
                self._base_urls = self._start_ssh_tunnel(instance)
                break
            except RuntimeError as exc:
                self._stop_ssh_tunnel()
                print(f"  {exc} — retrying in {_POLL_INTERVAL}s ...")
                time.sleep(_POLL_INTERVAL)
        else:
            instance_id = self._instance_id
            self._instance_id = None
            self._destroy_instance(instance_id)
            raise RuntimeError(
                f"SSH tunnel to instance {instance_id} never became available "
                f"within {self._timeout_ssh}s. Instance destroyed."
            )
        print(f"SSH tunnel open. llama-server endpoints: {self._base_urls}")

        # Step 6: poll health until GGUF is loaded
        deadline = time.monotonic() + self._timeout_load
        print(
            f"Waiting for llama-server(s) to load model "
            f"(timeout {self._timeout_load}s) ..."
        )
        while time.monotonic() < deadline:
            for _i, _proc in enumerate(self._tunnel_procs):
                if _proc.poll() is not None:
                    stderr = _read_proc_stderr(_proc)
                    raise RuntimeError(
                        f"[vastai] SSH tunnel GPU{_i} died unexpectedly "
                        f"(exit={_proc.returncode}): {stderr.strip()}"
                    )
            if self.health_check():
                print("All llama-servers ready.")
                return
            print("  llama-servers not ready yet ...")
            time.sleep(_POLL_INTERVAL)

        instance_id = self._instance_id
        self._instance_id = None
        self._destroy_instance(instance_id)
        raise RuntimeError(
            f"llama-servers did not become healthy within "
            f"{self._timeout_load}s. Instance destroyed."
        )

    def shutdown(self) -> None:
        """Close SSH tunnel and destroy the rented instance.

        Always called via finally — even if the batch raised an exception.
        If LLM_VAST_KEEP_ON_FAILURE=true, destruction is skipped so the instance
        can be SSH'd into for log inspection (llama-server stderr, dmesg, etc.).
        Remember to destroy the instance manually from the Vast.ai console afterward.
        """
        self._stop_ssh_tunnel()
        if self._instance_id is not None:
            if self._keep_on_failure:
                print(
                    f"[vastai] KEEP_ON_FAILURE=true — instance {self._instance_id} "
                    f"preserved. Inspect logs via Vast.ai console or:\n"
                    f"  vastai ssh-url {self._instance_id}\n"
                    f"Then destroy manually: vastai destroy instance {self._instance_id}"
                )
            else:
                self._destroy_instance(self._instance_id)
                self._instance_id = None
