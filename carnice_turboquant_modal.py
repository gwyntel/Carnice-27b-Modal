# ---
# pytest: false
# ---

# # Serve Carnice-27b with llama.cpp TurboQuant on Modal
#
# Uses **AmesianX/TurboQuant** fork — Google DeepMind's KV cache compression.
#
# ## Why TurboQuant?
#
# - **4-6x KV cache compression** via 3-4 bit quantization (vs 16-bit FP16)
# - **Training-free, model-agnostic** — no fine-tuning needed
# - **Enables longer contexts** on same hardware
# - **Speed gains under memory pressure** — 2-3x throughput when FP16 pushes GPU into swap
#
# ## Compression Modes
#
# | Mode | Bits | Description |
# |------|------|-------------|
# | `tbq3` | 3 | TurboQuant 3-bit (max compression, slight quality drop) |
# | `tbq4` | 4 | TurboQuant 4-bit (sweet spot, near-lossless) |
# | `q8_0` | 8 | Standard quantized (baseline) |
#
# ## Usage
#
# ```bash
# # Deploy TurboQuant endpoint
# modal deploy carnice_turboquant_modal.py
#
# # Test with different compression modes
# modal run carnice_turboquant_modal.py --cache-mode tbq4
# ```
#
# ## References
#
# - **Paper:** [TurboQuant (ICLR 2026, Google DeepMind)](https://arxiv.org/abs/2504.19874)
# - **Fork:** [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant)
# - **Original:** [llama.cpp](https://github.com/ggml-org/llama.cpp)

import os
import subprocess
import time

import modal
import modal.experimental

# ── Model config ──────────────────────────────────────────────────────────

GGUF_REPO = "kai-os/Carnice-27b-GGUF"
GGUF_FILE = "Carnice-27b-Q4_K_M.gguf"  # 16.5GB
SERVED_MODEL_NAME = "kai-os/Carnice-27b"

# ── TurboQuant Configuration ─────────────────────────────────────────────

# KV Cache compression modes:
# - "tbq3": 3-bit TurboQuant (max compression, ~4-6x savings)
# - "tbq4": 4-bit TurboQuant (sweet spot, near-lossless)
# - "q8_0": Standard 8-bit quantized (baseline, no compression)
CACHE_MODE = "tbq4"  # Default: 4-bit sweet spot

# ── Universal Secret Configuration ─────────────────────────────────────────

SECRET_NAME = "carnice-api-key"
SECRET_KEY = "CARNICE_API_KEY"

def _get_api_key():
    """Get API key from Modal Secret (runtime) or local env (testing)."""
    key = os.environ.get(SECRET_KEY)
    if not key or key == "CHANGE-ME":
        raise RuntimeError(
            f"No {SECRET_KEY} found. Create a Modal Secret:\n"
            f"  modal secret create {SECRET_NAME} {SECRET_KEY}=sk-...\n"
            f"Or set {SECRET_KEY} env var for local testing."
        )
    return key

# ── Infrastructure ────────────────────────────────────────────────────────

GPU_TYPE = "A100-80GB"  # $2.50/hr
PORT = 8000
MINUTES = 60
SCALEDOWN_WINDOW = 7 * MINUTES

# TurboQuant enables MASSIVE context lengths:
# Q4_K_M (16.5GB) on A100-80GB leaves ~60GB for KV cache
# With tbq4 (4-bit KV): ~240GB effective KV capacity
# Qwen3.5-27B: 28 layers, head_dim=128, 4 KV heads (GQA)
# KV per token (tbq4) ≈ 2 * 28 * 4 * 128 * 0.5 byte = 14,336 bytes ≈ 14KB
# 240GB / 14KB ≈ 18M tokens (parallel=1), ~4.5M (parallel=4)
#
# Conservative: 256K context (model's training length)
CONTEXT_LENGTH = 262144

# ── Container image ───────────────────────────────────────────────────────

# Use AmesianX/TurboQuant fork via ai-dock/llama.cpp-cuda
# This includes TurboQuant KV cache compression
LLAMA_CPP_RELEASE = "b8851"  # latest as of Apr 20 2026

llamacpp_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "tar", "libgomp1")
    # Download pre-built CUDA binaries from ai-dock/llama.cpp-cuda releases
    # These include TurboQuant support from AmesianX fork
    .run_commands(
        f"curl -fsSL https://github.com/ai-dock/llama.cpp-cuda/releases/download/{LLAMA_CPP_RELEASE}/llama.cpp-{LLAMA_CPP_RELEASE}-cuda-12.8.tar.gz -o /tmp/llama-cpp.tar.gz",
        "mkdir -p /opt/llama.cpp",
        "tar -xzf /tmp/llama-cpp.tar.gz -C /opt/llama.cpp",
        "rm /tmp/llama-cpp.tar.gz",
    )
    .pip_install(
        "huggingface_hub[cli]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "LD_LIBRARY_PATH": "/opt/llama.cpp/cuda-12.8:${LD_LIBRARY_PATH:-}",
    })
)

# ── Persistent volume ─────────────────────────────────────────────────────

model_vol = modal.Volume.from_name("carnice-gguf-cache", create_if_missing=True)
MODEL_DIR = "/models"

# ── App definition ────────────────────────────────────────────────────────

app = modal.App("carnice-27b-turboquant")


@app.cls(
    image=llamacpp_image,
    gpu=f"{GPU_TYPE}:1",
    memory=32768,  # 32GB RAM
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    volumes={MODEL_DIR: model_vol},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=["us-east"],
    exit_grace_period=15,
)
class CarniceTurboQuant:
    """llama.cpp server for Carnice-27b with TurboQuant KV cache compression."""

    @modal.enter()
    def startup(self):
        """Download GGUF if needed, then start llama-server with TurboQuant."""

        CARNICE_API_KEY = _get_api_key()
        gguf_path = f"{MODEL_DIR}/{GGUF_FILE}"

        # Download model if not cached
        if not os.path.exists(gguf_path):
            print(f"📥 Downloading {GGUF_FILE} from {GGUF_REPO}...")
            subprocess.run(
                [
                    "hf", "download",
                    GGUF_REPO,
                    GGUF_FILE,
                    "--local-dir", MODEL_DIR,
                ],
                check=True,
            )
            model_vol.commit()
            print(f"✅ Downloaded {GGUF_FILE}")
        else:
            print(f"✅ Model cached: {gguf_path}")

        # Launch llama-server with TurboQuant
        cmd = [
            "/opt/llama.cpp/cuda-12.8/llama-server",
            "--model", gguf_path,
            "--alias", SERVED_MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--api-key", CARNICE_API_KEY,
            # Context length
            "--ctx-size", str(CONTEXT_LENGTH),
            # Offload all layers to GPU
            "--n-gpu-layers", "-1",
            # Parallel sequences
            "--parallel", "1",
            # Flash attention
            "--flash-attn", "auto",
            # ── TURBOQUANT KV CACHE COMPRESSION ──
            # tbq3 = 3-bit (max compression, ~4-6x savings)
            # tbq4 = 4-bit (sweet spot, near-lossless)
            # q8_0 = 8-bit (baseline, no compression)
            "--cache-type-k", CACHE_MODE,
            "--cache-type-v", CACHE_MODE,
            # Metrics
            "--metrics",
        ]

        print(f"🚀 Launching llama-server with TurboQuant ({CACHE_MODE}):")
        print(f"   Mode: {CACHE_MODE} ({'3-bit' if CACHE_MODE == 'tbq3' else '4-bit' if CACHE_MODE == 'tbq4' else '8-bit'} KV cache)")
        print(f"   Expected compression: {'~4-6x' if CACHE_MODE in ['tbq3', 'tbq4'] else 'baseline'}")
        print(" ".join(cmd))

        # Print available TurboQuant flags
        help_rc = subprocess.run(
            ["/opt/llama.cpp/cuda-12.8/llama-server", "--help"],
            capture_output=True, text=True,
        )
        print("\n🔧 Available TurboQuant flags:")
        for line in help_rc.stdout.split("\n"):
            if "cache-type" in line.lower() or "turbo" in line.lower() or "tbq" in line.lower():
                print(f"   {line.strip()}")

        self.process = subprocess.Popen(cmd)
        self._wait_ready(timeout=5 * MINUTES)
        self._warmup(CARNICE_API_KEY)

    @modal.exit()
    def stop(self):
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

    def _wait_ready(self, timeout=5 * MINUTES):
        """Poll /health until llama-server is ready."""
        import urllib.request
        import urllib.error

        deadline = time.time() + timeout
        while time.time() < deadline:
            rc = self.process.poll()
            if rc is not None:
                print(f"❌ llama-server exited with code {rc}")
                raise subprocess.CalledProcessError(rc, cmd="llama-server")
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                if resp.status == 200:
                    print("✅ llama-server healthy")
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(5)
        raise TimeoutError(f"llama-server not healthy within {timeout}s")

    def _warmup(self, api_key: str):
        """Send a simple request to warm up CUDA kernels."""
        import urllib.request
        import json

        try:
            data = json.dumps({
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            resp = urllib.request.urlopen(req, timeout=120)
            if resp.status == 200:
                print("🔥 Warmup complete")
            else:
                print(f"⚠️  Warmup returned {resp.status}")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")


# ── Test entrypoint ────────────────────────────────────────────────────────

_TEST_API_KEY = os.environ.get(SECRET_KEY)


@app.local_entrypoint()
async def test():
    """Test the deployed endpoint — simple generation + tool-calling."""
    import aiohttp
    import asyncio
    import json

    if not _TEST_API_KEY:
        print(f"⚠️  No {SECRET_KEY} set. Set it in your environment to run tests.")
        return

    url = (await CarniceTurboQuant._experimental_get_flash_urls.aio())[0]
    print(f"Server URL: {url}")

    headers = {
        "Authorization": f"Bearer {_TEST_API_KEY}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        # Health check
        print("Running health check...")
        async with session.get("/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                print("✅ Server healthy")
            else:
                print(f"❌ Health check failed: {resp.status}")
                return

        # Simple generation
        print("\n📝 Simple generation test:")
        await _send_request(session, [
            {"role": "system", "content": "You are a helpful AI assistant with a dry sense of humor."},
            {"role": "user", "content": "Explain transformer attention in one paragraph."},
        ])

        # Tool-calling test
        print("\n🔧 Tool-calling test:")
        await _send_request(session, [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What's the weather in Sacramento, CA?"},
        ], tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g. Sacramento, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        }])


async def _send_request(session, messages, tools=None):
    """Send a streaming chat completion request and print the response."""
    import json

    payload = {
        "messages": messages,
        "model": SERVED_MODEL_NAME,
        "stream": True,
        "max_tokens": 512,
    }
    if tools:
        payload["tools"] = tools

    async with session.post(
        "/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                print(content, end="", flush=True)
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        print(f"\n🔧 Tool call: {fn['name']}({fn.get('arguments', '')})", end="")
    print()
