# ---
# pytest: false
# ---

# # Serve Carnice-27b with llama.cpp on Modal
#
# Carnice-27b is a Qwen3.5-27B fine-tune for hermes-agent tool-calling.
#
# Why llama.cpp instead of SGLang/vLLM:
# - Qwen3.5 `model_type=qwen3_5_text` is unrecognized by transformers/SGLang
#   (sitecustomize.py patch didn't propagate to AutoConfig's lazy registry)
# - llama.cpp natively supports the `qwen35` architecture (PR #19468, Feb 2026)
# - Built-in `--tool-call-parser hermes` for native tool-calling
# - Pre-quantized GGUF (Q4_K_M = 16.5GB) leaves tons of VRAM for KV cache
# - No Python venv hell, no registry hacks, no source builds
#
# Usage:
#   modal deploy carnice_llama_modal.py   # deploy persistent endpoint
#   modal run carnice_llama_modal.py       # test locally

import subprocess
import time

import modal
import modal.experimental

# ── Model config ──────────────────────────────────────────────────────────

GGUF_REPO = "kai-os/Carnice-27b-GGUF"
GGUF_FILE = "Carnice-27b-Q4_K_M.gguf"  # 16.5GB, fits A100-80GB with massive KV headroom
# GGUF_FILE = "Carnice-27b-Q6_K.gguf"  # 22.1GB, better quality
# GGUF_FILE = "Carnice-27b-Q8_0.gguf"  # 28.6GB, near-original quality

SERVED_MODEL_NAME = "kai-os/Carnice-27b"

# ── Security ───────────────────────────────────────────────────────────────

CARNICE_API_KEY = "CHANGEMEOKAY"

# ── Infrastructure ────────────────────────────────────────────────────────

GPU_TYPE = "A100-80GB"  # $2.50/hr. Q4_K_M = 16.5GB weights + ~60GB KV headroom
PORT = 8000
MINUTES = 60

# Keep container alive 7 minutes after last request, then scale to zero.
SCALEDOWN_WINDOW = 7 * MINUTES

# Context length: with Q4_K_M (16.5GB) on A100-80GB, we have ~60GB for KV.
# Qwen3.5-27B has 28 layers, head_dim=128, 4 KV heads (GQA).
# KV per token (q8_0) ≈ 2 * 28 * 4 * 128 * 1 byte = 28,672 bytes ≈ 28KB
# 59GB / 28KB ≈ 2.2M tokens (parallel=1), ~550K (parallel=4)
# Model trained on 262144 (256K). 252K tested successfully — no OOM.
# MAXIMUM FAT CONTEXT: 262144 = full model training length, single slot.
CONTEXT_LENGTH = 262144

# ── Container image ───────────────────────────────────────────────────────
#
# Use a plain CUDA runtime image + download pre-built llama.cpp binaries from GitHub.
# We avoid the official ghcr.io images because their ENTRYPOINT conflicts with Modal's
# add_python (both server-cuda and full-cuda pass "python" as an arg to the entrypoint).
# The ai-dock/llama.cpp-cuda releases have pre-built binaries (no compile needed).
LLAMA_CPP_RELEASE = "b8851"  # latest as of Apr 20 2026

llamacpp_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "tar", "libgomp1")
    # Download pre-built CUDA binaries from ai-dock/llama.cpp-cuda releases
    # Built with CUDA 12.8 for architectures 7.5-12.0 (covers A100 sm_80)
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

app = modal.App("carnice-27b-llamacpp")


@app.cls(
    image=llamacpp_image,
    gpu=f"{GPU_TYPE}:1",
    memory=32768,  # 32GB RAM
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    volumes={MODEL_DIR: model_vol},
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=["us-east"],
    exit_grace_period=15,
)
class CarniceLlamaCpp:
    """llama.cpp server for Carnice-27b with hermes tool-calling."""

    @modal.enter()
    def startup(self):
        """Download GGUF if needed, then start llama-server."""
        import os

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

        # Launch llama-server
        cmd = [
            "/opt/llama.cpp/cuda-12.8/llama-server",
            "--model", gguf_path,
            "--alias", SERVED_MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--api-key", CARNICE_API_KEY,
            # Context — Q4_K_M leaves tons of room for KV on A100-80GB
            "--ctx-size", str(CONTEXT_LENGTH),
            # Offload all layers to GPU
            "--n-gpu-layers", "-1",
            # Parallel sequences — single slot gets the full 256K context
            "--parallel", "1",
            # Flash attention for efficiency (auto = enabled when available)
            "--flash-attn", "auto",
            # KV cache type — q8_0 is a good balance (turbo3 blocked by GQA bug #78)
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            # Metrics
            "--metrics",
        ]

        print("🚀 Launching llama-server:")
        print(" ".join(cmd))

        # Print available tool-call flags
        help_rc = subprocess.run(
            ["/opt/llama.cpp/cuda-12.8/llama-server", "--help"],
            capture_output=True, text=True,
        )
        for line in help_rc.stdout.split("\n"):
            if "tool" in line.lower() or "parser" in line.lower() or "grammar" in line.lower():
                print(f"  FLAG: {line.strip()}")

        self.process = subprocess.Popen(cmd)
        self._wait_ready(timeout=5 * MINUTES)
        self._warmup()

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
                # Print last 50 lines of stderr for debugging
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

    def _warmup(self):
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
                    "Authorization": f"Bearer {CARNICE_API_KEY}",
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

@app.local_entrypoint()
async def test():
    """Test the deployed endpoint — simple generation + tool-calling."""
    import aiohttp
    import asyncio
    import json

    url = (await CarniceLlamaCpp._experimental_get_flash_urls.aio())[0]
    print(f"Server URL: {url}")

    headers = {
        "Authorization": f"Bearer {CARNICE_API_KEY}",
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
