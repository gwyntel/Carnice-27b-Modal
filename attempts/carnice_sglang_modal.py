# ---
# pytest: false
# ---

# # Serve Carnice-27b with SGLang on Modal
#
# Carnice-27b is a Qwen3.5-27B fine-tune for hermes-agent tool-calling.
#
# Key workaround: The model's config.json declares model_type=qwen3_5_text
# which transformers doesn't recognize. We fix this via sitecustomize.py
# which registers qwen3_5_text as an alias for qwen3_5 in transformers'
# CONFIG_MAPPING_NAMES at Python startup — before ANY SGLang code runs.
# This intercepts the error regardless of which code path triggers it.
#
# We also patch SGLang's qwen3_5.py EntryClass to register the text-only
# CausalLM classes (only VLM variants are registered by default).
#
# Usage:
#   modal deploy carnice_sglang_modal.py   # deploy persistent endpoint
#   modal run carnice_sglang_modal.py      # test locally

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

# ── Model config ──────────────────────────────────────────────────────────

MODEL_NAME = "kai-os/Carnice-27b"

# ── Security ───────────────────────────────────────────────────────────────

# ── Universal Secret Configuration ─────────────────────────────────────────
#
# Change SECRET_NAME to use your own Modal Secret, or set CARNICE_API_KEY env var.
SECRET_NAME = "carnice-api-key"
SECRET_KEY = "CARNICE_API_KEY"

import os
def _get_api_key():
    key = os.environ.get(SECRET_KEY)
    if not key or key == "CHANGE-ME":
        raise RuntimeError(f"Set {SECRET_KEY} env var or create Modal Secret: {SECRET_NAME}")
    return key

# ── Infrastructure ────────────────────────────────────────────────────────

GPU_TYPE = "A100-80GB"  # $2.50/hr, enough VRAM for 27B BF16 (54GB) + KV
N_GPU = 1
PORT = 8000
MINUTES = 60
TARGET_INPUTS = 50  # max concurrent requests per replica

# Keep container alive 7 minutes after last request, then scale to zero.
SCALEDOWN_WINDOW = 7 * MINUTES

# ── Container image ───────────────────────────────────────────────────────
#
# Use SGLang's official Docker image (includes CUDA, FlashInfer, etc).
# Then:
# 1. Install sitecustomize.py to register qwen3_5_text alias in transformers
# 2. Patch SGLang's qwen3_5.py EntryClass to add text-only CausalLM classes

sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.9-cu129-amd64-runtime",
    )
    .entrypoint([])  # silence chatty logs
    # Patch 1: sitecustomize.py — registers qwen3_5_text in transformers registry
    .add_local_file(
        "/home/hermes/projects/carnice-modal/sitecustomize.py",
        "/usr/local/lib/python3.12/dist-packages/sitecustomize.py",
        copy=True,
    )
    # Patch 2: SGLang EntryClass — add Qwen3_5ForCausalLM
    .add_local_file(
        "/home/hermes/projects/carnice-modal/patch_sglang.py",
        "/root/patch_sglang.py",
        copy=True,
    )
    .run_commands(
        "python3 /root/patch_sglang.py",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",  # fast HF downloads
        "SGLANG_ENABLE_JIT_DEEPGEMM": "1",  # JIT-compiled FP8 kernels
    })
)

# ── Persistent volumes ────────────────────────────────────────────────────
#
# Cache model weights + DeepGEMM JIT artifacts for fast cold starts.

hf_cache_vol = modal.Volume.from_name("carnice-hf-cache", create_if_missing=True)
dg_cache_vol = modal.Volume.from_name("carnice-dg-cache", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
DG_CACHE_PATH = "/root/.cache/deepgemm"

# ── App definition ────────────────────────────────────────────────────────

app = modal.App("carnice-27b-sglang")


@app.cls(
    image=sglang_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    memory=32768,  # 32GB RAM
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        DG_CACHE_PATH: dg_cache_vol,
    },
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=["us-east"],
    exit_grace_period=15,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class CarniceSGLang:
    """SGLang server for Carnice-27b with hermes tool-calling."""

    @modal.enter()
    def startup(self):
        """Verify sitecustomize loaded, then start the SGLang server."""
        # Quick check that sitecustomize.py took effect
        import subprocess as sp
        result = sp.run(
            ["python3", "-c",
             "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; "
             "print('qwen3_5_text' in CONFIG_MAPPING_NAMES)"],
            capture_output=True, text=True,
        )
        if "True" in result.stdout:
            print("✅ sitecustomize.py loaded: qwen3_5_text registered in transformers")
        else:
            print(f"⚠️ sitecustomize.py may not have loaded: {result.stdout.strip()} {result.stderr.strip()}")

        # Launch SGLang server
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path", MODEL_NAME,
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--tp", str(N_GPU),
            "--api-key", CARNICE_API_KEY,
            # Hermes tool-calling (Carnice uses hermes-agent template)
            "--tool-call-parser", "hermes",
            "--trust-remote-code",
            # Context length — without TurboQuant, FP16 KV limits us
            # 54GB weights + ~20GB KV cache = ~12K context on A100-80GB
            "--context-length", "12288",
            "--mem-fraction", "0.82",
            # Performance tuning
            "--cuda-graph-max-bs", str(TARGET_INPUTS * 2),
            "--enable-metrics",
            "--decode-log-interval", "100",
        ]

        print("Launching SGLang server:")
        print(" ".join(cmd))

        self.process = subprocess.Popen(cmd)
        self._wait_ready(timeout=8 * MINUTES)
        self._warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()

    def _wait_ready(self, timeout=8 * MINUTES):
        """Poll /health until SGLang is ready."""
        import requests as req

        deadline = time.time() + timeout
        while time.time() < deadline:
            rc = self.process.poll()
            if rc is not None:
                print(f"❌ SGLang process exited with code {rc}")
                raise subprocess.CalledProcessError(rc, cmd=self.process.args)
            try:
                resp = req.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
                resp.raise_for_status()
                print("✅ SGLang server healthy")
                return
            except (
                req.exceptions.ConnectionError,
                req.exceptions.HTTPError,
            ):
                time.sleep(5)
        raise TimeoutError(f"SGLang not healthy within {timeout}s")

    def _warmup(self):
        """Send a simple request to warm up JIT kernels."""
        import requests as req

        try:
            resp = req.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
                headers={"Authorization": f"Bearer {CARNICE_API_KEY}"},
                timeout=120,
            )
            if resp.ok:
                print("🔥 Warmup complete")
            else:
                print(f"⚠️  Warmup returned {resp.status_code}")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")


# ── Test entrypoint ────────────────────────────────────────────────────────

@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    url = (await CarniceSGLang._experimental_get_flash_urls.aio())[0]
    print(f"Server URL: {url}")

    headers = {
        "Authorization": f"Bearer {CARNICE_API_KEY}",
        "Content-Type": "application/json",
    }

    # Health check
    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        print("Running health check...")
        deadline = time.time() + test_timeout
        while time.time() < deadline:
            try:
                async with session.get("/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        print("✅ Server healthy")
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(3)

        # Simple generation
        print("\n📝 Simple generation test:")
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant with a dry sense of humor."},
            {"role": "user", "content": "Explain transformer attention in one paragraph."},
        ]
        await _send_request(session, messages)

        # Tool-calling test
        print("\n🔧 Tool-calling test:")
        tool_messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What's the weather in Sacramento, CA?"},
        ]
        tools = [
            {
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
            }
        ]
        await _send_request(session, tool_messages, tools=tools)


async def _send_request(session, messages, tools=None):
    """Send a streaming chat completion request and print the response."""
    payload = {
        "messages": messages,
        "model": MODEL_NAME,
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
            content = delta.get("content") or delta.get("reasoning_content")
            if content:
                print(content, end="", flush=True)
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        print(f"\n🔧 Tool call: {fn['name']}({fn.get('arguments', '')})", end="")
    print()
