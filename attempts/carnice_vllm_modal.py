# ---
# pytest: false
# ---

# # Serve Carnice-27b with TurboQuant KV cache on Modal
#
# Carnice-27b is a Qwen3.5-27B fine-tune for hermes-agent tool-calling.
# TurboQuant (PR #38479, merged) compresses KV cache 2.6-4.9x,
# letting us serve long contexts on a single A100-80GB at $2.50/hr.
#
# Usage:
#   modal run carnice_vllm_modal.py          # test locally
#   modal deploy carnice_vllm_modal.py        # deploy persistent endpoint

import json
import subprocess
import time

import aiohttp
import modal

# ── Model config ──────────────────────────────────────────────────────────

MODEL_NAME = "kai-os/Carnice-27b"
# Pin revision to avoid surprises when repo updates
MODEL_REVISION = "05340447be66231e1801c042a56471c073bc12e9"  # update if needed

# ── TurboQuant KV cache compression ──────────────────────────────────────
#
# Options (quality ↓, compression ↑):
#   turboquant_k8v4    — 2.6x compression, ~96% quality (recommended)
#   turboquant_4bit_nc — 3.8x compression, ~93% quality
#   turboquant_k3v4_nc — 4.3x compression, ~87% quality
#   turboquant_3bit_nc — 4.9x compression, ~80% quality
#
# k8v4 = FP8 keys + 4-bit values. Near-lossless, faster on long sequences.
# "nc" variants add norm correction for better quality at lower bit widths.

KV_CACHE_DTYPE = "turboquant_k8v4"

# Skip boundary layers from TQ compression — first/last layers keep FP16.
# Protects embedding-adjacent representations. Set to empty string to disable.
KV_CACHE_SKIP_LAYERS = "0,1"  # skip first 2 layers

# ── Security ───────────────────────────────────────────────────────────────
#
# Bearer token for API authentication. All requests must include:
#   Authorization: Bearer <CARNICE_API_KEY>
# Without this, the server rejects requests. Prevents unauthorized use.
# Change this to your own token before deploying.

CARNICE_API_KEY = "CHANGE-ME"  # placeholder — use Modal Secret in production

# ── Infrastructure ────────────────────────────────────────────────────────

GPU_TYPE = "A100-80GB"  # $2.50/hr, enough VRAM for 27B BF16 + KV cache
N_GPU = 1
VLLM_PORT = 8000
MINUTES = 60

# Keep container alive 7 minutes after last request, then scale to zero.
# You only pay for time the container is running.
SCALEDOWN_WINDOW = 7 * MINUTES

# ── Container image ───────────────────────────────────────────────────────
#
# TurboQuant merged into vLLM main (PR #38479, Apr 15 2026) but not yet
# in a pip release. We install from PR #39316 (feat/qwen35-text branch)
# which adds native Qwen3_5ForCausalLM text-only support AND includes
# TurboQuant (rebased onto main April 18).
#
# Without PR #39316, vLLM only supports Qwen3_5ForConditionalGeneration
# (VLM path) and fails on text-only checkpoints like Carnice-27b.

VLLM_BRANCH = "feat/qwen35-text"
VLLM_FORK = "https://github.com/jefcoder/vllm.git"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install(
        "aiohttp",
        "setuptools>=77.0.3",  # PEP 639 license format required by vLLM main
        "setuptools_scm",
        "ninja",
    )
    # Install vLLM from PR #39316 branch (Qwen3.5 text-only + TurboQuant).
    # Building from source with A100 builder (~80GB RAM) for CUDA kernels.
    # --no-build-isolation reuses the setuptools/ninja we just installed.
    .run_commands(
        f"pip install --no-build-isolation git+{VLLM_FORK}@{VLLM_BRANCH}",
        gpu="A100",  # A100 builder: ~80GB RAM + CUDA toolkit for compilation
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ── Persistent volumes ────────────────────────────────────────────────────
#
# Cache model weights + vLLM JIT artifacts so cold starts are fast
# after the first download.

hf_cache_vol = modal.Volume.from_name("carnice-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("carnice-vllm-cache", create_if_missing=True)

# ── App definition ────────────────────────────────────────────────────────

app = modal.App("carnice-27b-turboquant")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    memory=32768,               # 32GB RAM — vllm pip install + model loading needs headroom
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=10 * MINUTES,  # container startup grace period
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=50)  # max concurrent requests per replica
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Launch vLLM server with TurboQuant KV cache compression."""

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", CARNICE_API_KEY,  # require bearer token auth
        "--uvicorn-log-level=info",

        # Tensor parallelism (1 GPU)
        "--tensor-parallel-size", str(N_GPU),

        # ── TurboQuant KV cache ──
        "--kv-cache-dtype", KV_CACHE_DTYPE,
    ]

    # Boundary layer protection — skip TQ on first N layers
    if KV_CACHE_SKIP_LAYERS:
        cmd += ["--kv-cache-dtype-skip-layers", KV_CACHE_SKIP_LAYERS]

    cmd += [
        # ── Tool calling (Carnice is hermes-agent trained) ──
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",

        # PR #39316 adds native Qwen3_5ForCausalLM support — no hf-overrides needed

        # Longer max context for agentic use (TQ makes this cheap)
        "--max-model-len", "32768",
    ]

    print("Launching vLLM server:")
    print(" ".join(cmd))

    subprocess.Popen(" ".join(cmd), shell=True)


# ── Test entrypoint ────────────────────────────────────────────────────────
#
# Run: modal run carnice_vllm_modal.py
#
# Spins up a fresh server replica and tests it with:
# 1. Health check
# 2. Simple generation
# 3. Tool-calling test


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES):
    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        # Health check
        print("Running health check...")
        async with session.get("/health", timeout=aiohttp.ClientTimeout(total=test_timeout)) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
            print("✅ Server healthy")

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
            {
                "role": "user",
                "content": "What's the weather in Sacramento, CA?",
            },
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        await _send_request(session, tool_messages, tools=tools)

        # Thinking/reasoning test
        print("\n🧠 Reasoning test (thinking enabled):")
        think_messages = [
            {"role": "user", "content": "What is 17 * 23? Think step by step."},
        ]
        await _send_request(session, think_messages, extra_params={"chat_template_kwargs": {"enable_thinking": True}})


async def _send_request(session, messages, tools=None, extra_params=None):
    """Send a streaming chat completion request and print the response."""
    payload = {
        "messages": messages,
        "model": MODEL_NAME,
        "stream": True,
        "max_tokens": 512,
    }
    if tools:
        payload["tools"] = tools
    if extra_params:
        payload.update(extra_params)

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {CARNICE_API_KEY}",
    }

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers,
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
            # Handle both content and reasoning_content (thinking)
            content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            if content:
                print(content, end="", flush=True)
            # Handle tool calls
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        print(f"\n🔧 Tool call: {fn['name']}({fn.get('arguments', '')})", end="")
    print()  # newline after streaming
