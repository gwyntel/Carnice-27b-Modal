# 🧠 Carnice-27b on Modal

**Serverless GPU inference for [Carnice-27b](https://huggingface.co/kai-os/Carnice-27b-GGUF)** — a Qwen3.5-27B fine-tune with native hermes-style tool-calling.

Deployed on [Modal](https://modal.com) with llama.cpp on A100-80GB GPUs. OpenAI-compatible API. Scales to zero.

## Why This Exists

Carnice-27b uses Qwen3.5's `qwen3_5_text` architecture, which **breaks** in standard Python inference frameworks:

- **vLLM**: Source build times out (607 CUDA compile targets). No native `qwen35` support.
- **SGLang**: `qwen3_5_text` not in transformers registry. Patching `EntryClass` and `get_config` didn't stick.
- **llama.cpp**: ✅ Natively supports `qwen35` architecture (PR #19468). Pre-built binaries. GGUF quantization. Just works.

**The pivot from SGLang/vLLM to llama.cpp saved this project.** No more Python venv hell, no registry hacks, no source builds.

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│  Your App    │────▶│  Modal Serverless Endpoint               │
│              │     │  (A100-80GB, $2.50/hr)                   │
│  OpenAI SDK  │◀────│                                          │
│  or curl     │     │  llama.cpp server (ai-dock b8851 build)  │
└──────────────┘     │  Q4_K_M GGUF (16.5GB weights)           │
                     │  ~60GB KV cache headroom                 │
                     │  32K context, flash attention, q8_0 KV   │
                     │  7-minute scaledown window               │
                     └──────────────────────────────────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| **Prompt eval** | ~677 tok/s (A100-80GB, Q4_K_M) |
| **Generation** | ~21-45 tok/s |
| **Context** | 32,768 tokens |
| **Parallel sequences** | 4 |
| **Model size** | 16.5GB (Q4_K_M), ~60GB VRAM left for KV cache |
| **Cold start** | ~2-3 min (first), ~30s (cached) |
| **Cost** | $2.50/hr, scales to zero after 7 min idle |

## Tool-Calling

Carnice-27b's chat template (**peg-native** format) handles tool-calling natively — no `--tool-call-parser` flag needed. The model generates proper `tool_calls` arrays with function names, arguments, and auto-generated IDs.

**Single tool:**
```json
{
  "finish_reason": "tool_calls",
  "message": {
    "tool_calls": [{
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\":\"Sacramento, CA\"}"
      }
    }]
  }
}
```

**Multi-tool:** Correctly sequences calls (e.g. `web_search` before `save_to_reading_list`).

**Reasoning:** Includes `reasoning_content` field showing the model's thinking about which tool to use.

## Quick Start

### Prerequisites

- [Modal](https://modal.com) account with API token configured (`modal token set`)
- Python 3.12+

### Deploy

```bash
git clone https://github.com/gwyntel/Carnice-27b-Modal.git
cd Carnice-27b-Modal
modal deploy carnice_llama_modal.py
```

### Test

```bash
# After deploying, grab your endpoint URL from the output, then:
curl -s "https://YOUR-ENDPOINT.modal.direct/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kai-os/Carnice-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### Tool-Calling Test

```bash
curl -s "https://YOUR-ENDPOINT.modal.direct/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kai-os/Carnice-27b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "What is the weather in Sacramento, CA?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City and state"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 512
  }'
```

## Configuration

### Quantization Options

| Format | Size | Quality | Recommendation |
|--------|------|---------|----------------|
| **Q4_K_M** | 16.5GB | Good (PPL 6.6053) | ✅ Sweet spot — max KV headroom |
| Q5_K_M | ~20GB | Better (PPL ~6.58) | If you have VRAM and need quality |
| Q6_K | ~23GB | Marginal gain (PPL 6.5456) | ❌ Uncanny valley — 7GB more for <1% PPL improvement |
| Q8_0 | ~29GB | Near-original (PPL 6.5352) | Only for benchmarks |

Edit `GGUF_FILE` in `carnice_llama_modal.py` to switch:

```python
GGUF_FILE = "Carnice-27b-Q4_K_M.gguf"  # Change this
```

### Key Parameters

```python
CONTEXT_LENGTH = 32768    # Max context window
GPU_TYPE = "A100-80GB"    # GPU selection
SCALEDOWN_WINDOW = 420    # 7 min idle before scale-down
```

## How It Works

### Image Build Strategy

The official `ghcr.io/ggml-org/llama.cpp:server-cuda` images crash-loop on Modal because their `ENTRYPOINT` conflicts with `add_python`. Instead:

1. Use `nvidia/cuda:12.8.1-runtime-ubuntu24.04` as base
2. Download [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda) pre-built binaries (release b8851)
3. Set `LD_LIBRARY_PATH` to include `/opt/llama.cpp/cuda-12.8/`
4. Install `libgomp1` (OpenMP runtime, required by llama.cpp)

This gives us a working llama-server in ~2 minutes vs 15+ minute failed source builds.

### Library Gotchas

- `libllama-common.so.0` not found → set `LD_LIBRARY_PATH=/opt/llama.cpp/cuda-12.8`
- `libgomp.so.1` missing → `apt_install("libgomp1")`
- `--flash-attn` requires explicit value in this build → use `--flash-attn auto` (not bare `--flash-attn`)
- `--tool-call-parser` flag doesn't exist in ai-dock b8851 → chat template handles tool-calling natively

### KV Cache Math

With Q4_K_M (16.5GB) on A100-80GB:
- ~60GB available for KV cache
- Qwen3.5-27B: 28 layers, head_dim=128, 4 KV heads (GQA)
- KV per token ≈ 2 × 28 × 4 × 128 × sizeof(f16) = 57,344 bytes
- Theoretical max: ~1M tokens (flash attention overhead ~2x)
- Safe practical: 80K-100K tokens (we use 32K as sensible default)

**TurboQuant KV** (`--cache-type-k turbo3`) is blocked by [llama.cpp GQA bug #78](https://github.com/TheTom/llama-cpp-turboquant/issues/78) — crashes when `n_head ≠ n_head_kv` (which Carnice-27B uses). Using `q8_0` KV instead.

## The Saga (What Didn't Work)

1. **vLLM source build** — 607 CUDA targets, Modal build runner times out after ~200
2. **vLLM TurboQuant** — only in git main, not v0.19.1 pip release, and the source build times out
3. **Official llama.cpp Docker images** — ENTRYPOINT conflicts with Modal's `add_python`
4. **`--tool-call-parser hermes`** — flag doesn't exist in ai-dock b8851 build (added in later llama.cpp releases). Turned out unnecessary — the chat template handles it.
5. **SGLang** — `qwen3_5_text` architecture not in transformers registry. Multiple patching approaches failed:
   - Patching `EntryClass` in `qwen3_5.py` to register `Qwen3_5ForCausalLM`
   - Patching `get_config` to override `model_type` on load
   - `sitecustomize.py` injection (didn't propagate to AutoConfig's lazy registry)

## Files

- `carnice_llama_modal.py` — **Main deployment** (working, production)
- `carnice_vllm_modal.py` — Abandoned vLLM attempt (source build timeout)
- `carnice_sglang_modal.py` — Abandoned SGLang attempt (architecture registry hack)
- `patch_sglang.py` — SGLang qwen3_5 registry patch (didn't work)
- `sitecustomize.py` — Transformers model_type override injection (didn't work)
- `monitor.sh` — Quick endpoint health check script
- `diag.py` — Container diagnostics helper

## License

This deployment code is MIT licensed. The Carnice-27b model itself follows its own license — see [kai-os/Carnice-27b-GGUF](https://huggingface.co/kai-os/Carnice-27b-GGUF).
