# 🧠 Carnice-27b on Modal

**Serverless GPU inference for [Carnice-27b](https://huggingface.co/kai-os/Carnice-27b-GGUF)** — a Qwen3.5-27B fine-tune with native hermes-style tool-calling.

Deployed on [Modal](https://modal.com) with llama.cpp on A100-80GB GPUs. OpenAI-compatible API. Scales to zero. **252K tokens of context tested.**

## Why This Exists

Carnice-27b uses Qwen3.5's `qwen3_5_text` architecture, which **breaks** in standard Python inference frameworks:

- **vLLM**: Source build times out (607 CUDA compile targets). No native `qwen35` support.
- **SGLang**: `qwen3_5_text` not in transformers registry. Patching `EntryClass` and `get_config` didn't stick.
- **llama.cpp**: ✅ Natively supports `qwen35` architecture (PR #19468). Pre-built binaries. GGUF quantization. Just works.

**The pivot from SGLang/vLLM to llama.cpp saved this project.** No more Python venv hell, no registry hacks, no source builds.

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────┐
│  Your App    │────▶│  Modal Serverless Endpoint                    │
│              │     │  (A100-80GB, $2.50/hr)                        │
│  OpenAI SDK  │◀────│                                               │
│  llama.cpp server (ai-dock b8851 build)       │
│  Q4_K_M GGUF (16.5GB weights)                │
│  ~60GB VRAM available for KV cache             │
│  256K context (262144 trained, 252K tested ✅)  │
                     │  Flash attention, q8_0 KV cache               │
                     │  Checkpoint caching (82x faster on cache hit) │
                     │  7-minute scaledown window                    │
                     └──────────────────────────────────────────────┘
```

## Performance

### Throughput by Context Size

| Prompt Tokens | Prompt Eval (tok/s) | Generation (tok/s) | Prompt Time |
|---------------|---------------------|--------------------|-------------|
| 8K | 1,244 | 44.2 | <1s |
| 24K | 1,219 | 40.0 | 19.6s |
| 64K | 1,028 | 34.3 | 21.3s |
| 85K | 943 | 31.5 | 23.4s |
| 128K | 1,005 | 30.1 | ~100s cold / **0.1s cached** |
| 252K | 761 | 21.0 | 332s cold |

### Checkpoint Caching (the big deal)

| Scenario | Prompt Tokens | Time | Speedup |
|----------|---------------|------|---------|
| Cold (no cache) | 128K | 164s | — |
| Exact prompt match (cache hit) | 128K | **2s** | **82x faster** |
| Cache hit rate | 128K | **100%** on exact match | — |

Qwen3.5's hybrid SSM/attention architecture creates **32 checkpoints** during prompt processing. On exact prompt match, all 128K+ tokens are served from cache in 0.1 seconds. This is transformative for agent workflows where the system prompt + conversation history is reused.

**Caveat:** The current build (ai-dock b8851) doesn't support `--ctx-checkpoints` / `--checkpoint-every-n-tokens` (added in llama.cpp ~Mar 2026). Without these flags, **changing any part of the prompt forces full reprocessing** — the hybrid architecture invalidates the entire checkpoint when the prefix changes. With those flags, partial cache hits would work for typical agent patterns (same system prompt, changing user message). A newer llama.cpp build would unlock this.

### Key Specs

| Metric | Value |
|--------|-------|
| **Model** | Carnice-27b (Qwen3.5-27B fine-tune) |
| **Quantization** | Q4_K_M (16.5GB) |
| **GPU** | A100-80GB ($2.50/hr) |
| **Context (default)** | 131,072 tokens (128K), 2 parallel slots of 64K |
| **Context (model trained)** | 262,144 tokens (256K) |
| **Context (tested max)** | 252,352 tokens — no OOM on A100 |
| **KV cache type** | q8_0 (half the size of f16) |
| **Cold start** | ~2-3 min (first), ~30s (GGUF cached) |
| **Cost** | $2.50/hr, scales to zero after 7 min idle |
| **Tool-calling** | ✅ Native via chat template (no parser flag needed) |

## KV Cache Deep Dive

### The Math

Qwen3.5-27B architecture: 28 layers, 4 KV heads (GQA), head_dim=128.

| KV Type | Per Token | Theoretical Max (parallel=1) | Theoretical Max (parallel=2) |
|---------|-----------|------------------------------|------------------------------|
| f16 | 56 KB | ~1.1M tokens | ~552K per slot |
| **q8_0** | **28 KB** | **~2.2M tokens** | **~1.1M per slot** |

Available KV VRAM: 80GB - 16.5GB (weights) - 1.5GB (CUDA) - 3GB (activation) ≈ **59GB**

**We're using <10% of the A100's KV capacity at 128K context.** The bottleneck isn't memory — it's prompt processing latency. At 761 tok/s, filling 252K tokens takes ~5.5 minutes for the initial prompt.

### TurboQuant KV (blocked)

`--cache-type-k turbo3` would compress KV cache further, but is blocked by [llama.cpp GQA bug #78](https://github.com/TheTom/llama-cpp-turboquant/issues/78) — crashes when `n_head ≠ n_head_kv` (which Carnice-27B uses via GQA). Watch for a fix.

### Tiered KV Caching

llama.cpp supports multiple KV cache tiers:

- **`--cache-type-k q8_0`** (current) — quantized KV in VRAM, half the size of f16
- **`--cache-ram`** — offload prompt checkpoints to host RAM. Saves VRAM for even more context. Makes sense on systems where VRAM is scarce but RAM is plentiful.
- **`--ctx-checkpoints N`** — create checkpoints every N token batches. Critical for Qwen3.5 hybrid architecture: without it, any context truncation forces full reprocessing (37s → 0.9s, 39x speedup per [PR #19970](https://github.com/ggml-org/llama.cpp/pull/19970))
- **`--cache-disk`** — (feature request, [issue #20697](https://github.com/ggml-org/llama.cpp/issues/20697)) checkpoint to NVMe SSD. Would unlock near-unlimited context on UMA systems where RAM=VRAM.
- **`--slot-save-path`** — persist slot state to disk. Survives server restarts but doesn't help with runtime cache sharing.

**Our current limitation:** The ai-dock b8851 build doesn't have `--ctx-checkpoints` or `--checkpoint-every-n-tokens`. A newer llama.cpp build would enable partial cache hits for agent workflows (same system prompt, new user message = reprocess only the changed suffix).

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

### Context Length & Parallelism

The context window is split evenly across parallel slots. Configure based on your use case:

| Config | Context | Slots | Slot Size | Best For |
|--------|---------|-------|-----------|----------|
| `--ctx-size 262144 --parallel 1` | 256K | 1 | 256K | **Default** — max capacity, tested up to 252K tokens |
| `--ctx-size 131072 --parallel 2` | 128K | 2 | 64K | 2 concurrent requests, good balance |
| `--ctx-size 131072 --parallel 1` | 128K | 1 | 128K | Single long-context request (agent loops) |

Configured in `carnice_llama_modal.py`:
```python
CONTEXT_LENGTH = 262144  # 256K = full model training length, single slot
```

### Quantization Options

| Format | Size | PPL | KLD 99.9% | Recommendation |
|--------|------|-----|-----------|----------------|
| **Q4_K_M** | 16.5GB | 6.6053 | 0.5478 | ✅ Sweet spot — max KV headroom |
| Q5_K_M | ~20GB | ~6.58 | ~0.24 | If you need quality bump and have VRAM |
| Q6_K | ~23GB | 6.5456 | ~0.22 | ❌ Uncanny valley — 7GB more for <1% PPL improvement over Q4 |
| Q8_0 | ~29GB | 6.5352 | ~0.10 | Only for benchmarks |

Edit `GGUF_FILE` to switch:
```python
GGUF_FILE = "Carnice-27b-Q4_K_M.gguf"  # Change this
```

### Key Parameters

```python
CONTEXT_LENGTH = 262144   # Full model context window (256K trained, 252K tested)
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

### Modal Container Lifecycle Gotcha

**`modal deploy` does NOT restart running containers.** The deployment spec updates but the warm container keeps its old `--ctx-size` and `--parallel` flags. To actually change server parameters:

1. `modal app stop <app-id>` — kill the running container
2. Wait ~30 seconds for the container to fully shut down
3. `modal deploy carnice_llama_modal.py` — new container starts with updated config
4. Wait for cold start (GGUF cached: ~2 min, fresh: ~5 min)

Check actual running config via logs:
```bash
modal app logs <app-id> | grep "n_ctx"
```

Look for `n_ctx_slot` (not just `n_ctx`) — `n_ctx_slot = n_ctx / parallel`.

## The Saga (What Didn't Work)

1. **vLLM source build** — 607 CUDA targets, Modal build runner times out after ~200
2. **vLLM TurboQuant** — only in git main, not v0.19.1 pip release, and the source build times out
3. **Official llama.cpp Docker images** — ENTRYPOINT conflicts with Modal's `add_python`
4. **`--tool-call-parser hermes`** — flag doesn't exist in ai-dock b8851 build (added in later llama.cpp releases). Turned out unnecessary — the chat template handles it.
5. **SGLang** — `qwen3_5_text` architecture not in transformers registry. Multiple patching approaches failed:
   - Patching `EntryClass` in `qwen3_5.py` to register `Qwen3_5ForCausalLM`
   - Patching `get_config` to override `model_type` on load
   - `sitecustomize.py` injection (didn't propagate to AutoConfig's lazy registry)
6. **Modal container reuse** — `modal deploy` updates the spec but warm containers keep old `--ctx-size`. Spent way too long debugging why "131072 ctx" was still serving 32768-slot requests before realizing the container wasn't restarted.
7. **TurboQuant KV** — would compress KV cache further but [GQA bug #78](https://github.com/TheTom/llama-cpp-turboquant/issues/78) crashes on models where `n_head ≠ n_head_kv` (i.e. every GQA model including Carnice)

## Next Steps

- [ ] **Newer llama.cpp build** — unlock `--ctx-checkpoints` and `--checkpoint-every-n-tokens` for partial cache hits on hybrid Qwen3.5 architecture. This would make agent workflows (same system prompt, new user message) near-instant instead of forcing full reprocessing.
- [ ] **`--cache-ram`** — offload checkpoints to host RAM, freeing VRAM for even more context or parallelism
- [ ] **Hermes-agent integration** — add Carnice as a provider in hermes-agent config
- [ ] **TurboQuant KV** — watch for GQA bug fix, then swap `--cache-type-k q8_0` → `turbo3` for even more context headroom
- [ ] **`--cache-disk`** — when it lands in llama.cpp, persist checkpoints across server restarts

## Files

### Production

- `carnice_llama_modal.py` — **Main deployment** (working, 256K context)
- `monitor.sh` — Quick endpoint health check script

### Attempts (non-working)

- `attempts/carnice_vllm_modal.py` — vLLM + TurboQuant attempt (source build timeout — 607 CUDA targets, Modal build runner dies after ~200)
- `attempts/carnice_sglang_modal.py` — SGLang attempt (`qwen3_5_text` not in transformers registry, multiple patches failed)
- `attempts/patch_sglang.py` — SGLang EntryClass registry patch (added `Qwen3_5ForCausalLM` — didn't resolve config loading)
- `attempts/sitecustomize.py` — Transformers `CONFIG_MAPPING_NAMES` override (didn't propagate to AutoConfig's lazy registry)
- `attempts/diag.py` — Container diagnostics helper (used during SGLang debugging)

## License

This deployment code is MIT licensed. The Carnice-27b model itself follows its own license — see [kai-os/Carnice-27b-GGUF](https://huggingface.co/kai-os/Carnice-27b-GGUF).
