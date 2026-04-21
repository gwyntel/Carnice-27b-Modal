"""Patch SGLang to support Qwen3.5 text-only models.

Single patch:
- Add Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM to EntryClass
  (SGLang only registers VLM variants by default)

Config model_type issue is handled at runtime by pre-patching
config.json in the HF cache volume (see startup() in deploy script).
"""
import glob

# --- EntryClass patch ---
qwen35_path = glob.glob("/sgl-workspace/**/sglang/srt/models/qwen3_5.py", recursive=True)
if not qwen35_path:
    qwen35_path = glob.glob("/**/sglang/srt/models/qwen3_5.py", recursive=True)
if not qwen35_path:
    raise FileNotFoundError("Cannot find qwen3_5.py")

qwen35_path = qwen35_path[0]
print(f"Patching {qwen35_path}")

with open(qwen35_path) as f:
    content = f.read()

old = "EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]"
new = "EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForCausalLM, Qwen3_5ForCausalLM]"
content = content.replace(old, new)
print(f"Patched EntryClass: added Qwen3_5ForCausalLM + Qwen3_5MoeForCausalLM")

with open(qwen35_path, "w") as f:
    f.write(content)

print("EntryClass patch applied successfully!")
