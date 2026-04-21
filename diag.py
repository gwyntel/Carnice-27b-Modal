import modal

app = modal.App("carnice-diag")

@app.function(image=modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime"))
def diag():
    import transformers
    print(f"transformers version: {transformers.__version__}")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    print(f"qwen3_5 in mapping: {'qwen3_5' in CONFIG_MAPPING_NAMES}")
    print(f"qwen3_5_text in mapping: {'qwen3_5_text' in CONFIG_MAPPING_NAMES}")
    for k in sorted(CONFIG_MAPPING_NAMES):
        if "qwen" in k.lower():
            print(f"  {k} -> {CONFIG_MAPPING_NAMES[k]}")

if __name__ == "__main__":
    with app.run():
        diag.remote()
