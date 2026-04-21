"""sitecustomize.py — auto-loaded by Python before any other module.

Registers qwen3_5_text model type in transformers' AutoConfig registry
so that AutoConfig.from_pretrained can load Carnice-27b's config.json
without crashing with "Transformers does not recognize this architecture."

SGLang has its own Qwen3_5TextConfig at sglang.srt.configs.qwen3_5
which extends Qwen3NextConfig. We add the import path to transformers'
CONFIG_MAPPING_NAMES so AutoConfig can lazy-load it.
"""
import importlib


def _register_qwen35_text():
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return  # transformers not installed

    # SGLang provides Qwen3_5TextConfig — register it with transformers
    # so AutoConfig.from_pretrained can load qwen3_5_text model_type configs.
    # CONFIG_MAPPING_NAMES format: {"model_type": "module.ClassName"}
    if "qwen3_5_text" not in CONFIG_MAPPING_NAMES:
        CONFIG_MAPPING_NAMES["qwen3_5_text"] = "sglang.srt.configs.qwen3_5.Qwen3_5TextConfig"
    
    # Also register qwen3_5 (the VLM wrapper) in case it's needed
    if "qwen3_5" not in CONFIG_MAPPING_NAMES:
        CONFIG_MAPPING_NAMES["qwen3_5"] = "sglang.srt.configs.qwen3_5.Qwen3_5Config"

    # Register in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES for AutoModelForCausalLM
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        if "qwen3_5_text" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["qwen3_5_text"] = "sglang.srt.models.qwen3_5.Qwen3_5ForCausalLM"
    except (ImportError, AttributeError):
        pass


_register_qwen35_text()
