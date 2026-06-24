"""
Registry module for Keras Hub to vLLM Integration.

Hooks into vLLM's model loading mechanism so that `LLM(model="keras_hub:preset_name")`
is recognized and routed to the `KerasVLLMAdapter`.
"""

import json
import logging
import os
import tempfile

from keras_hub.src.vllm.adapter import KerasVLLMAdapter


def _register_model_architecture() -> None:
    """Registers KerasVLLMAdapter with vLLM's internal model registry."""
    try:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.register_model("KerasVLLMAdapter", KerasVLLMAdapter)
    except ImportError:
        logging.warning(
            "Skipping KerasVLLMAdapter registration. vLLM is not installed "
            "or the ModelRegistry module could not be imported."
        )


def _patch_tpu_model_loader() -> None:
    """Patches vLLM-TPU to treat KerasVLLMAdapter as a JAX-native architecture.

    If we don't do this, older versions of vLLM will fall back to tracing the PyTorch
    forward pass, which will cause JAX to capture Keras variables as large static constants.
    """
    try:
        from vllm.model_executor import model_loader

        if hasattr(model_loader, "JAX_NATIVE_ARCHITECTURES"):
            if "KerasVLLMAdapter" not in model_loader.JAX_NATIVE_ARCHITECTURES:
                model_loader.JAX_NATIVE_ARCHITECTURES = list(
                    model_loader.JAX_NATIVE_ARCHITECTURES
                ) + ["KerasVLLMAdapter"]
        if hasattr(model_loader, "_JAX_NATIVE_ARCHITECTURES"):
            if "KerasVLLMAdapter" not in model_loader._JAX_NATIVE_ARCHITECTURES:
                model_loader._JAX_NATIVE_ARCHITECTURES = list(
                    model_loader._JAX_NATIVE_ARCHITECTURES
                ) + ["KerasVLLMAdapter"]
    except (ImportError, AttributeError) as e:
        logging.debug("vLLM TPU model_loader patch skipped: %s", e)


def register_keras_hub_models() -> None:
    """Registers the keras_hub schema with vLLM's internal ModelRegistry.

    When `setup_vllm_model` is used to create a model directory, vLLM will
    natively load the `KerasVLLMAdapter`.
    """
    _register_model_architecture()
    _patch_tpu_model_loader()


def setup_vllm_model(preset: str, dtype: str = "float16") -> str:
    """Creates a configuration directory for vLLM to load a Keras Hub preset.

    Args:
        preset: The Keras Hub preset name (e.g., "gemma_2b_en").
        dtype: The torch dtype to run inference with.

    Returns:
        The path to the temporary configuration directory to pass to `vllm.LLM`.
    """
    temp_dir = tempfile.mkdtemp(prefix="keras_hub_vllm_")

    preset_lower = preset.lower()
    if "opt" in preset_lower:
        model_type = "opt"
    elif "gpt2" in preset_lower or "gpt_2" in preset_lower:
        model_type = "gpt2"
    elif "gemma" in preset_lower:
        model_type = "gemma2"
    else:
        model_type = "gemma2"

    # Derive the true vocabulary size from the preset's tokenizer. vLLM relies
    # on vocab_size for KV-cache memory profiling, so a hardcoded value
    # silently mis-profiles any model other than the one it was tuned for
    # (e.g. GPT-2's 50257 vs. Gemma's 256000).
    vocab_size = None
    try:
        from keras_hub import models

        tokenizer = models.Tokenizer.from_preset(preset)
        vocab_size = int(tokenizer.vocabulary_size())
    except Exception as e:  # noqa: BLE001 - fall back to a sane default
        logging.warning(
            "Could not infer vocab_size for preset %r (%s); falling back to a "
            "default. Set it explicitly if memory profiling looks wrong.",
            preset,
            e,
        )
        vocab_size = 50272 if "opt" in preset_lower else 256000

    config_dict = {
        "architectures": ["KerasVLLMAdapter"],
        "_name_or_path": f"keras_hub:{preset}",
        "keras_hub_preset": preset,
        "torch_dtype": dtype,
        "model_type": model_type,
        "vocab_size": vocab_size,
    }

    with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    return temp_dir
