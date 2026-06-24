"""
Registry module for Keras Hub to vLLM Integration.

Hooks into vLLM's model loading mechanism so that `LLM(model="keras_hub:preset_name")`
is recognized and routed to the `KerasVLLMAdapter`.
"""

import json
import logging
import os
import shutil
import tempfile

from keras_hub.src.vllm.adapter import KerasVLLMAdapter


def _export_hf_tokenizer(tokenizer, temp_dir: str) -> bool:
    """Writes HF-compatible tokenizer files so vLLM can tokenize raw text.

    KerasHub's byte-level BPE tokenizers (GPT-2, OPT, ...) share GPT-2's
    vocab/merges format, so we export them and let vLLM load a stock
    `GPT2Tokenizer`. Returns True on success. SentencePiece presets (Gemma,
    Llama, Mistral) are not handled yet — callers should fall back to
    ``skip_tokenizer_init=True`` + pre-tokenization for those.
    """
    if not (hasattr(tokenizer, "merges") and hasattr(tokenizer, "save_assets")):
        return False
    try:
        # Writes vocabulary.json + merges.txt into temp_dir.
        tokenizer.save_assets(temp_dir)
    except Exception as e:  # noqa: BLE001
        logging.warning("Tokenizer save_assets failed: %s", e)
        return False

    vocab_src = os.path.join(temp_dir, "vocabulary.json")
    vocab_dst = os.path.join(temp_dir, "vocab.json")  # HF expects this name
    if os.path.exists(vocab_src) and not os.path.exists(vocab_dst):
        shutil.copyfile(vocab_src, vocab_dst)
    merges_path = os.path.join(temp_dir, "merges.txt")
    if not (os.path.exists(vocab_dst) and os.path.exists(merges_path)):
        return False

    # HF's GPT2Tokenizer drops the first line of merges.txt (expects a version
    # header). KerasHub writes no header, so prepend one or the first real merge
    # rule would be silently lost.
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_content = f.read()
    if not merges_content.startswith("#version"):
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n" + merges_content)

    eos = "<|endoftext|>"

    # Prefer emitting a fast tokenizer (tokenizer.json), which vLLM v1's
    # incremental detokenizer favors. Building GPT2TokenizerFast from the
    # vocab/merges and calling save_pretrained writes tokenizer.json plus
    # consistent tokenizer_config.json / special_tokens_map.json.
    try:
        from transformers import GPT2TokenizerFast

        fast = GPT2TokenizerFast(
            vocab_file=vocab_dst,
            merges_file=merges_path,
            unk_token=eos,
            bos_token=eos,
            eos_token=eos,
            add_prefix_space=False,
        )
        fast.save_pretrained(temp_dir)
        return True
    except Exception as e:  # noqa: BLE001 - fall back to the slow tokenizer
        logging.warning(
            "Could not emit fast tokenizer.json (%s); writing slow GPT2Tokenizer "
            "config instead.",
            e,
        )

    # Slow-tokenizer fallback: minimal config over vocab.json + merges.txt.
    with open(
        os.path.join(temp_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "tokenizer_class": "GPT2Tokenizer",
                "bos_token": eos,
                "eos_token": eos,
                "unk_token": eos,
                "add_prefix_space": False,
                "clean_up_tokenization_spaces": False,
            },
            f,
        )
    with open(
        os.path.join(temp_dir, "special_tokens_map.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"bos_token": eos, "eos_token": eos, "unk_token": eos}, f)
    return True


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
    eos_token_id = None
    try:
        from keras_hub import models

        tokenizer = models.Tokenizer.from_preset(preset)
        vocab_size = int(tokenizer.vocabulary_size())
        eos_token_id = getattr(tokenizer, "end_token_id", None)
        # Export HF-format tokenizer files so vLLM can accept raw-text prompts.
        # If unsupported (e.g. SentencePiece), pass skip_tokenizer_init=True to
        # LLM(...) and feed token IDs tokenized with the KerasHub tokenizer.
        if _export_hf_tokenizer(tokenizer, temp_dir):
            logging.info("Exported HF tokenizer assets for preset %r.", preset)
        else:
            logging.warning(
                "Could not export an HF tokenizer for preset %r; use "
                "skip_tokenizer_init=True + pre-tokenized input.",
                preset,
            )
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
    if eos_token_id is not None:
        config_dict["eos_token_id"] = int(eos_token_id)
        config_dict["bos_token_id"] = int(
            getattr(tokenizer, "start_token_id", None) or eos_token_id
        )

    with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    return temp_dir
