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


def _verify_tokenizer_dir(temp_dir: str) -> bool:
    """Loads the exported tokenizer back and checks it tokenizes to non-empty.

    Guards against transformers/tokenizers versions that build an empty
    tokenizer from the exported assets (which is worse than no tokenizer).
    """
    from transformers import AutoTokenizer

    rt = AutoTokenizer.from_pretrained(temp_dir)
    return len(rt("hello world").get("input_ids", [])) > 0


def _export_sentencepiece(tokenizer, temp_dir: str, proto: bytes) -> bool:
    """Exports a KerasHub SentencePiece tokenizer (Gemma/Llama/Mistral) for HF.

    Writes the raw SP proto as `tokenizer.model` plus a tokenizer_config naming
    the matching HF class, then verifies it round-trips. On failure, removes the
    files so the caller falls back to skip_tokenizer_init + pre-tokenization.
    """
    try:
        with open(os.path.join(temp_dir, "tokenizer.model"), "wb") as f:
            f.write(proto)
    except Exception as e:  # noqa: BLE001
        logging.warning("Could not write SentencePiece tokenizer.model: %s", e)
        return False

    cls_name = type(tokenizer).__name__.lower()
    hf_class = "GemmaTokenizer" if "gemma" in cls_name else "LlamaTokenizer"
    with open(
        os.path.join(temp_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "tokenizer_class": hf_class,
                "legacy": False,
                "add_bos_token": True,
                "add_eos_token": False,
            },
            f,
        )

    try:
        if _verify_tokenizer_dir(temp_dir):
            return True
        raise ValueError("exported SentencePiece tokenizer produced empty output")
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "SentencePiece tokenizer unusable (%s); use skip_tokenizer_init=True "
            "+ pre-tokenized input for this preset.",
            e,
        )
        for fn in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json"):
            p = os.path.join(temp_dir, fn)
            if os.path.exists(p):
                os.remove(p)
        return False


def _export_hf_tokenizer(tokenizer, temp_dir: str) -> bool:
    """Writes HF-compatible tokenizer files so vLLM can tokenize raw text.

    Handles KerasHub's byte-level BPE tokenizers (GPT-2, OPT, ... -> stock
    `GPT2Tokenizer`) and SentencePiece tokenizers (Gemma, Llama, Mistral ->
    `tokenizer.model` + Llama/Gemma tokenizer). Returns True on success; on
    failure the caller should use ``skip_tokenizer_init=True`` + pre-tokenized
    input.
    """
    # SentencePiece family: KerasHub stores the serialized proto on `.proto`.
    proto = getattr(tokenizer, "proto", None)
    if proto:
        return _export_sentencepiece(tokenizer, temp_dir, proto)

    # Byte-level BPE family.
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
        from transformers import AutoTokenizer, GPT2TokenizerFast

        fast = GPT2TokenizerFast(
            vocab_file=vocab_dst,
            merges_file=merges_path,
            unk_token=eos,
            bos_token=eos,
            eos_token=eos,
            add_prefix_space=False,
        )
        fast.save_pretrained(temp_dir)
        # Verify the emitted tokenizer.json actually round-trips (some
        # transformers/tokenizers versions build an empty byte-level BPE from
        # vocab+merges). If it tokenizes to nothing, it's worse than the slow
        # tokenizer — remove it and fall back.
        reloaded = AutoTokenizer.from_pretrained(temp_dir)
        if len(reloaded("hello world").get("input_ids", [])) > 0:
            return True
        raise ValueError("emitted fast tokenizer produced empty output")
    except Exception as e:  # noqa: BLE001 - fall back to the slow tokenizer
        logging.warning(
            "Fast tokenizer.json unusable (%s); using slow GPT2Tokenizer.", e
        )
        # Remove a possibly-broken tokenizer.json so vLLM loads the slow files
        # (vocab.json + merges.txt) instead of the bad fast tokenizer.
        broken = os.path.join(temp_dir, "tokenizer.json")
        if os.path.exists(broken):
            os.remove(broken)

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


def keras_hub_llm(preset: str, dtype: str = "bfloat16", **llm_kwargs):
    """One-line constructor: a `vllm.LLM` serving a KerasHub preset.

    Wraps the registration + `setup_vllm_model` + `LLM(...)` dance so callers
    can simply do::

        from keras_hub.src.vllm import keras_hub_llm
        llm = keras_hub_llm("gpt2_base_en")
        llm.generate(["The future of AI is"], sampling_params)

    `dtype` defaults to bfloat16 to match the TPU paged KV cache. Extra keyword
    args (e.g. `max_model_len`, `enforce_eager`, `gpu_memory_utilization`) are
    forwarded to `vllm.LLM`. Set `KERAS_BACKEND=jax` (and other env vars) before
    importing keras/vllm, as usual.

    Args:
        preset: KerasHub preset name (e.g. "gpt2_base_en").
        dtype: torch dtype string for inference.
        **llm_kwargs: forwarded to `vllm.LLM` (e.g. tokenizer override).

    Returns:
        A constructed `vllm.LLM` instance.
    """
    from vllm import LLM

    # KerasVLLMAdapter is a torch.nn.Module and must use tpu-inference's
    # torchax path, not the flax/nnx path. Set only if the caller hasn't.
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")

    register_keras_hub_models()
    model_dir = setup_vllm_model(preset, dtype=dtype)
    llm_kwargs.setdefault("tokenizer", model_dir)
    return LLM(model=model_dir, **llm_kwargs)
