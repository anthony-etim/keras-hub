"""
Integration module for serving Keras Hub models through vLLM on TPU.

Serves any supported KerasHub `CausalLM` through tpu-inference's native flax/nnx
path (no torch, no torchax) via a one-line `KerasHubLLM("keras_hub:<preset>")`.
"""

from .nnx_adapter import KerasNNXModel
from .registry import KerasHubLLM
from .registry import setup_vllm_model
from .tokenizer import KerasVLLMTokenizerAdapter

__all__ = [
    "KerasHubLLM",
    "KerasNNXModel",
    "KerasVLLMTokenizerAdapter",
    "setup_vllm_model",
]
