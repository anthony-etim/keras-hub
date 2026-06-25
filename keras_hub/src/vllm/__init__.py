"""
Integration module for vLLM and Keras Hub.

This module exposes the necessary adapters and hooks to allow vLLM to use Keras Hub
models natively as the backend for LLM generation.
"""

from .adapter import KerasVLLMAdapter
from .registry import keras_hub_llm
from .registry import register_keras_hub_models
from .registry import setup_vllm_model
from .tokenizer import KerasVLLMTokenizerAdapter

__all__ = [
    "KerasVLLMAdapter",
    "KerasVLLMTokenizerAdapter",
    "keras_hub_llm",
    "register_keras_hub_models",
    "setup_vllm_model",
]
