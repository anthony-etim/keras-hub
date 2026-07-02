"""JAX-native (flax/nnx) model for serving KerasHub CausalLMs on vLLM-TPU.

Serves a KerasHub model through tpu-inference's native ``flax_nnx`` path — no
torch, no torchax. It reuses the existing KerasHub backbone:

- ``__init__`` builds the structure (``load_weights=False``, so it survives the
  loader's ``nnx.eval_shape`` abstract construction),
- ``load_weights`` fills the real preset weights,
- ``__call__`` re-implements the backbone forward to thread vLLM's per-layer
  paged KV cache into each attention layer and run the RPA kernel (via the
  serving context, into which the paged-attention function is injected).

Validated end-to-end on ``gpt2_base_en`` (coherent output, next-token-exact vs
native KerasHub). Remaining: shard weights as ``nnx.Param`` for TP>1; validate
more model families.
"""

import inspect

from keras import ops

from keras_hub import models
from keras_hub.src.vllm.context import clear_vllm_context
from keras_hub.src.vllm.context import set_vllm_context

try:
    from flax import nnx

    _NNX_AVAILABLE = True
except ImportError:  # pragma: no cover - flax not installed
    nnx = None
    _NNX_AVAILABLE = False


if _NNX_AVAILABLE:

    class KerasNNXModel(nnx.Module):
        """A KerasHub `CausalLM` served as a native flax/nnx model on vLLM-TPU.

        tpu-inference's loader routes any config carrying `keras_hub_preset` to
        this class, so it is served on the native `flax_nnx` path with no
        torch/torchax involvement.
        """

        # Tell the loader not to build this model under `nnx.eval_shape` / an
        # outer jit; construction and weight loading are handled here.
        _self_manages_sharding = True

        def __init__(self, vllm_config, rng_key, mesh=None):
            """Builds the model structure (weights come from `load_weights`).

            Args:
                vllm_config: The vLLM config; `model_config.hf_config` carries
                    the `keras_hub_preset` written by `setup_vllm_model`.
                rng_key: JAX PRNG key (unused; weights come from the preset).
                mesh: JAX device mesh used by the paged-attention kernel.
            """
            self.vllm_config = vllm_config
            self.mesh = mesh
            self.preset_name = (
                vllm_config.model_config.hf_config.keras_hub_preset
            )
            self._layer_params = None
            self._final_norm = None
            self._is_gemma = "gemma" in (self.preset_name or "").lower()

            # Build structure only. `load_weights=False` skips from_preset's
            # `jax_memory_cleanup` (which calls `.delete()` and fails under the
            # loader's `nnx.eval_shape` trace); real weights arrive in
            # `load_weights`.
            self.model = models.CausalLM.from_preset(
                self.preset_name, dtype="bfloat16", load_weights=False
            )
            self.backbone = self.model.backbone

        def __call__(
            self,
            kv_caches,
            input_ids,
            attention_metadata,
            *args,
            **kwargs,
        ):
            """Runs one forward step with vLLM's paged KV cache.

            Re-implements the backbone forward (embeddings -> transformer layers
            -> final norm) so the per-layer paged cache is threaded into each
            attention layer, which dispatches to the RPA kernel via the serving
            context.

            Returns:
                `(updated_kv_caches, hidden_states, None, None)` — the tuple the
                native runner expects.
            """
            positions = getattr(attention_metadata, "input_positions", None)
            if positions is None:
                positions = kwargs.get("positions")

            token_ids = input_ids
            if len(token_ids.shape) == 1:
                token_ids = ops.expand_dims(token_ids, axis=-1)

            x = self._embed(token_ids, positions)
            self._set_serving_context(attention_metadata, positions)
            try:
                hidden_states, updated_kv_caches = self._run_layers(
                    x, kv_caches, attention_metadata, positions
                )
            finally:
                clear_vllm_context()

            return updated_kv_caches, hidden_states, None, None

        def compute_logits(self, hidden_states, *args, **kwargs):
            """Projects hidden states to vocab logits via the tied embedding."""
            return self.backbone.token_embedding(hidden_states, reverse=True)

        def load_weights(self, *args, **kwargs):
            """Fills the structure built in `__init__` with real preset weights.

            The native loader calls this after `nnx.eval_shape`, on the concrete
            model, so we load a real `from_preset` and copy its weights in.
            """
            full = models.CausalLM.from_preset(
                self.preset_name, dtype="bfloat16"
            )
            for dst, src in zip(self.model.weights, full.weights):
                dst.assign(src.value if hasattr(src, "value") else src)
            del full

        # ------------------------------------------------------------------ #
        # Helpers
        # ------------------------------------------------------------------ #
        def _embed(self, token_ids, positions):
            """Token embedding + (for non-RoPE models) learned positions."""
            backbone = self.backbone
            x = backbone.token_embedding(token_ids)

            if self._is_gemma:
                x = x * ops.sqrt(ops.cast(backbone.hidden_dim, x.dtype))

            # Learned position embeddings (GPT-2/OPT); RoPE models skip this and
            # apply rotary positions inside attention.
            position_embedding = getattr(backbone, "position_embedding", None)
            if position_embedding is not None and positions is not None:
                pos_ids = ops.cast(ops.reshape(positions, (-1,)), "int32")
                pos_emb = ops.take(
                    position_embedding.position_embeddings, pos_ids, axis=0
                )
                x = x + ops.cast(ops.reshape(pos_emb, ops.shape(x)), x.dtype)
            return x

        def _set_serving_context(self, attention_metadata, positions):
            """Publishes the paged-attention context for the attention hooks.

            The native path does not inject a paged-attention function, so we
            supply tpu-inference's `_jax_attn_func`. Its signature matches what
            `vllm_paged_attention` calls, so the guarded attention hooks
            run the RPA kernel instead of falling back to dense attention.
            """
            block_tables = getattr(attention_metadata, "block_tables", None)
            slot_mapping = getattr(
                attention_metadata,
                "slot_mapping_tensor",
                getattr(attention_metadata, "slot_mapping", None),
            )
            paged_attn_func = getattr(
                attention_metadata, "paged_attention_func", None
            )
            if paged_attn_func is None:
                try:
                    from tpu_inference.layers.vllm.backends.flash_attn import (
                        _jax_attn_func,
                    )

                    paged_attn_func = _jax_attn_func
                except ImportError:
                    paged_attn_func = None
            set_vllm_context(
                block_tables,
                slot_mapping,
                attention_metadata,
                paged_attn_func,
                self.mesh,
                positions=positions,
            )

        def _run_layers(self, x, kv_caches, attention_metadata, positions):
            """Runs the transformer layers, threading the paged cache per layer.

            A plain `backbone(inputs)` leaves `cache=None`, which the RPA kernel
            can't use, so we call each layer with its vLLM paged cache. Returns
            `(hidden_states, updated_kv_caches)`.
            """
            layers = self.backbone.transformer_layers
            if self._layer_params is None:
                self._layer_params = (
                    set(inspect.signature(layers[0].call).parameters)
                    if len(layers)
                    else set()
                )
            params = self._layer_params

            # Only build kwargs the layer accepts — e.g. skip computing a
            # padding_mask the paged-attention path ignores.
            kwargs_base = {}
            if "positions" in params and positions is not None:
                kwargs_base["positions"] = positions
            if "seq_lens" in params:
                seq_lens = getattr(attention_metadata, "seq_lens", None)
                if seq_lens is not None:
                    kwargs_base["seq_lens"] = seq_lens
            if "padding_mask" in params:
                kwargs_base["padding_mask"] = ops.ones(
                    ops.shape(x)[:-1], dtype="bool"
                )

            updated = list(kv_caches) if kv_caches is not None else None
            for i, layer in enumerate(layers):
                call_kwargs = dict(kwargs_base)
                cache = (
                    updated[i]
                    if updated is not None and len(updated) > i
                    else None
                )
                if "self_attention_cache" in params:
                    call_kwargs["self_attention_cache"] = cache
                elif "cache" in params:
                    call_kwargs["cache"] = cache
                if "kv_cache" in params:
                    call_kwargs["kv_cache"] = cache

                out = layer(x, **call_kwargs)
                if isinstance(out, tuple):
                    x = out[0]
                    if len(out) > 1 and updated is not None:
                        updated[i] = out[1]
                else:
                    x = out

            if self._final_norm is None:
                for name in ("layer_norm", "final_layer_norm", "norm"):
                    candidate = getattr(self.backbone, name, None)
                    if candidate is not None:
                        self._final_norm = candidate
                        break
            hidden_states = self._final_norm(x)
            if len(hidden_states.shape) == 3 and hidden_states.shape[1] == 1:
                hidden_states = ops.squeeze(hidden_states, axis=1)
            return hidden_states, updated

else:

    class KerasNNXModel:  # pragma: no cover - flax/nnx not available
        """Placeholder raised on use when flax (nnx) is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "KerasNNXModel requires flax (nnx) and a JAX/vLLM-TPU "
                "environment."
            )
