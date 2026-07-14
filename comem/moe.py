"""CoMem on a ``device_map``-sharded MoE backbone (e.g. Tencent Hunyuan Hy3,
``hy_v3``, 80-layer MoE).

Why a separate class
--------------------
:class:`~comem.model.CoMem` assumes the whole backbone lives on ONE device (it
creates ids / RoPE positions on ``self.device`` and runs ``layers[...]`` + norm +
lm_head there). A 500GB+ MoE only fits when ``device_map="auto"`` shards its
decoder layers across several GPUs — so the depth-partitioned WRITE/READ loop must
hop the residual-stream hidden (and the mask + RoPE cos/sin) onto whichever GPU
the *next* layer sits on. :class:`CoMemMoE` is that thin, device-aware subclass: it
reuses ALL of the parent's packing / masking / read / decode semantics and ONLY
overrides the pieces that touch a device.

The MoE router is a pure function of the token hidden (position-blind), so a
chunk-local WRITE routes each token exactly as the full-context forward would and
the cached ``h_j`` is reproducible — this is the same claim the dense self-test
validates, and it holds across the sharded MoE.

Limitation: the resumed-band KV-cache decode (``read_prefill`` / ``decode_step``)
is not supported when the model is sharded across devices (the cache would span
GPUs); use ``generate*(use_kv_cache=False)`` (the recompute path) there. On a
single device (e.g. the CPU tiny self-test) everything works.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .model import CoMem


def _module_device(module: nn.Module) -> torch.device:
    """Device of a module's first parameter, falling back to its first buffer."""
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device("cpu")


def _to_device(obj, device: torch.device):
    """Move a tensor / tuple / list / dict of tensors (or ``None``) to ``device``."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


class _DeviceMovingCall:
    """Wrap a module so ``wrapper(x, ...)`` first moves ``x`` (and extra tensor
    args/kwargs) to the module's own device, then calls it. Used for ``norm`` /
    ``lm_head`` so the parent's read tail lands hidden on the right GPU."""

    def __init__(self, module: nn.Module):
        self.module = module
        self.device = _module_device(module)

    def __call__(self, x, *args, **kwargs):
        x = _to_device(x, self.device)
        args = tuple(_to_device(a, self.device) for a in args)
        kwargs = {k: _to_device(v, self.device) for k, v in kwargs.items()}
        return self.module(x, *args, **kwargs)


class CoMemMoE(CoMem):
    """Multi-GPU (``device_map``-sharded) CoMem over a stock MoE ``*ForCausalLM``.

    Drop-in for :class:`CoMem` — same ``write_chunk`` / ``read`` / ``read_core`` /
    ``resume_forward_ids`` / ``full_forward_logits`` / ``encode`` / ``generate``
    API and the same ``resume_j`` / ``top_prepay_b`` / ``block_diagonal``
    semantics. The only behavioural difference is that every layer / norm / lm_head
    call moves its inputs to the correct GPU first."""

    def __init__(
        self,
        model: nn.Module,
        resume_j: int,
        top_prepay_b: int = 0,
        block_diagonal: bool = False,
        tokenizer=None,
    ):
        super().__init__(model, resume_j, top_prepay_b, block_diagonal,
                         tokenizer=tokenizer)

        self.embed_device = _module_device(self.embed_tokens)
        self.device = self.embed_device
        self._layer_devices = [_module_device(layer) for layer in self.layers]
        self.norm = _DeviceMovingCall(self.norm)
        self.lm_head = _DeviceMovingCall(self.lm_head)
        self.hf_device_map = getattr(model, "hf_device_map", None)
        self.is_sharded = len({str(d) for d in self._layer_devices}) > 1

    def _run_layers(
        self,
        hidden: torch.Tensor,
        layer_slice: slice,
        causal_mask,
        positions: torch.Tensor,
        position_embeddings,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Run ``self.layers[layer_slice]`` moving ``(hidden, mask, position_ids,
        cos/sin)`` onto each layer's device before the call. On a single-device
        model every move is a no-op, so this is a strict superset of the parent.
        ``past_key_values`` / ``use_cache`` support the single-device recompute and
        KV-cache paths; the sharded KV-cache decode is not supported (see module
        docstring)."""
        use_ckpt = (
            self.grad_checkpoint
            and torch.is_grad_enabled()
            and hidden.requires_grad
        )
        start = 0 if layer_slice.start is None else layer_slice.start
        for offset, layer in enumerate(self.layers[layer_slice]):
            dev = self._layer_devices[start + offset]
            hidden = _to_device(hidden, dev)
            mask_d = _to_device(causal_mask, dev)
            pos_d = _to_device(positions, dev)
            pe_d = _to_device(position_embeddings, dev)
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    lambda h, _l=layer, _m=mask_d, _p=pos_d, _pe=pe_d: _l(
                        h,
                        attention_mask=_m,
                        position_ids=_p,
                        position_embeddings=_pe,
                        use_cache=False,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                out = layer(
                    hidden,
                    attention_mask=mask_d,
                    position_ids=pos_d,
                    position_embeddings=pe_d,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            hidden = self._layer_out_hidden(out)
        return hidden


def load_moe_comem(
    model_path: str,
    resume_j: int,
    top_prepay_b: int = 0,
    block_diagonal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
    max_memory: Optional[dict] = None,
    tokenizer=None,
) -> CoMemMoE:
    """Load a sharded MoE ``*ForCausalLM`` and wrap it in :class:`CoMemMoE`.

    Uses native ``transformers`` support with ``device_map="auto"`` to split the
    decoder layers across the visible GPUs (each layer stays whole on one device,
    required for the per-layer hop)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=True,
        max_memory=max_memory,
    ).eval()
    return CoMemMoE(
        model,
        resume_j=resume_j,
        top_prepay_b=top_prepay_b,
        block_diagonal=block_diagonal,
        tokenizer=tokenizer,
    )
