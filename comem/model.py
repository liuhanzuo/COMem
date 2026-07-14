"""CoMem — Comprehension Memory: a self-contained mid-depth-resume long-context
memory over a plain (un-patched) decoder LLM.

The insight
-----------
A transformer LM completes most of its *comprehension* of a token span in its
lower layers; the upper layers are increasingly about producing the next-token
distribution. CoMem exploits this by splitting the backbone at an arbitrary depth
``j`` (``resume_j``):

  * WRITE  (per chunk, chunk-local): ``embed -> layers[0:j]`` with a chunk-local
    causal mask + RoPE positions ``0:T`` -> cache the depth-``j`` hidden ``h_j``.
    Each chunk is comprehended in isolation and its mid-layer hidden is stored.
  * READ   (per query): retrieve the ``topk`` most relevant cached chunks, pack
    ``[sink ; h_j^{c1} ; ... ; h_j^{ck} ; h_j^{query}]`` into ONE sequence with
    FRESH contiguous RoPE positions and a causal mask, then RESUME ``layers[j:]
    -> norm -> lm_head``. Only the upper layers are recomputed, over a FIXED-size
    read pack (``topk`` chunks) — so read cost is constant in the context length.

``j = 0`` degenerates to a selective full re-forward of the retrieved chunks (the
RAG upper bound); ``j = L`` is the closed-book endpoint. The load-bearing
correctness claim — resume-from-layer-``j`` over the packed sequence reproduces a
stock full forward at ``j = 0`` to floating-point tolerance — is the self-test
gate (``python -m comem.selftest``; see the package tests).

Headline results (Qwen3-8B; see the paper): full-context attention collapses to 0
past its RoPE window while CoMem holds RULER ~100 / LongEval ~0.98 at 128k, and
the resumed-band KV-cache decode runs 4-16x faster per token than re-running the
whole read every step.

This module is self-contained: it reads ``embed_tokens / layers / norm /
rotary_emb`` + ``lm_head`` off a stock ``*ForCausalLM`` (Llama, Qwen3, MoE
variants) and NEVER mutates the backbone.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import DynamicCache

from . import selectors as _sel


# baseline arms are expressed as a re-parameterisation of the CoMem primitives.
#   comem    : retrieval (selector + topk) + the given resume_j.
#   kvdirect : full-depth recompute (build with resume_j=0) + NO retrieval.
#   hcache   : mid-layer recompute (given resume_j) + NO retrieval.
# ``mode`` in ``generate`` toggles retrieval; the resume depth is a construction
# choice (build CoMem(resume_j=0) for a faithful kvdirect).
_MODE_NO_RETRIEVAL = {"comem": False, "kvdirect": True, "hcache": True}


class CoMem:
    """Mid-depth-resume comprehension memory over a stock ``*ForCausalLM``.

    Parameters
    ----------
    model:
        A loaded causal LM exposing ``.model.{embed_tokens,layers,norm,rotary_emb}``
        and ``.lm_head`` (Llama-3, Qwen3, and in-tree MoE backbones all qualify).
        Used read-only; never patched.
    resume_j:
        Bottom prepay depth (layer split index). ``[0, num_hidden_layers]``.
        ``0`` = RAG upper bound (selective full re-forward); ``L`` = closed-book.
    top_prepay_b:
        Number of TOP layers to run query-local at read (approximate top-prepay,
        ``0`` = exact connective resume). ``[0, L - resume_j]``.
    block_diagonal:
        Ablation: replace the read's causal mask with a block-diagonal one
        (sink global; each chunk within-block only, query-blind; query reads
        sink + all chunks + itself). Only defined for ``top_prepay_b == 0``.
    tokenizer:
        Optional HF tokenizer, enabling the text-level ``encode`` / ``generate``
        API and BOS/EOS resolution. Not required for the id-level primitives.
    """

    def __init__(
        self,
        model: nn.Module,
        resume_j: int,
        top_prepay_b: int = 0,
        block_diagonal: bool = False,
        tokenizer=None,
    ):
        self.model = model
        inner = getattr(model, "model", model)
        self.inner = inner
        self.embed_tokens = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.rotary_emb = inner.rotary_emb
        self.lm_head = model.lm_head
        self.config = inner.config
        self.num_layers = int(self.config.num_hidden_layers)
        self.tokenizer = tokenizer

        if not (0 <= int(resume_j) <= self.num_layers):
            raise ValueError(
                f"resume_j must be in [0, {self.num_layers}]; got {resume_j}"
            )
        self.resume_j = int(resume_j)

        if not (0 <= int(top_prepay_b) <= self.num_layers - self.resume_j):
            raise ValueError(
                f"top_prepay_b must be in [0, {self.num_layers - self.resume_j}] "
                f"(num_layers - resume_j); got {top_prepay_b}"
            )
        self.top_prepay_b = int(top_prepay_b)
        self.mid_end = self.num_layers - self.top_prepay_b  # == L-b

        self.block_diagonal = bool(block_diagonal)
        # Optional gradient checkpointing on the (grad-bearing) read layer loop.
        self.grad_checkpoint = False

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.hidden_size = int(self.config.hidden_size)

        # populated by ``encode`` (the cached comprehension memory)
        self._ctx_chunks: List[torch.Tensor] = []
        self._ctx_hj: List[torch.Tensor] = []
        self._sink_hj: Optional[torch.Tensor] = None
        self._chunk_size: int = 512
        self._sink_tokens: str = "bos"

    # ------------------------------------------------------------------ #
    # low-level helpers
    # ------------------------------------------------------------------ #
    def _as_ids(self, token_ids) -> torch.Tensor:
        """Coerce a chunk's token ids to a [1, T] LongTensor on the model device."""
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        ids = token_ids.to(self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(
                f"expected token_ids of shape [T] or [1, T]; got {tuple(ids.shape)}"
            )
        return ids.long()

    def _make_mask_and_rope(self, hidden_like: torch.Tensor, positions: torch.Tensor):
        """Build the causal mask + RoPE (cos, sin) exactly the way
        ``LlamaModel.forward`` would for a sequence with the given ``positions``."""
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_like,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        position_embeddings = self.rotary_emb(hidden_like, position_ids=positions)
        return causal_mask, position_embeddings

    def _make_block_diagonal_mask_and_rope(
        self,
        hidden_like: torch.Tensor,
        positions: torch.Tensor,
        seg_lens: Sequence[tuple],
    ):
        """Build a ``[1, 1, H, H]`` BLOCK-DIAGONAL attention mask + RoPE.

        ``seg_lens`` is the packed layout as ordered ``(kind, length)`` tuples with
        ``kind`` in ``{"sink","chunk","query"}``. Connectivity: the sink is global;
        each context chunk attends only to the sink + causally within its own block
        (query-blind, chunk-isolated KV reuse); the query attends to the sink +
        every context chunk + causally within itself. RoPE positions are the SAME
        contiguous ``0:H`` as the standard read, so the only difference vs. the
        exact causal read is attention connectivity (the ablation variable)."""
        H = int(positions.shape[1])
        device = hidden_like.device

        block = torch.empty(H, dtype=torch.long, device=device)
        num_chunks = sum(1 for kind, _ in seg_lens if kind == "chunk")
        pos = 0
        chunk_c = 0
        for kind, length in seg_lens:
            length = int(length)
            if length <= 0:
                continue
            if kind == "sink":
                block[pos:pos + length] = -1
            elif kind == "chunk":
                block[pos:pos + length] = chunk_c
                chunk_c += 1
            elif kind == "query":
                block[pos:pos + length] = num_chunks
            else:
                raise ValueError(f"unknown segment kind {kind!r}")
            pos += length
        if pos != H:
            raise ValueError(f"seg_lens sum {pos} != packed length {H}")

        row_block = block.view(H, 1)
        col_block = block.view(1, H)
        is_sink_col = (col_block == -1)
        is_ctx_chunk_col = (col_block >= 0) & (col_block < num_chunks)
        is_query_row = (row_block == num_chunks)
        same_block = (row_block == col_block)

        allow = is_sink_col | same_block | (is_query_row & is_ctx_chunk_col)
        row_idx = torch.arange(H, device=device).view(H, 1)
        col_idx = torch.arange(H, device=device).view(1, H)
        causal = col_idx <= row_idx
        keep = allow & causal
        keep = keep.view(1, 1, H, H)

        ref = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_like,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        if isinstance(ref, torch.Tensor) and ref.dtype != torch.bool:
            min_val = torch.finfo(ref.dtype).min
            mask = torch.zeros(1, 1, H, H, dtype=ref.dtype, device=device)
            mask = mask.masked_fill(~keep, min_val)
        else:
            mask = keep

        position_embeddings = self.rotary_emb(hidden_like, position_ids=positions)
        return mask, position_embeddings

    @staticmethod
    def _layer_out_hidden(out):
        """Coerce a decoder layer's return to the residual-stream hidden tensor.

        In-tree transformers >=5.x decoder layers (dense AND MoE) return a bare
        ``hidden_states`` tensor. Older / custom ``trust_remote_code`` modeling may
        return a tuple whose FIRST element is the hidden state; we defensively
        unwrap that so CoMem works on both conventions."""
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        for attr in ("last_hidden_state", "hidden_states"):
            val = getattr(out, attr, None)
            if torch.is_tensor(val):
                return val
        raise TypeError(
            f"decoder layer returned unsupported type {type(out)!r}; expected a "
            "tensor or a tuple whose first element is the hidden state"
        )

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
        """Run ``self.layers[layer_slice]`` on ``hidden`` with the given mask/RoPE.

        ``past_key_values`` / ``use_cache`` drive an optional standard transformers
        KV cache (used ONLY by the resumed-band decode fast path). At the defaults
        every layer runs cache-free, so write/read/resume are byte-for-byte the
        exact recompute path. Works uniformly for dense and MoE backbones: MoE
        routing lives inside ``layer.mlp`` (position-blind), so a chunk-local WRITE
        routes each token exactly as the full-context forward would."""
        use_ckpt = (
            self.grad_checkpoint
            and torch.is_grad_enabled()
            and hidden.requires_grad
        )
        for layer in self.layers[layer_slice]:
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    lambda h, _l=layer: _l(
                        h,
                        attention_mask=causal_mask,
                        position_ids=positions,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                out = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=positions,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            hidden = self._layer_out_hidden(out)
        return hidden

    # ------------------------------------------------------------------ #
    # WRITE side: embed + layers[0:j] over one chunk (chunk-local)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write_chunk(self, token_ids) -> torch.Tensor:
        """Encode ONE chunk in isolation to depth ``j``. Returns ``h_j`` [1, T, d]."""
        ids = self._as_ids(token_ids)
        T = ids.shape[1]
        inputs_embeds = self.embed_tokens(ids)
        positions = torch.arange(T, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(
            inputs_embeds, positions
        )
        h_j = self._run_layers(
            inputs_embeds, slice(0, self.resume_j),
            causal_mask, positions, position_embeddings,
        )
        return h_j  # [1, T, d]

    @torch.no_grad()
    def write_chunks(
        self,
        chunk_list: Sequence,
        max_batch_tokens: int = 8192,
        max_batch: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Batched WRITE: encode many chunks to depth ``j`` in grouped forwards.

        Semantically IDENTICAL to ``[self.write_chunk(c) for c in chunk_list]``
        (each chunk stays chunk-local), but chunks of the same length are stacked
        along the batch axis and run through ``layers[0:j]`` in one forward. Chunks
        of different lengths are grouped separately (no padding). Returns a list of
        ``[1, T_c, d]`` depth-``j`` hiddens aligned with ``chunk_list``."""
        n = len(chunk_list)
        if n == 0:
            return []
        ids_list = [self._as_ids(c) for c in chunk_list]  # each [1, T]
        results: List[Optional[torch.Tensor]] = [None] * n

        by_len: dict = {}
        for i, ids in enumerate(ids_list):
            by_len.setdefault(int(ids.shape[1]), []).append(i)

        for T, idxs in by_len.items():
            bs = max(1, max_batch_tokens // max(1, T))
            if max_batch is not None:
                bs = min(bs, int(max_batch))
            bs = max(1, bs)
            positions = torch.arange(T, device=self.device).unsqueeze(0)  # [1,T]
            for s in range(0, len(idxs), bs):
                batch_idx = idxs[s:s + bs]
                ids = torch.cat([ids_list[i] for i in batch_idx], dim=0)  # [B,T]
                inputs_embeds = self.embed_tokens(ids)                    # [B,T,d]
                causal_mask, position_embeddings = self._make_mask_and_rope(
                    inputs_embeds, positions
                )
                h = self._run_layers(
                    inputs_embeds, slice(0, self.resume_j),
                    causal_mask, positions, position_embeddings,
                )  # [B,T,d]
                for b, i in enumerate(batch_idx):
                    results[i] = h[b:b + 1]  # [1,T,d]
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # READ side: pack cached h_j pieces + resume layers[j:] -> logits
    # ------------------------------------------------------------------ #
    def read_core(
        self,
        sink_hj: Optional[torch.Tensor],
        selected_hj_list: Sequence[torch.Tensor],
        query_hj: torch.Tensor,
        logits_tail: Optional[int] = None,
    ) -> torch.Tensor:
        """Grad-bearing read: resume from layer ``a`` over the packed sequence.

        This is the computational core of :meth:`read` without the ``no_grad``
        guard, so the distillation trainer can run it inside its own autograd
        context (the trainable LoRA lives in ``layers[a:]``).

        * ``top_prepay_b == 0`` — exact connective resume: pack
          ``[sink ; ctx... ; query]`` with fresh contiguous RoPE positions, run
          ``layers[a:L]`` + norm + lm_head; logits for every packed position
          (or the last ``logits_tail`` positions). If ``block_diagonal`` is set,
          the causal mask is replaced by the block-diagonal one.
        * ``top_prepay_b > 0`` — approximate top-prepay: recompute the middle band
          over the full pack, then push only the query tail through the top band
          with fresh contiguous positions; logits for the query tail only.
        """
        pieces: List[torch.Tensor] = []
        seg_lens: List[tuple] = []
        if sink_hj is not None:
            pieces.append(sink_hj)
            seg_lens.append(("sink", int(sink_hj.shape[1])))
        for h in selected_hj_list:
            if h is not None and h.shape[1] > 0:
                pieces.append(h)
                seg_lens.append(("chunk", int(h.shape[1])))
        pieces.append(query_hj)
        seg_lens.append(("query", int(query_hj.shape[1])))

        packed = torch.cat(pieces, dim=1)  # [1, |H|, d]
        H = packed.shape[1]
        positions = torch.arange(H, device=self.device).unsqueeze(0)
        if self.block_diagonal:
            if self.top_prepay_b != 0:
                raise ValueError(
                    "block_diagonal read is only defined for top_prepay_b == 0 "
                    f"(the exact-depth resume path); got b={self.top_prepay_b}"
                )
            causal_mask, position_embeddings = self._make_block_diagonal_mask_and_rope(
                packed, positions, seg_lens
            )
        else:
            causal_mask, position_embeddings = self._make_mask_and_rope(packed, positions)

        if self.top_prepay_b == 0:
            hidden = self._run_layers(
                packed, slice(self.resume_j, self.num_layers),
                causal_mask, positions, position_embeddings,
            )
            if logits_tail is not None and logits_tail > 0:
                hidden = hidden[:, -int(logits_tail):, :]
            hidden = self.norm(hidden)
            return self.lm_head(hidden)  # [1, |H| or tail, V]

        # --- b > 0: recompute middle band (query-aware) over full pack ---
        mid = self._run_layers(
            packed, slice(self.resume_j, self.mid_end),
            causal_mask, positions, position_embeddings,
        )

        T_q = int(query_hj.shape[1])
        top_in = mid[:, -T_q:, :]
        top_pos = torch.arange(T_q, device=self.device).unsqueeze(0)
        top_mask, top_pe = self._make_mask_and_rope(top_in, top_pos)
        top_out = self._run_layers(
            top_in, slice(self.mid_end, self.num_layers),
            top_mask, top_pos, top_pe,
        )
        if logits_tail is not None and logits_tail > 0:
            top_out = top_out[:, -int(logits_tail):, :]
        top_out = self.norm(top_out)
        return self.lm_head(top_out)  # [1, T_q or tail, V]

    @torch.no_grad()
    def read(
        self,
        sink_hj: Optional[torch.Tensor],
        selected_hj_list: Sequence[torch.Tensor],
        query_hj: torch.Tensor,
    ) -> torch.Tensor:
        """Inference read (``no_grad`` wrapper around :meth:`read_core`)."""
        return self.read_core(sink_hj, selected_hj_list, query_hj)

    # ------------------------------------------------------------------ #
    # resumed-band KV cache decode — O(1)/step generation
    # ------------------------------------------------------------------ #
    def _decode_attn_mask(self, kv_len: int):
        """Attention mask for a single-query decode step attending to all
        ``kv_len`` cached keys. For SDPA / FlashAttention ``None`` + ``q_len==1``
        attends to all keys (cheapest); for eager we return an all-zero float mask."""
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        if attn_impl in ("sdpa", "flash_attention_2", "flash_attention_3"):
            return None
        return torch.zeros(1, 1, 1, kv_len, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def write_prefill(self, token_ids):
        """Bottom-band prefill WITH a KV cache (query-only helper for decode).
        Returns ``(h_j [1,T,d], bottom_cache, T)``."""
        ids = self._as_ids(token_ids)
        T = ids.shape[1]
        inputs_embeds = self.embed_tokens(ids)
        positions = torch.arange(T, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(
            inputs_embeds, positions
        )
        cache = DynamicCache(config=self.config)
        h_j = self._run_layers(
            inputs_embeds, slice(0, self.resume_j),
            causal_mask, positions, position_embeddings,
            past_key_values=cache, use_cache=True,
        )
        return h_j, cache, T

    @torch.no_grad()
    def read_prefill(self, sink_hj, selected_hj_list, query_hj):
        """Top-band prefill WITH a KV cache; returns ``(logits_last [1,1,V],
        top_cache, H)``. Only supported for the exact resume."""
        if self.top_prepay_b != 0 or self.block_diagonal:
            raise NotImplementedError(
                "resumed-band KV cache decode is only defined for the exact resume "
                f"(top_prepay_b == 0, block_diagonal == False); got "
                f"top_prepay_b={self.top_prepay_b}, block_diagonal={self.block_diagonal}"
            )
        pieces: List[torch.Tensor] = []
        if sink_hj is not None:
            pieces.append(sink_hj)
        for h in selected_hj_list:
            if h is not None and h.shape[1] > 0:
                pieces.append(h)
        pieces.append(query_hj)
        packed = torch.cat(pieces, dim=1)  # [1, H, d]
        H = packed.shape[1]
        positions = torch.arange(H, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(packed, positions)
        cache = DynamicCache(config=self.config)
        hidden = self._run_layers(
            packed, slice(self.resume_j, self.num_layers),
            causal_mask, positions, position_embeddings,
            past_key_values=cache, use_cache=True,
        )
        last = self.norm(hidden[:, -1:, :])
        logits_last = self.lm_head(last)  # [1, 1, V]
        return logits_last, cache, H

    @torch.no_grad()
    def decode_step(self, token_id, bottom_cache, top_cache, q_local_pos, pack_pos):
        """One O(1) decode step: push a single new token through both bands.
        Returns ``logits_last [1,1,V]``; both caches are extended in place."""
        ids = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        emb = self.embed_tokens(ids)  # [1, 1, d]
        if self.resume_j > 0:
            b_pos = torch.tensor([[int(q_local_pos)]], device=self.device)
            b_pe = self.rotary_emb(emb, position_ids=b_pos)
            b_mask = self._decode_attn_mask(int(q_local_pos) + 1)
            new_hj = self._run_layers(
                emb, slice(0, self.resume_j),
                b_mask, b_pos, b_pe,
                past_key_values=bottom_cache, use_cache=True,
            )
        else:
            new_hj = emb  # j == 0: h_j IS the embedding
        t_pos = torch.tensor([[int(pack_pos)]], device=self.device)
        t_pe = self.rotary_emb(new_hj, position_ids=t_pos)
        t_mask = self._decode_attn_mask(int(pack_pos) + 1)
        hidden = self._run_layers(
            new_hj, slice(self.resume_j, self.num_layers),
            t_mask, t_pos, t_pe,
            past_key_values=top_cache, use_cache=True,
        )
        hidden = self.norm(hidden)
        return self.lm_head(hidden)  # [1, 1, V]

    # ------------------------------------------------------------------ #
    # convenience: full split forward on a single packed token sequence
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def resume_forward_ids(self, token_ids) -> torch.Tensor:
        """Split-at-j forward over a SINGLE contiguous token sequence. At ``j=0``
        this equals the stock ``model(input_ids)`` forward to fp tolerance."""
        ids = self._as_ids(token_ids)
        T = ids.shape[1]
        inputs_embeds = self.embed_tokens(ids)
        positions = torch.arange(T, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(
            inputs_embeds, positions
        )
        hidden = self._run_layers(
            inputs_embeds, slice(0, self.resume_j),
            causal_mask, positions, position_embeddings,
        )
        hidden = self._run_layers(
            hidden, slice(self.resume_j, self.num_layers),
            causal_mask, positions, position_embeddings,
        )
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    @torch.no_grad()
    def full_forward_logits(self, token_ids) -> torch.Tensor:
        """Stock ``model(input_ids)`` logits — the self-test reference."""
        ids = self._as_ids(token_ids)
        return self.model(input_ids=ids, use_cache=False).logits

    # ------------------------------------------------------------------ #
    # generation: shared decode core (recompute or resumed-band KV cache)
    # ------------------------------------------------------------------ #
    def _bos_eos(self, tokenizer, fallback_first=None):
        tok = tokenizer if tokenizer is not None else self.tokenizer
        bos = getattr(tok, "bos_token_id", None) if tok is not None else None
        eos = getattr(tok, "eos_token_id", None) if tok is not None else None
        if bos is None and fallback_first is not None:
            bos = int(fallback_first)
        return bos, eos

    @torch.no_grad()
    def _decode_from_pack(
        self,
        sink_hj,
        selected_hj,
        query_ids,             # list[int] query-chunk token ids
        eos_id,
        max_new_tokens: int,
        use_kv_cache: bool,
        stats=None,
        n_context_chunks: Optional[int] = None,
    ) -> List[int]:
        """Greedy decode from the packed read. Returns the generated token ids.

        Shared by :meth:`generate` and :meth:`generate_from_ids` so the two entry
        points cannot diverge. The recompute path (``use_kv_cache=False``) re-runs
        both bands over their full sequences every step; the KV-cache path prefills
        both bands once and pushes ONE token/step. The two produce identical logits
        to fp tolerance (correctness gate)."""
        generated: List[int] = []

        if stats is not None:
            sink_len = int(sink_hj.shape[1]) if sink_hj is not None else 0
            sel_len = int(sum(h.shape[1] for h in selected_hj))
            stats["read_len"] = sink_len + sel_len + len(query_ids)
            stats["n_selected_chunks"] = len(selected_hj)
            if n_context_chunks is not None:
                stats["n_context_chunks"] = n_context_chunks

        capture = stats is not None and stats.get("capture_step_logits")
        step_logits = [] if capture else None

        can_kv = (use_kv_cache and self.top_prepay_b == 0
                  and not self.block_diagonal)
        if can_kv:
            q_hj, bottom_cache, q_local_pos = self.write_prefill(query_ids)
            logits1, top_cache, pack_pos = self.read_prefill(sink_hj, selected_hj, q_hj)
            next_logits = logits1[0, -1].float()
            if eos_id is not None:
                next_logits[eos_id] = float("-inf")  # step 0 never emits EOS
            if capture:
                step_logits.append(next_logits.detach().clone())
            next_tok = int(next_logits.argmax().item())
            generated.append(next_tok)
            for step in range(1, max_new_tokens):
                logits = self.decode_step(
                    next_tok, bottom_cache, top_cache, q_local_pos, pack_pos)
                q_local_pos += 1
                pack_pos += 1
                next_logits = logits[0, -1].float()
                if capture:
                    step_logits.append(next_logits.detach().clone())
                next_tok = int(next_logits.argmax().item())
                if eos_id is not None and next_tok == eos_id:
                    break
                generated.append(next_tok)
        else:
            cur = list(query_ids)
            for step in range(max_new_tokens):
                q_hj = self.write_chunk(cur)
                logits = self.read(sink_hj, selected_hj, q_hj)   # [1, |H|, V]
                next_logits = logits[0, -1].float()
                if step == 0 and eos_id is not None:
                    next_logits[eos_id] = float("-inf")
                if capture:
                    step_logits.append(next_logits.detach().clone())
                next_tok = int(next_logits.argmax().item())
                if eos_id is not None and next_tok == eos_id and step > 0:
                    break
                generated.append(next_tok)
                cur = cur + [next_tok]

        if capture:
            stats["step_logits"] = step_logits
            stats["generated_ids"] = list(generated)
        return generated

    # ------------------------------------------------------------------ #
    # monolithic entry point (query == last chunk of a single prompt)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_from_ids(
        self,
        input_ids: torch.Tensor,      # [1, L] full formatted prompt
        *,
        chunk_size: int = 512,
        max_new_tokens: int = 20,
        selector: str = "bm25",
        topk: int = 12,
        sink_tokens: str = "bos",
        needle_chunk_set=None,
        bare_question_ids=None,
        no_retrieval: bool = False,
        stats=None,
        iter_rounds: int = 0,
        iter_hop_topk: int = 2,
        iter_score: str = "meanpool",
        iter_conf_ratio: float = 0.3,
        iter_max_chunks: int = 64,
        use_kv_cache: bool = True,
        tokenizer=None,
    ) -> str:
        """Chunk a single prompt (query == the LAST chunk), write each chunk to
        depth ``j``, select ``topk`` context chunks, then greedily decode from the
        packed read. This is the reference forward path; the ergonomic
        :meth:`encode` + :meth:`generate` pair reproduces it token-for-token when
        the context is a multiple of ``chunk_size`` and the query is one chunk.

        ``no_retrieval`` packs ALL context chunks (the KV-Direct / HCache baseline,
        read grows O(context)); otherwise ``selector`` + ``topk`` keep a fixed read."""
        tok = tokenizer if tokenizer is not None else self.tokenizer
        tokens = input_ids[0]
        chunks = list(tokens.split(chunk_size))
        context_chunks = chunks[:-1]
        query_chunk = chunks[-1]

        sink_hj = None
        if sink_tokens == "bos":
            bos_id, _ = self._bos_eos(tok, fallback_first=int(tokens[0].item()))
            sink_hj = self.write_chunk([bos_id])

        context_hj = None
        query_hj_for_sel = None
        if not no_retrieval and selector in ("reader_attn", "iter_reader_attn"):
            context_hj = self.write_chunks(context_chunks)
            query_hj_for_sel = self.write_chunk(query_chunk)

        if no_retrieval:
            sel_idx = list(range(len(context_chunks)))
        else:
            sel_idx = _sel.select_context_chunk_indices(
                selector, context_chunks, bare_question_ids or [], topk,
                needle_chunk_set, context_hj=context_hj, query_hj=query_hj_for_sel,
                iter_rounds=iter_rounds, iter_hop_topk=iter_hop_topk,
                iter_score=iter_score,
                iter_conf_ratio=iter_conf_ratio, iter_max_chunks=iter_max_chunks,
            )

        if context_hj is not None:
            selected_hj = [context_hj[i] for i in sel_idx]
        else:
            selected_hj = self.write_chunks([context_chunks[i] for i in sel_idx])

        _, eos_id = self._bos_eos(tok)
        generated = self._decode_from_pack(
            sink_hj, selected_hj, query_chunk.tolist(), eos_id,
            max_new_tokens, use_kv_cache, stats=stats,
            n_context_chunks=len(context_chunks),
        )
        if tok is not None:
            return tok.decode(generated, skip_special_tokens=True).strip()
        return generated

    # ------------------------------------------------------------------ #
    # ergonomic API: encode(context) once, generate(query) many times
    # ------------------------------------------------------------------ #
    def encode_ids(self, context_ids, chunk_size: int = 512, sink_tokens: str = "bos"):
        """Comprehend a context given as token ids: split into ``chunk_size``
        chunks, write each to depth ``j`` (cached ``h_j``), and record the raw
        chunk token ids (for lexical retrieval) + the sink. Returns ``self``."""
        ids = self._as_ids(context_ids)[0]
        chunks = list(ids.split(chunk_size))
        self._ctx_chunks = [c for c in chunks]
        self._ctx_hj = self.write_chunks(self._ctx_chunks)
        self._chunk_size = int(chunk_size)
        self._sink_tokens = sink_tokens
        self._sink_hj = None
        if sink_tokens == "bos":
            bos_id, _ = self._bos_eos(self.tokenizer,
                                      fallback_first=int(ids[0].item()))
            self._sink_hj = self.write_chunk([bos_id])
        return self

    def encode(self, context: str, chunk_size: int = 512, sink_tokens: str = "bos"):
        """Comprehend a context string (requires ``self.tokenizer``). Tokenises
        with ``add_special_tokens=True`` (BOS-prefixed) then delegates to
        :meth:`encode_ids`."""
        if self.tokenizer is None:
            raise ValueError("encode(text) needs a tokenizer; pass tokenizer= to "
                             "CoMem(...) or call encode_ids on token ids")
        ids = self.tokenizer.encode(context, add_special_tokens=True,
                                    return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        return self.encode_ids(ids, chunk_size=chunk_size, sink_tokens=sink_tokens)

    @torch.no_grad()
    def generate_ids(
        self,
        query_ids,                    # list[int] / tensor: the query chunk
        *,
        selector: str = "bm25",
        topk: int = 12,
        mode: str = "comem",
        max_new_tokens: int = 32,
        bm25_query_ids=None,
        needle_chunk_set=None,
        iter_rounds: int = 0,
        iter_hop_topk: int = 2,
        iter_score: str = "meanpool",
        iter_conf_ratio: float = 0.3,
        iter_max_chunks: int = 64,
        use_kv_cache: bool = True,
        stats=None,
    ) -> List[int]:
        """Answer a query (token ids) against the previously :meth:`encode`-d
        context. Selects ``topk`` cached chunks, packs the read, greedy-decodes.
        Returns the generated token ids. ``mode`` in {comem, kvdirect, hcache}
        toggles retrieval (kvdirect/hcache pack all chunks; build CoMem(resume_j=0)
        for a faithful kvdirect)."""
        if mode not in _MODE_NO_RETRIEVAL:
            raise ValueError(f"unknown mode {mode!r}; expected {list(_MODE_NO_RETRIEVAL)}")
        no_retrieval = _MODE_NO_RETRIEVAL[mode]

        q = self._as_ids(query_ids)[0]
        query_ids_list = q.tolist()
        query_chunk = q
        context_chunks = self._ctx_chunks
        context_hj = self._ctx_hj

        query_hj_for_sel = None
        if not no_retrieval and selector in ("reader_attn", "iter_reader_attn"):
            query_hj_for_sel = self.write_chunk(query_chunk)

        if no_retrieval:
            sel_idx = list(range(len(context_chunks)))
        else:
            bq = bm25_query_ids if bm25_query_ids is not None else query_ids_list
            sel_idx = _sel.select_context_chunk_indices(
                selector, context_chunks, bq, topk, needle_chunk_set,
                context_hj=context_hj, query_hj=query_hj_for_sel,
                iter_rounds=iter_rounds, iter_hop_topk=iter_hop_topk,
                iter_score=iter_score,
                iter_conf_ratio=iter_conf_ratio, iter_max_chunks=iter_max_chunks,
            )

        selected_hj = [context_hj[i] for i in sel_idx]
        _, eos_id = self._bos_eos(self.tokenizer)
        return self._decode_from_pack(
            self._sink_hj, selected_hj, query_ids_list, eos_id,
            max_new_tokens, use_kv_cache, stats=stats,
            n_context_chunks=len(context_chunks),
        )

    @torch.no_grad()
    def generate(
        self,
        query: str,
        *,
        selector: str = "bm25",
        topk: int = 12,
        mode: str = "comem",
        max_new_tokens: int = 32,
        bm25_query: Optional[str] = None,
        needle_chunk_set=None,
        iter_rounds: int = 0,
        iter_hop_topk: int = 2,
        iter_score: str = "meanpool",
        iter_conf_ratio: float = 0.3,
        iter_max_chunks: int = 64,
        use_kv_cache: bool = True,
        stats=None,
    ) -> str:
        """Answer a query string against the previously :meth:`encode`-d context
        (requires ``self.tokenizer``). See :meth:`generate_ids` for the semantics.
        Returns the decoded answer string."""
        if self.tokenizer is None:
            raise ValueError("generate(text) needs a tokenizer; pass tokenizer= to "
                             "CoMem(...) or call generate_ids on token ids")
        q_ids = self.tokenizer.encode(query, add_special_tokens=False)
        bq_ids = None
        if bm25_query is not None:
            bq_ids = self.tokenizer.encode(bm25_query, add_special_tokens=False)
        generated = self.generate_ids(
            q_ids, selector=selector, topk=topk, mode=mode,
            max_new_tokens=max_new_tokens, bm25_query_ids=bq_ids,
            needle_chunk_set=needle_chunk_set, iter_rounds=iter_rounds,
            iter_hop_topk=iter_hop_topk, iter_score=iter_score,
            iter_conf_ratio=iter_conf_ratio, iter_max_chunks=iter_max_chunks,
            use_kv_cache=use_kv_cache, stats=stats,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
