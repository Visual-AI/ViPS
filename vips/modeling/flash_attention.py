from __future__ import annotations

from typing import Any

import torch


def install_flattened_attention() -> None:
    """Enable the flattened FlashAttention path used by Qwen's SFT code."""

    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    apply_rotary_pos_emb = modeling_qwen3_vl.apply_rotary_pos_emb

    def attention_forward(
        self: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cosine, sine = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cosine, sine)

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sine,
                "cos": cosine,
                "cache_position": cache_position,
            }
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )

        query = query.transpose(1, 2).squeeze(0)
        key = key.transpose(1, 2).squeeze(0)
        value = value.transpose(1, 2).squeeze(0)
        # FlashAttention requires cumulative sequence lengths to be int32.
        # Some Trainer/DeepSpeed paths promote integer inputs to int64.
        lengths = attention_mask.to(dtype=torch.int32)
        max_length = max(
            lengths[index + 1] - lengths[index]
            for index in range(lengths.shape[0] - 1)
        ).item()
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=lengths,
            cu_seqlens_k=lengths,
            max_seqlen_q=max_length,
            max_seqlen_k=max_length,
            causal=True,
        )
        output = output.unsqueeze(0).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), None

    def return_flattened_mask(
        config: Any,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Any,
        position_ids: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return attention_mask

    modeling_qwen3_vl.Qwen3VLTextAttention.forward = attention_forward
    modeling_qwen3_vl.create_causal_mask = return_flattened_mask
