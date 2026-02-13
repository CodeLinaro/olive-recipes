#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


'''
Model adaptations required for Qwen 2.5VL ViT
'''


from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention, Qwen2_5_VisionTransformerPretrainedModel, repeat_kv
import torch
import torch.nn as nn
from typing import Optional


def _apply_rope_single(x, rope_vals: tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real,x_prod_im),dim=2).view(*x.shape)
    return x



class QcQwen2_5_VLVisionAttention(Qwen2_5_VLVisionAttention):

    def __init__(self, config):
        super().__init__(config)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,

    ):
        # The adapted attention class forward is a combination of `Qwen2_5_VLVisionAttention` and `eager_attention_forward` from modeling_qwen2_5_vl
        
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            cos = rotary_pos_emb.cos()
            sin = rotary_pos_emb.sin()
        else:
            cos, sin = position_embeddings
        
        orig_q_dtype = query_states.dtype
        orig_k_dtype = key_states.dtype
        query_states, key_states = query_states.float(), key_states.float()
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        query_states = _apply_rope_single(query_states, (cos, sin))
        key_states = _apply_rope_single(key_states, (cos, sin))
        query_states = query_states.to(orig_q_dtype)
        key_states = key_states.to(orig_k_dtype)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            if attention_mask.shape[-1] != value_states.shape[-2]:
                attention_mask = attention_mask[:, :, :, : value_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output
        
        

class QcQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings_cos: torch.Tensor,
            position_embeddings_sin: torch.Tensor,
            full_attention_mask: torch.Tensor,
            window_attention_mask: torch.Tensor,
    ):
        # We have yanked out the initial and ending part of `Qwen2_5_VisionTransformerPretrainedModel.forward`
        # The reason is that we cannot pass in `window_index`, `cu_seqlens`, `cu_window_seqlens`, hence the permutation of `hidden_states` according to `window_index` is now done outside
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = position_embeddings_cos, position_embeddings_sin

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask = full_attention_mask
            else:
                attention_mask = window_attention_mask

            hidden_states = blk(
                hidden_states,
                # We do not need to use `cu_seqlens` in the attention operation since we are using eager attention. We pass a placeholder tensor since the `Qwen2_5_VLVisionBlock` class expects this argument
                cu_seqlens=torch.tensor([]),
                rotary_pos_emb=None,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.merger(hidden_states)

        return hidden_states

        