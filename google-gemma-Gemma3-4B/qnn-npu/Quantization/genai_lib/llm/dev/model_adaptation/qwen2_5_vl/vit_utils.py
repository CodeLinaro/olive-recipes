#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


'''
Utils required for Qwen 2.5VL ViT Onboarding
'''


import torch
import functools
import torch.nn.functional as F



def vit_get_window_params(visual_model, hidden_states, grid_thw):
    '''
    Precompute window attention related parameters
    We yank this section out from `Qwen2_5_VisionTransformerPretrainedModel.forward`
    '''

    window_index, cu_window_seqlens = visual_model.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=hidden_states.device,
                dtype=torch.int32,
            )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                dtype=torch.int32,
            )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    spatial_merge_unit = visual_model.spatial_merge_unit

    return window_index, cu_window_seqlens, cu_seqlens, spatial_merge_unit



def vit_prepare_attention_mask(inputs_tensor, cu_seqlens, mask_neg = -1e3) -> torch.Tensor:
    '''
    We yank out the attention mask preparation for ViT attention from transformers 4.53
    Refer to `Qwen2_5_VisionTransformerPretrainedModel._prepare_attention_mask`
    Transformers recent version apply attention on each window separately which excessively tiles the computation graph
    Hence we stick to the transformers<=4.53 way of creating the windowed-attention aware attention mask
    '''

    seq_length = inputs_tensor.shape[0]
    attention_mask = torch.full(
        [1, 1, seq_length, seq_length],
        torch.finfo(inputs_tensor.dtype).min,
        device=inputs_tensor.device,
        dtype=inputs_tensor.dtype,
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    
    attention_mask = attention_mask.clamp_min(mask_neg)

    return attention_mask



def vit_get_position_embeddings(visual_model, hidden_states, grid_thw, window_index):
    '''
    Precompute RoPE position embeddings
    We yank this section out from `Qwen2_5_VisionTransformerPretrainedModel.forward`
    '''

    seq_len, _ = hidden_states.size()

    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual_model.spatial_merge_unit, visual_model.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    position_embeddings_cos = rotary_pos_emb.cos()
    position_embeddings_sin = rotary_pos_emb.sin()

    return position_embeddings_cos.to(hidden_states.device), position_embeddings_sin.to(hidden_states.device)



@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    config = AutoConfig.from_pretrained(model_id_or_path)
    config.num_hidden_layers = config.text_config.num_hidden_layers = config.vision_config.depth = 1
    model = Qwen2_5_VLModel(config)
    return model