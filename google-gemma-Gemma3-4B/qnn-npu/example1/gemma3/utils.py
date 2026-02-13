#!/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""  This file provides utilities that the pipeline would need to work with the adaptations made to Gemma3 model. """

import torch
import functools
from typing import Optional, List
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3RotaryEmbedding

def llm_update_causal_mask(prepared_1d_attn_mask, input_tensor, max_input_tokens, model_context_len, model_id_or_path, mask_neg = -100.0, 
                           cache_index=None, pad_to_left = True, sliding_window = None, token_type_ids=None):
    '''

    This function creates a causal mask (2D) from the 1D attention mask

    params:
    1. prepared_1d_attn_mask: attention mask of shape (batch_size, model_context_length)
    2. input_tensor : input_ids/ input_embeddings
    3. max_input_tokens: maximum number of tokens that can be consumed by the model at each inference (equals ARN)
    4. model_context_len: maximum number of tokens that the model can consume in total
    5. model_id: Model name or path to pretrained model
    6. mask_neg: proxy for minus infinity since minus infinity is not quantization friendly. This value should be large
    enough to drown out tokens that should not be attended to
    7. cache_index: the index for the starting position of kvcaches
    8. pad_to_left: determines if the KV cache is padded to the left or right
    9. sliding_window: this represents the length of window each token should look at if we want the sliding window mask

    '''
    gemma3_model = _get_model(model_id_or_path)

    # if the cache position is None, then we assume that the current input ids will be concatenated to the right end and
    # hence we construct the cache position accordingly to be sent into the update_causal_mask

    if pad_to_left:
        # Cache index should not be passed. Concat op is used in doing the KV cache update
        assert cache_index is None, "Invalid argument error: we do not support the combination of performing left padding and doing scatter for KV cache update."
    else:
        # if the user is doing right padding, it is necessary to pass the cache_index.
        assert cache_index is not None, "Invalid argument error: we do not support the combination of performing right padding and doing concat for KV cache update"

    if cache_index is None:
        cache_position = torch.arange(model_context_len-max_input_tokens, model_context_len, device = input_tensor.device)
    else:
        cache_position = torch.arange(max_input_tokens, dtype=torch.float32, device=input_tensor.device) + cache_index.to(input_tensor.device)

    input_embeds = torch.ones((input_tensor.shape[0], input_tensor.shape[1], 1), device = input_tensor.device)
    
    prepared_attention_mask = gemma3_model._update_causal_mask(attention_mask=prepared_1d_attn_mask, input_tensor=input_embeds, output_attentions=True,
                              cache_position=cache_position, past_key_values=None)

    # Apply bidirectional mask on images if token type ids are provided (only for prefill stage)
    if token_type_ids is not None and input_tensor.shape[1] != 1:
        prompt_length = token_type_ids.shape[-1]  # token_type_ids length should be the same as the actual prompt length (exclude padding)
        token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
        token_type_mask[token_type_ids == 0] = False  # if text token do not change anything
        token_type_mask = token_type_mask.unsqueeze(1).to(prepared_attention_mask.device, dtype=torch.bool)
        prepared_attention_mask = prepared_attention_mask.clone()
        # assume pad_to_left is True here for static model
        prepared_attention_mask[:, :, -prompt_length:, -prompt_length:] = prepared_attention_mask[:, :, -prompt_length:, -prompt_length:].masked_fill(token_type_mask, 0.0)

    prepared_attention_mask = prepared_attention_mask.clamp_min(mask_neg)

    if sliding_window is not None:
        # if we want to get the sliding window mask, we need to make the original mask strided so the model only looks at the KV$ which lies within it's sliding window
        effective_seq_len = max(cache_position.shape[0], sliding_window)  # for dynamic input case
        min_dtype = torch.finfo(prepared_attention_mask.dtype).min

        # this is a triangular mask which reflects where should we mask out.
        """
        For instance, suppose we have the attention mask of shape [1, 1, 8192, 8192], now for the sliding layer, the tokens can only look at the most recent 1024 tokens including themselves, 
        this means that we can no longer use this let's say hypothetical lower triangular matrix since the token at 1024th position [1025th] will look at the token at index 0 (first token), so we need to create strides.
        """
        sliding_window_mask = torch.tril(
            torch.ones_like(prepared_attention_mask, dtype=torch.bool), diagonal=-sliding_window
        )

        # the masked positions should be made min so we don't attend to those, else retain that position
        prepared_attention_mask = torch.where(sliding_window_mask, min_dtype, prepared_attention_mask)
        # In case we are beyond the sliding window, we need to correctly offset the mask slicing
        # `last_cache_position` is equivalent to `cache_position[-1]` but without breaking dynamo

        # the following section of code will make our prepared causal mask's last dimension set to sliding window length/ or the initial prompt length, this is because in the HF/ original implementation, 
        # we will send the whole input prompt at once, and then we will have to stride our attn mask to only look at most recent sliding window tokens.
        
        offset = cache_position[-1] - effective_seq_len
        
        # Should only be used when beyond the sliding window (i.e. offset > 0)
        offset = max(0, offset)

        prepared_attention_mask = prepared_attention_mask[:, :, :, offset: offset + effective_seq_len]
    
    return prepared_attention_mask


def llm_create_position_embeddings(config, position_ids=None):
    '''
    This function creates position embedding (RoPE) from the position ids.
    params:
    1. config: model configuration to create the GaussRotaryEmbedding object, expect config for backward compatibility, in future transformers, we only expect to pass the config, the one we have in docker, takes in the req argument
    2. position_ids: required position ids passed into the model
    '''

    hidden_size = config.hidden_size
    max_position_embeddings = config.max_position_embeddings
    num_attention_heads = config.num_attention_heads
    rope_theta = config.rope_theta
    dim = config.head_dim if hasattr(config, "head_dim") else int((hidden_size // num_attention_heads))
    device = position_ids.device
    x = torch.ones(1, device=device)
    rotary_emb = _get_rotary_embedding(config=config)
    cos, sin = rotary_emb(x, position_ids=position_ids)
    cos, sin = cos.unsqueeze(dim = 1), sin.unsqueeze(dim = 1)
    cos = cos[:,:,:,:dim//2]
    sin = sin[:,:,:, :dim//2]
    return cos, sin

def _get_rotary_embedding(config=None):
    rotary_emb = Gemma3RotaryEmbedding(config=config)
    return rotary_emb

def llm_get_kv_length(outputs, global_layer_idx, layer_indices_to_perform_sliding_eviction):
    # kv_length (global layer), kv_length_sliding (sliding layer)
    if outputs['past_key_values'] is None:
        kv_length = 0
        kv_length_sliding = 0
    elif not isinstance(outputs['past_key_values'], tuple):
        kv_length = outputs['past_key_values'].get_seq_length(global_layer_idx)
        kv_length_sliding = outputs['past_key_values'].get_seq_length(layer_indices_to_perform_sliding_eviction[0])
    else:
        kv_length = outputs['past_key_values'][global_layer_idx][1].shape[-2]
        kv_length_sliding = outputs['past_key_values'][layer_indices_to_perform_sliding_eviction[0]][1].shape[-2]
    return kv_length, kv_length_sliding

def lmm_preprocess_inputs(input_ids=None, pixel_values=None, inputs_embeds=None, past_key_values=None, image_token_index=None,
                          embedding_layer=None, vision_model=None):
    """
        Preprocess the inputs to accommodate image features to inputs_embeds, by default the first inference shouldn't
        contain past_key_values to run multimodal generation
        Inputs:
            input_ids: input_ids sent to the model
            pixel_values: images sent to the model
            inputs_embeds: embedded inputs sent to the model
            past_key_values: past_key_valyes sent to the model
            image_token_index: image token id defined in config
            embedding_layer: the embedding layer used to embed input_ids
            vision_model: the model (vision tower + projector) used to generate image features from given images
        Output:
            input_embeds: embedded inputs

    """

    if past_key_values is None or len(past_key_values) == 0:
        assert inputs_embeds is None, "inputs_embeds should be generated from the input_ids and image_embeds " \
                                      "(if provided) for the first inference (none kvcache mode)"
        assert input_ids is not None, f"input_ids should be provided for the first inference"
        if not _is_empty(pixel_values):  # multi-modality
            assert embedding_layer is not None
            assert vision_model is not None

            assert len(input_ids) == len(pixel_values), f"{len(input_ids)} {len(pixel_values)}"
            multi_flags = [True if image_token_index in input_id.tolist() else False for input_id in input_ids]

            inputs_embeds = embedding_layer(input_ids)
            image_features = vision_model(pixel_values)

            # Process inputs_embeds for each sample and batch them back to tensors
            new_input_embeds = []
            image_count = 0
            for i in range(len(input_ids)):
                input_id = input_ids[i]
                input_embed = inputs_embeds[image_count]
                image_feature = image_features[image_count]
                if multi_flags[i]:
                    image_mask = ((input_id == image_token_index).unsqueeze(-1).expand_as(input_embed).to(input_embed.device))
                    image_feature = image_feature.to(input_embed.device, input_embed.dtype)
                    new_input_embeds.append(input_embed.masked_scatter(image_mask, image_feature))
                    image_count += 1
                else:
                    new_input_embeds.append(inputs_embeds[i])
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            return inputs_embeds

    return embedding_layer(input_ids)

def _is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if image_list is not None:
            return False
    return True

@functools.cache
def _get_model(model_id_or_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code = True)
    if hasattr(config, "text_config"):  # LMM
        config.text_config.num_hidden_layers = 1
        model = Gemma3TextModel(config.text_config)
    else:                               # LLM
        config.num_hidden_layers = 1
        model = Gemma3TextModel(config)
    return model

def slice_token_type_ids(token_type_ids, max_input_tokens):
    # token_type_ids needed only during prompt processing stage, and the length of token_type_ids should be the same as length of prompt
    input_length = token_type_ids.shape[-1]

    # assume remainder_first slicing
    token_type_ids_slice = []
    for idx in range(0, input_length, max_input_tokens)[::-1]:
        idx = input_length - idx
        slice_beginning = max(0, idx-max_input_tokens)
        token_type_ids_slice.append(token_type_ids[:,slice_beginning:idx])

    return token_type_ids_slice
