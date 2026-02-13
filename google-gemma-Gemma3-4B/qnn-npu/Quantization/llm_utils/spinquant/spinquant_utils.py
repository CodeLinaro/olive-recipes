from typing import Any, Iterable

import torch
import torch.nn as nn
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from .transforms import HadamardTransform


def _default_r2_fusion_func(layer):
    """Default R2 fusion function"""
    r2_direction_pairs = []
    r2_direction_pairs.extend(
        [
            (layer.self_attn.v_proj, False),
            (layer.self_attn.o_proj, True),
        ]
    )
    return r2_direction_pairs


@torch.no_grad()
def apply_spinquant_r2(model: Gemma3ForCausalLM, config: Any) -> None:
    device = model.device
    model_config: Gemma3TextConfig = model.config

    hidden_size = model_config.hidden_size
    if hidden_size % model_config.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    # R2 transform (not supporting online rotations for Gemma3: R1/R3/R4)
    for layer_idx, layer in enumerate(model.model.layers):
        had_transform = HadamardTransform(
            size=model_config.head_dim, init_type="randomized_hadamard", device=device
        )

        modules_and_fuse_directions = _default_r2_fusion_func(layer)

        for module, fuse_before in modules_and_fuse_directions:
            _fuse_r2_rotation(module, fuse_before, had_transform, model_config.head_dim)


def _fuse_r2_rotation(module, fuse_before, had_transform, had_dim):
    with torch.no_grad():
        if isinstance(module, torch.nn.Linear):
            W_tmp = module.weight.data

            if not fuse_before:
                W_tmp = W_tmp.t()
            
            shape_tmp = W_tmp.shape
            last_dim = shape_tmp[-1]
            blocks = last_dim // had_dim
            W_tmp = W_tmp.reshape(-1, blocks, had_dim)
            
            # Perform Hadamard transform on weights
            W_tmp = had_transform.apply_transform(W_tmp)

            W_tmp = W_tmp.reshape(shape_tmp)

            if not fuse_before:
                W_tmp = W_tmp.t()

            # Restore rotated weights
            module.weight.data = W_tmp

