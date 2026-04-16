# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import math

import onnx
import torch
import torch.nn.functional as F

from onnxsim import simplify
from torch import nn
from transformers import Swin2SRForImageSuperResolution

def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.reshape(
        batch_size * height // window_size, window_size, width // window_size, window_size * num_channels
    )
    windows = input_feature.permute(0, 2, 1, 3).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.reshape(-1, width // window_size, window_size, window_size *  num_channels)
    windows = windows.permute(0, 2, 1, 3).contiguous().view(-1, height, width, num_channels)
    return windows

class ModSwin2SRSelfAttention(nn.Module):
    """
    Swin2SR self attention with modifications to reduce tensor ranks.
    """
    def __init__(self, self_attention, device):
        super().__init__()
        self.model = self_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        query_layer = (
            self.model.query(hidden_states)
            .view(batch_size, -1, self.model.num_attention_heads, self.model.attention_head_size)
            .transpose(1, 2)
        )
        key_layer = (
            self.model.key(hidden_states)
            .view(batch_size, -1, self.model.num_attention_heads, self.model.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.model.value(hidden_states)
            .view(batch_size, -1, self.model.num_attention_heads, self.model.attention_head_size)
            .transpose(1, 2)
        )

        query_layer = query_layer.reshape(batch_size * self.model.num_attention_heads, -1, self.model.attention_head_size)
        key_layer   = key_layer.reshape(batch_size * self.model.num_attention_heads, -1, self.model.attention_head_size)
        value_layer = value_layer.reshape(batch_size * self.model.num_attention_heads, -1, self.model.attention_head_size)

        # cosine attention
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)
        logit_scale = torch.clamp(self.model.logit_scale, max=math.log(1.0 / 0.01)).exp()

        logit_scale = logit_scale.expand(batch_size, self.model.num_attention_heads, 1, 1).reshape(batch_size * self.model.num_attention_heads, 1, 1)
        logit_scale = logit_scale.reshape(batch_size * self.model.num_attention_heads, 1, 1)

        attention_scores = attention_scores * logit_scale
        relative_position_bias_table = self.model.continuous_position_bias_mlp(self.model.relative_coords_table).view(
            -1, self.model.num_attention_heads
        )
        # [window_height*window_width,window_height*window_width,num_attention_heads]
        relative_position_bias = relative_position_bias_table[self.model.relative_position_index.view(-1)].view(
            self.model.window_size[0] * self.model.window_size[1], self.model.window_size[0] * self.model.window_size[1], -1
        )
        # [num_attention_heads,window_height*window_width,window_height*window_width]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        relative_position_bias = relative_position_bias.expand(batch_size,
                                                               self.model.num_attention_heads,
                                                               self.model.window_size[0] * self.model.window_size[1],
                                                               self.model.window_size[0] * self.model.window_size[1])
        relative_position_bias = relative_position_bias.reshape(batch_size * self.model.num_attention_heads,
                                                               self.model.window_size[0] * self.model.window_size[1],
                                                               self.model.window_size[0] * self.model.window_size[1])
        attention_scores = attention_scores + relative_position_bias

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swin2SRModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_mask = attention_mask.unsqueeze(1).expand(mask_shape,
                                                                self.model.num_attention_heads,
                                                                dim,
                                                                dim)
            attention_mask = attention_mask.reshape(mask_shape * self.model.num_attention_heads,
                                                    dim,
                                                    dim)
            attention_scores = attention_scores + attention_mask
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.model.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.reshape(batch_size, self.model.num_attention_heads, -1, self.model.attention_head_size)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.model.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class Swin2SRLayer(nn.Module):
    def __init__(self, layer, device):
        super().__init__()
        self.model = layer
        self.model.attention.self = ModSwin2SRSelfAttention(self.model.attention.self, device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None= False,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        # pad hidden_states to multiples of window size
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.model.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.model.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.model.shift_size, -self.model.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.model.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.model.window_size * self.model.window_size, channels)
        attn_mask = self.model.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.model.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.model.window_size, self.model.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.model.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.model.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.model.shift_size, self.model.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.model.layernorm_before(attention_windows)
        hidden_states = shortcut + self.model.drop_path(hidden_states)

        layer_output = self.model.intermediate(hidden_states)
        layer_output = self.model.output(layer_output)
        layer_output = hidden_states + self.model.drop_path(self.model.layernorm_after(layer_output))

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs

class Swin2SRModel(nn.Module):
    def __init__(
        self,
        swin2sr,
        device,
    ) -> None:
        super().__init__()
        self.model = swin2sr
        for i, stage in enumerate(self.model.swin2sr.encoder.stages):
            for j, layer in enumerate(stage.layers):
                self.model.swin2sr.encoder.stages[i].layers[j] = Swin2SRLayer(layer, device)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return self.model(pixel_values, output_attentions, output_hidden_states, return_dict, kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["npu", "gpu"], default="npu")
    args = parser.parse_args()

    model = Swin2SRModel(Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"), args.device)

    inputs = {"pixel_values": torch.rand((1, 3, 136, 136))}

    with torch.no_grad():
        torch.onnx.export(model, inputs, "swin2sr.onnx", opset_version=20, do_constant_folding=True,
                          dynamo = False, input_names = list(inputs.keys()))

    onnx_model = onnx.load("swin2sr.onnx")
    simplified_onnx_model, check = simplify(onnx_model)

    if check:
        onnx.save(simplified_onnx_model, "swin2sr.onnx", save_as_external_data=False)

if __name__ == "__main__":
    main()
