import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import repeat_kv, apply_rotary_pos_emb
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention as Gemma3Attention_original
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM as Gemma3ForCausalLM_original
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer as Gemma3DecoderLayer_original
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel as Gemma3TextModel_original
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector as Gemma3MultiModalProjector_original

from transformers.utils import logging

logger = logging.get_logger(__name__)


## ADAPTATION_1: we apply the rope separately to query and key states for on-target efficiency
def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    rope_real = rope_vals[0] # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1] # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:,:,:,:x.shape[-1]//2] # extract first half elements
    x_im = x[:,:,:,x.shape[-1]//2:] # extract second half elements

    x_prod_real = x_real*rope_real - x_im * rope_im
    x_prod_im = x_real*rope_im + x_im*rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real,x_prod_im),dim=3).view(*x.shape)
    return x


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        transposed_key_cache=False,
        **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # ADAPTATION_2: We send the transposed key cache to avoid the transpose inside matmul, it is expensive on target
    if transposed_key_cache:
        attn_weights = torch.matmul(query, key_states) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask #[:, :, :, : value_states.shape[-2]], value_states.shape[-2] is same as attention_mask length for static graph
        # The following section of code will only run when we want to trace the adapted model when creating the MPP model 
        # (we will pass KV$ for num_sequential_layers-1), otherwise the last dimension will have shape mismatch between the attn_weights (=ARN) and the attention_mask (=CL)
        if attention_mask.shape[-1] != value_states.shape[-2]:
            causal_mask = attention_mask[:, :, :, :value_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma3Attention(Gemma3Attention_original):
    def __init__(self, config, layer_idx: int):
        super(Gemma3Attention, self).__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        #QC
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config, 'return_new_key_value_only') else False
        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config, 'transposed_key_cache') else False

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        # query_states = _apply_rope_single(query_states, position_embeddings)
        key_states = _apply_rope_single(key_states, position_embeddings)

        if transposed_key_cache:
            key_states = key_states.transpose(2, 3)

        # ADAPTATION_3: We require to redefine the attention class since we need to pass additional cache_kwargs (specifically the transposed key cache and the return_new_key_value_only)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
                "transposed_key_cache": transposed_key_cache,
                "num_key_value_heads": self.config.num_key_value_heads,
                "return_new_key_value_only": return_new_key_value_only,
                "head_dim": self.head_dim
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        # This is moved after KV$ update to align ops between the prefill and generation graph.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self.q_norm(query_states)
        query_states = _apply_rope_single(query_states, position_embeddings)

        attention_interface: Callable = eager_attention_forward

        # Below code is commented because it adds a cast op between add and the input node due to which input encodings for attention_mask and swa_attention_mask are not generated.
        # if attention_mask is not None:
        #     # backwards compatibility
        #     attention_mask = attention_mask.to(query_states)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            transposed_key_cache=transposed_key_cache,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(Gemma3DecoderLayer_original):
    def __init__(self, config, layer_idx: int):
        super(Gemma3DecoderLayer, self).__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # ADAPTATION_4: the Gemma3DecoderLayer layer in the original fp/hf implementation will compute the sliding causal mask inside the model, which is removed since
        '''
        This is expensive for us
        1. we already compute the global layer causal mask outside the model as computing it on HTP is expensive
        2. We cannot infer the sliding causal mask from the global causal mask inside the model trivially, for multi-stream use cases.
        '''
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma3TextModel(Gemma3TextModel_original):
    def __init__(self, config):
        super(Gemma3TextModel, self).__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        swa_cache_position: Optional[torch.LongTensor] = None,
        swa_attention_mask: Optional[torch.Tensor] = None,
        swa_position_ids: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # ADAPTATION_5: We need to return the output from the model in dict format otehrwise jit trace cannot trace the outputs correctly and they are not seen in the onnx graph.
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            # batch_size, seq_len, _ = inputs_embeds.shape
            # past_key_values = HybridCache(
            #     self.config,
            #     max_batch_size=batch_size,
            #     max_cache_len=seq_len,
            #     dtype=inputs_embeds.dtype,
            # )
            past_key_values = DynamicCache()

        # ADAPTAION_6: We convert the past_key_values which flow into the model in the tuple format into the Cache object format. 
        # Even though we create a dynamic cache object here, the past_key_values tuple already contains the hybrid KV cache information i.e the sliding window layer have the KV cache of the size sliding_window.
        return_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            return_legacy_cache = True

            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

        if cache_position is None:
            # if cache_position (for global layer) is None, compute like HF does
            past_seen_tokens = past_key_values.get_seq_length(
                layer_idx=self.config.sliding_window_pattern - 1) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if swa_cache_position is None:
            # Ideally if the cache_position is None, we expect user to not pass index for both global and the local layers. [so both the ifs are either executed or not executed, for readability kept sep]
            # if swa_cache_position (for local layer) is None, compute like HF does, note not using the global layer idx here.
            layer_idx = 0 if self.config.sliding_window_pattern == self.config.num_hidden_layers else self.config.sliding_window_pattern
            past_seen_tokens = past_key_values.get_seq_length(layer_idx=layer_idx) if past_key_values is not None else 0
            swa_cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = torch.tensor([0], dtype=torch.int32)
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1]
                )
        # ADAPTATION_6: PRECOMPUTE OUTSIDE ALREADY.
        # causal_mask = self._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        #     output_attentions,
        # )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if isinstance(position_ids, (tuple, list)):  # QC
            position_embeddings_global = position_ids
            position_embeddings_local = swa_position_ids
        else:
            position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # Inferring the sliding causal mask from the global causal mask
        global_causal_mask = attention_mask

        # Inferring the sliding cache position from the global cache position
        global_cache_position = cache_position
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Choose appropriate attention mask and cache index based on attention type
            if decoder_layer.is_sliding:
                causal_mask = swa_attention_mask
                cache_position = swa_cache_position
            else:
                causal_mask = global_causal_mask
                cache_position = global_cache_position
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3ForCausalLM(Gemma3ForCausalLM_original):
    def __init__(self, config):
        super(Gemma3ForCausalLM, self).__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # ADAPTATION_7: This is needed to perform right padding where we need to send the cache index which signifies where will the new KV be scattered, right where the old KV ends.
        # we only take the cache index and not the whole tensor and internally broadcast as we do not want to introduce additional logic to check whether the tensor is contigous or not.
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference),
                                 persistent=False)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            cache_index: Optional[torch.Tensor] = None,
            swa_cache_index: Optional[torch.Tensor] = None,
            swa_attention_mask: Optional[torch.Tensor] = None,
            swa_position_ids: Optional[torch.Tensor] = None,
            **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)
        
        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcLlamaForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"

            # cache_position for global layer
            cache_position = cache_index + self.cache_tensor

        swa_cache_position = None
        # swa_cache_index determines the starting index where the new KV$ should be scattered for the local/ sliding window layer.
        if swa_cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcLlamaForCausal doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            # cache_position for local layer
            swa_cache_position = swa_cache_index + self.cache_tensor
        
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            swa_cache_position=swa_cache_position,
            swa_attention_mask=swa_attention_mask,
            swa_position_ids=swa_position_ids,
            **loss_kwargs,
        )

        if not isinstance(outputs, tuple):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Gemma3MultiModalProjector(Gemma3MultiModalProjector_original):
    def __init__(self, config):
        super(Gemma3MultiModalProjector, self).__init__(config)
        self.config = config

    # ADAPTATION: Replace MatMul in HF model with Conv2d layer
    def replace_matmul_with_conv(self):        
        self.mm_input_projection = nn.Conv2d(
            in_channels=self.config.vision_config.hidden_size,
            out_channels=self.config.text_config.hidden_size,
            kernel_size=1,
            bias=False
        )
        self.mm_input_projection.weight.data.copy_(self.mm_input_projection_weight.T.unsqueeze(-1).unsqueeze(-1))

        self.forward_orig = self.forward
        self.forward = self.forward_conv

    def forward_conv(self, vision_outputs: torch.Tensor):
        # Here vision_outputs comes from adapted siglip, so shape is (bsz, hid_size, n_patch, n_patch) instead of (bsz, n_patch^2, hid_size)
        batch_size, seq_length, patches_per_image, patches_per_image = vision_outputs.shape

        pooled_vision_outputs = self.avg_pool(vision_outputs)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs.permute(0,2,3,1)).permute(0,3,1,2)

        # ADAPTATION: Replace MatMul in HF model with Conv2d layer
        projected_vision_outputs = self.mm_input_projection(normed_vision_outputs)

        projected_vision_outputs = projected_vision_outputs.flatten(2).transpose(1, 2)

        return projected_vision_outputs


# ADAPTATION_8: We provide our implementation of the DYnamicCache KV cache update since we cannot rely on HF DynamicCache update as our keys are 
# transposed and we only want to return a sliver or new KV from the model for I/O bandwidth
def DynamicCache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += value_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    else:
        return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
        transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
        cache_position = cache_kwargs.get('cache_position')
        num_key_value_heads = cache_kwargs.get('num_key_value_heads')
        head_dim = cache_kwargs.get('head_dim')
        key_cat_dim = -1 if transposed_key_cache else -2

        # if the size of past key cache passed is smaller in value than the last position where the new kv is to be inserted
        # [in case when Cache position determined automatically by HF] (Ctx_len+ARN), then we want to perform concat and not do scattering.
        if self.value_cache[layer_idx].shape[-2] <= cache_position[-1]:
            key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=key_cat_dim)
            value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else:
            # the cache_position passed in as model i/p by user is a 1d tensor reflecting the positions
            # from valid_kv_end to valid_kv_end+ARN, we convert this into the indices for scattering. [# bsz, num_key_value_heads, head_dim, seq_len]-> works for transposed keys
            indices = cache_position.view(1, 1, 1, -1).expand(value_states.shape[0], num_key_value_heads, head_dim, cache_position.shape[-1])

            value_cache = self.value_cache[layer_idx].scatter(dim=-2, index=indices.transpose(-1,-2), src=value_states)

            indices = indices.transpose(-1, -2) if key_cat_dim== -2 else indices
            key_cache = self.key_cache[layer_idx].scatter(dim=key_cat_dim, index=indices, src=key_states)


        if return_new_key_value_only:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = value_cache
        return key_cache, value_cache


# ADAPTATION_9: as we work with transposed key, we cannot use it to return the seq_len, it will be incorrect, transposed key shape [bsz, n_heads, head_dim, seq_len]
def DynamicCache_get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(self.value_cache) <= layer_idx:
        return 0
    return self.value_cache[layer_idx].shape[-2]


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False


# These would be used if we don't comment the part where the update_causal_mask is invoked 
# inside the gemma3 model. when adding the adaptations in the example, uncomment the section and add below
orig_causal_mask = Gemma3TextModel_original._update_causal_mask

def adapted_update_causal_mask(self, attention_mask, *args, **kwargs):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    else:
        return orig_causal_mask(self, attention_mask, *args, **kwargs)


