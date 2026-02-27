#!/usr/bin/env python
# coding: utf-8

# # AIMET Quantization workflow for DeepSeek-R1 Distill Qwen2 7B
# 
# This notebook shows a working code example of how to use AIMET to quantize DeepSeek-R1 Distill Qwen2 7B models

# ---
# ### Required packages
# The notebook assumes AIMET and Qwen2 related packages are already installed.

# In[1]:

# Guard to prevent child processes from executing the main script
if __name__ != '__main__':
    import sys
    sys.exit(0)

# ---
# ### Configuration Loading System
# Supports loading configuration from JSON file with 3-tier priority:
# 1. JSON config file (if provided)
# 2. Environment variables
# 3. Default values

import json
import argparse

# Parse command-line arguments for optional config file
parser = argparse.ArgumentParser(
    description='DeepSeek-R1 Distill Qwen2 7B Quantization Script',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Configurable Variables (via JSON config file, environment variables, or defaults):

Feature Configuration:
  CONTEXT_LENGTH            Context length for the model (default: 4096)

Quantization Configuration:
  APPLY_SEQMSE              Apply SeqMSE optimization (default: True)
  APPLY_DECODER_LPBQ        Apply LPBQ to decoder (default: True)
  APPLY_LM_HEAD_LPBQ        Apply LPBQ to LM head (default: False)
  APPLY_CLIPPING            Apply activation clipping (default: False)
  EMBEDDING_TABLE_BITWIDTH  Embedding table bitwidth: 8 or 16 (default: 16)
  KEY_VALUE_BITWIDTH        Key-value cache bitwidth: 8 or 16 (default: 8)
  NUM_CALIBRATION_BATCHES   Number of calibration batches (default: 200)
  NUM_SEQMSE_BATCHES        Number of SeqMSE batches (default: 20)
  NUM_SEQMSE_CANDIDATES     Number of SeqMSE candidates (default: 20)

Speed Configuration:
  ENABLE_FP16               Enable FP16 flow (default: False)
  RUN_PPL_EVAL              Run perplexity evaluation (default: True)

AdaScale Configuration:
  ENABLE_ADASCALE           Enable AdaScale optimization (default: False)
  ADASCALE_ITERATIONS       AdaScale iterations (default: 1500)

QNN SDK Configuration:
  QNN_SDK_ROOT              Path to QNN SDK root directory (required - no default)
  LD_LIBRARY_PATH           Library path for QNN SDK (default: None)

NSP Target Configuration:
  TARGET_PLATFORM           Target platform: Windows/Android (default: Android)
  PLATFORM_GEN              Platform generation: 2/4/5 (default: 5)
  HTP_CONFIG_FILE           Path to HTP quantsim config file (default: hardcoded path)

Model Configuration:
  MODEL_NAME                Model name identifier (default: deepseek_r1_qwen2)
  MODEL_ID                  Model ID or path (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
  CACHE_DIR                 Cache directory path (default: ./cache_dir)
  OUTPUT_DIR                Output directory path (default: computed from model_id)
  ADASCALE_DIR              AdaScale directory path (default: computed from model_id)
  NUM_HIDDEN_LAYERS         Number of hidden layers, 0=use model default (default: 0)

Dataloader Configuration:
  C4_DATASET_PATH           Path to C4 dataset for AdaScale (default: None)
  BATCH_SIZE                Batch size for AdaScale (default: 2)
  PERCENT_DATASET_TO_LOAD   Percent of dataset to load (default: 3)
  NUM_SAMPLES               Number of samples for AdaScale (default: 1000)

ARN Configuration:
  ARN                       Auto-regression length (default: 2073)
  MASK_NEG                  Mask negative value (default: -3100)

Prepare Configuration:
  SKIP_PREPARE              Skip model preparation (default: False)

Test Vector Configuration:
  NUM_TEST_VECTORS          Number of test vectors to generate (default: 1)

Priority Order: JSON config > Environment variables > Default values

Example usage:
  python deepseek_r1_qwen2.py --config my_config.json
  python deepseek_r1_qwen2.py --help
''')
parser.add_argument('--config', type=str, default=None, 
                    help='Path to JSON configuration file')
args, unknown = parser.parse_known_args()

# Load JSON config if provided
json_config = {}
if args.config:
    try:
        with open(args.config, 'r') as f:
            json_config = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Warning: Config file not found: {args.config}")
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config file: {e}")

def get_config_value(key, default, value_type='str'):
    """
    Get configuration value with 3-tier priority:
    1. JSON config file
    2. Environment variable
    3. Default value
    
    Args:
        key: Configuration key name
        default: Default value if not found in config or environment
        value_type: Type of value ('str', 'int', 'bool', 'none')
    
    Returns:
        Configuration value with appropriate type
    """
    import os
    
    # Priority 1: Check JSON config
    if key in json_config:
        value = json_config[key]
        if value_type == 'bool':
            if isinstance(value, bool):
                return value
            # Handle string boolean values from JSON
            return str(value).lower() in ('true', '1', 't', 'yes')
        elif value_type == 'int':
            return int(value)
        elif value_type == 'none':
            return value  # Can be None or a value
        else:  # str
            return str(value) if value is not None else None
    
    # Priority 2: Check environment variable
    env_value = os.getenv(key)
    if env_value is not None:
        if value_type == 'bool':
            return env_value.lower() in ('true', '1', 't')
        elif value_type == 'int':
            return int(env_value)
        elif value_type == 'none':
            return env_value
        else:  # str
            return env_value
    
    # Priority 3: Use default value
    return default


# ### Overall flow
# This notebook covers the following
# 1. Parametrizing the Environment
# 2. Instantiate and evaluate FP32 HuggingFace model
# 3. Optionally quantize model using AIMET AdaScale
# 4. Optionally export and evaluate AdaScale model
# 3. Instantiate and adapt FP32 HuggingFace or AdaScale model
# 4. Model Sample Input
# 5. Prepare model using QAIRT model preparer
# 6. Evaluation of prepared base model
# 7. Quantization
# 8. Exporting base model onnx, encodings and test vectors
# 
# ### What this notebook is not 
# * This notebook is not intended to show the full scope of optimization. For example, the flow will not use QAT, KD-QAT as deliberate choice to have the notebook execute more quickly.

# ## 1. Parametrizing the Environment

# ---
# ### 1.1 Notebook Configs

# In[2]:

print("=" * 80)
print("1.1 Notebook Config")
print("=" * 80)

import os

# Feature knobs
context_length = get_config_value("CONTEXT_LENGTH", 4096, 'int')

# Quantization knobs
apply_seqmse = get_config_value("APPLY_SEQMSE", True, 'bool')
apply_decoder_lpbq = get_config_value("APPLY_DECODER_LPBQ", True, 'bool')
apply_lm_head_lpbq = get_config_value("APPLY_LM_HEAD_LPBQ", False, 'bool')
apply_clipping = get_config_value("APPLY_CLIPPING", False, 'bool')
embedding_table_bitwidth = get_config_value("EMBEDDING_TABLE_BITWIDTH", 16, 'int')
key_value_bitwidth = get_config_value("KEY_VALUE_BITWIDTH", 8, 'int')
num_calibration_batches = get_config_value('NUM_CALIBRATION_BATCHES', 200, 'int')
num_seqmse_batches = get_config_value('NUM_SEQMSE_BATCHES', 20, 'int')
num_seqmse_candidates = get_config_value('NUM_SEQMSE_CANDIDATES', 20, 'int')

# Speed knobs
enable_fp16 = get_config_value("ENABLE_FP16", False, 'bool')
run_ppl_eval = get_config_value("RUN_PPL_EVAL", True, 'bool')

# Adascale knobs
enable_adascale = get_config_value("ENABLE_ADASCALE", False, 'bool')
adascale_iterations = get_config_value("ADASCALE_ITERATIONS", 1500, 'int')


# In[3]:


assert not (apply_decoder_lpbq and apply_lm_head_lpbq), "Applying LPBQ to both Decoder and LM-Head has not been validated for accuracy"
assert embedding_table_bitwidth in (8, 16), "Only 8-bit and 16-bit Emebdding Table have been validated"
assert not enable_fp16, "FP16 based quantization has not been tested"
assert key_value_bitwidth in (8, 16), "Only 8-bit and 16-bit Key-Value Cache have been validated"


# ---
# ### 1.2 Setting QNN SDK

# In[4]:

print("=" * 80)
print("1.2 Setting QNN SDK")
print("=" * 80)

import sys

sys.path.append("../")
sys.path.append("../../")


# In[5]:


QNN_SDK_ROOT = get_config_value('QNN_SDK_ROOT', None, 'none')
assert QNN_SDK_ROOT is not None, 'Please point the QNN_SDK_ROOT variable to your QNN SDK'
assert os.path.exists(QNN_SDK_ROOT), "QNN_SDK_ROOT doesn't exist!"
QNN_SDK_ROOT=str(QNN_SDK_ROOT)
lib_clang_path = os.path.join(QNN_SDK_ROOT, 'lib', 'x86_64-linux-clang')
sys.path.insert(0, QNN_SDK_ROOT + '/lib/python')
LD_LIBRARY_PATH = get_config_value('LD_LIBRARY_PATH', None, 'none')
os.environ['LD_LIBRARY_PATH'] = lib_clang_path + ':' + LD_LIBRARY_PATH if LD_LIBRARY_PATH is not None else lib_clang_path


# ---
# ### 1.3 Setting NSP Target

# In[6]:

print("=" * 80)
print("1.3 Setting NSP Target")
print("=" * 80)

sys.path.append("../../")
from common.utilities.nsptargets import NspTargets

# setup Target platform and its generation
TARGET_PLATFORM = get_config_value("TARGET_PLATFORM", "Android", 'str').capitalize()

# Android GEN4 and GEN5 is supported for this notebook
PLATFORM_GEN = get_config_value("PLATFORM_GEN", 5, 'int')

nsp_target = eval(f"NspTargets.{TARGET_PLATFORM}.GEN{PLATFORM_GEN}")

# Select quantsim config based on target
htp_config_file = get_config_value("HTP_CONFIG_FILE", "/home/jovyan/new_nb_venv/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_v73.json", 'str')


# ---
# ### 1.4 Validate QNN SDK and AIMET installed versions

# In[7]:


#from common.utilities.version_checker import get_sdk_version, validate_sdk_version, get_supported_aimet_version, get_installed_aimet_version, validate_aimet_torch_version

#assert validate_sdk_version(QNN_SDK_ROOT, notebook_config.supported_sdk_version), f"WARNING: Found SDK version {get_sdk_version(QNN_SDK_ROOT)} installed in current environment. However, the notebooks are verified to work with QNN SDK version {notebook_config.supported_sdk_version}."
#assert validate_aimet_torch_version(os.getcwd()), f"WARNING: Found AIMET version {get_installed_aimet_version()} installed in current environment. However, the notebooks are verified to work with AIMET version {get_supported_aimet_version(os.getcwd())}."


# ---
# ## 2. Instantiate and evaluate HuggingFace model

# In[8]:

print("=" * 80)
print("2. Instantiate and evaluate HuggingFace model")
print("=" * 80)

import torch
from transformers.models.qwen2 import modeling_qwen2
from aimet_torch.utils import place_model, change_tensor_device_placement
from genai_lib.common.debug.profiler import event_marker

model_name = get_config_value("MODEL_NAME", 'deepseek_r1_qwen2', 'str')  # Only DeepSeek-R1-Distill-Qwen-7B have been validated in this Notebook

model_id = get_config_value("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 'str')

cache_dir = get_config_value("CACHE_DIR", './cache_dir', 'str')

output_dir = get_config_value("OUTPUT_DIR", f"./output_dir_{os.path.basename(model_id)}", 'str')

adascale_dir = get_config_value("ADASCALE_DIR", "./adascale_dir_" + os.path.basename(model_id), 'str')

os.makedirs(output_dir, exist_ok=True)

if enable_adascale:
    os.makedirs(adascale_dir, exist_ok=True)


# In[9]:


# Note: This cell (and the corresponding cells with Recipe_logger tag) can be removed after dumping and verifying the recipe without
# impacting notebook functionality
from genai_lib.common.debug.recipe_logger import recipe_dump_init
from genai_lib.common.debug.recipe_logger import llm_lib_log_env_info

# Recipe_logger: Initialize the logger and log environment details
recipe_dump_init(output_dir, "genai_lib_debug")

llm_lib_log_env_info()


# ---
# ### 2.1 Configurable setting by users and loading HF model

# In[10]:

print("=" * 80)
print("2.1 Configurable setting by users")
print("=" * 80)

from transformers import AutoConfig, AutoTokenizer
llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
num_hidden_layers = get_config_value("NUM_HIDDEN_LAYERS", 0, 'int')
llm_config.num_hidden_layers = num_hidden_layers if num_hidden_layers > 0 else llm_config.num_hidden_layers
print(f'num_layer: {llm_config.num_hidden_layers}, context_length: {context_length}, '
      f'num_hidden_size: {llm_config.num_attention_heads}, num_kv_heads: {llm_config.num_key_value_heads}')

with event_marker('HuggingFace FP model creation'):
    model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(model_id, config=llm_config)

    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
    # Adjust the tokenizer to limit to context_length
    tokenizer.model_max_length = context_length

# Reduce the precision of the model to FP16 to minimize the amount of GPU memory needed
if enable_fp16:
    model.half()


# ---
# ### 2.2 Instantiate Dataloaders

# In[11]:

print("=" * 80)
print("2.2 Instantiate Dataloaders")
print("=" * 80)

from llm_utils.wikitext_dataloader import get_wiki_dataset
from llm_utils.generic_dataloader import get_local_dataset

with event_marker("Instantiate dataloaders"):
    wikitext_train_dataloader, wikitext_test_dataloader, _ = get_wiki_dataset(context_length, tokenizer, cache_dir)

if enable_adascale:
    with event_marker("Instantiate adascale dataloaders"):
        adascale_train_dataloader, _ = get_local_dataset(context_length, tokenizer, json_path = get_config_value("C4_DATASET_PATH", None, 'none'), key = "input_ids", 
                                                         batch_size = get_config_value("BATCH_SIZE", 2, 'int'), 
                                                         percent_dataset_to_load = get_config_value("PERCENT_DATASET_TO_LOAD", 3, 'int'), 
                                                         num_samples = get_config_value("NUM_SAMPLES", 1000, 'int'))


# ---
# ### 2.3 HuggingFace FP model eval

# In[12]:

print("=" * 80)
print("2.3 HuggingFace FP model eval")
print("=" * 80)
    

from genai_lib.llm.evaluation_utils import llm_evaluate_ppl_with_dataloader

if run_ppl_eval:
    with event_marker("HuggingFace FP model eval"):
        with place_model(model, torch.device('cuda')):
            orig_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader)

    print(f"PPL score of HuggingFace FP model = {orig_ppl}")


# In[13]:


from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property
from genai_lib.common.debug.recipe_logger import llm_lib_log_metric, ModelType, Metric

# Recipe_logger: Log the context_length property and the metrics.
llm_lib_log_property({Property.context_length : context_length})

if run_ppl_eval:
    llm_lib_log_metric(ModelType.hf_model, Metric.ppl, orig_ppl, model_name="base")


# ---
# ## 3. AdaScale
# 
# This section modifies the model's weights to perform better with quantization.

# #### 3.1 Redefine forward for JIT tracing in Quantsim Creation

# In[14]:

print("=" * 80)
print("3. AdaScale")
print("=" * 80)

import torch
from transformers import DynamicCache
import types

# AIMET requires KV Cache to be of type Tuple during the forward pass, so we wrap the forward to convert the KV Cache during inference
def custom_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, *args, **kwargs):
    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    lm_logits, new_past_key_values = self.__original_forward__(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        num_logits_to_return=0,
        return_dict=False,
        *args,
        **kwargs,
    )
    return lm_logits, new_past_key_values.to_legacy_cache()

if enable_adascale:
    # Save original forward method
    model.__original_forward__ = model.forward
    # Replace with custom forward
    model.forward = types.MethodType(custom_forward, model)


# #### 3.2 Create quantsim configured for QNN HTP target 

# In[15]:

print("=" * 80)
print("3.2 Create quantsim configured for QNN HTP target")
print("=" * 80)

from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel

if enable_adascale:
    dummy_input = torch.randint(0, 1, (1, 1, context_length), device="cuda")

    with event_marker("create Adascale Quantsim"):
        with place_model(model, "cuda"):
            model.config.return_dict=False
            quantsim = QuantizationSimModel(model=model,
                                            quant_scheme=QuantScheme.post_training_tf,
                                            default_output_bw=16,
                                            default_param_bw=4,
                                            in_place=True,
                                            dummy_input=tuple(list(dummy_input)),
                                            config_file=htp_config_file)


# In[16]:


from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.nn.modules import custom as aimet_ops

if enable_adascale:
    propagate_output_encodings(quantsim, aimet_ops.Concat)


# #### 3.3 Enable per channel quantization
# 
# Configure linear quantizers to support per channel quantization to mimic Convolution layer behavior on-device.

# In[17]:

print("=" * 80)
print("3.3 Enable per channel quantization")
print("=" * 80)

from aimet_torch.v2.nn.true_quant import QuantizedLinear
from aimet_torch.v2.quantization.affine import QuantizeDequantize

if enable_adascale:
    for name, qmodule in quantsim.named_qmodules():
        if isinstance(qmodule, QuantizedLinear):
            assert (len(qmodule.weight.shape)) == 2, f"Per channel quantization for linear weights is only supported for 2d weights, got shape: {qmodule.weight.shape} instead"
            qmodule.param_quantizers["weight"] = QuantizeDequantize(shape=(qmodule.weight.shape[0], 1), 
                                                                    bitwidth=qmodule.param_quantizers["weight"].bitwidth, 
                                                                    symmetric=qmodule.param_quantizers["weight"].symmetric).to(next(quantsim.model.parameters()).device)


# #### 3.4 Manual mixed prescision + Disable un-needed quantizers
# 
# Adascale only operates on the decoder blocks of the model, therefore we can disable quantizers on non decoder blocks.

# In[18]:

print("=" * 80)
print("3.4 Manual mixed prescision + Disable un-needed quantizers")
print("=" * 80)

import re

if enable_adascale:
    # Remove quantizers for non decoder blocks
    quantsim.model.model.embed_tokens.param_quantizers["weight"] = None
    quantsim.model.lm_head.param_quantizers["weight"] = None

    # Increase bitwidth for rmsnorm due to the op having higher quantization sensitivity
    for name, qmodule in quantsim.named_qmodules():
        if re.search(r'rmsnorm', qmodule.__class__.__name__.lower()):
            qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False).to(next(quantsim.model.parameters()).device)


# #### 3.5 AdaScale
# 
# Appplying AdaScale to optimize weights for quantization

# In[19]:

print("=" * 80)
print("3.5 Appplying AdaScale to optimize weights")
print("=" * 80)

from aimet_torch.experimental.adascale import apply_adascale

if enable_adascale:
    with event_marker("apply AdaScale", flush_ram=True):
        with place_model(quantsim.model, "cuda"):
            apply_adascale(qsim=quantsim,
                        data_loader=adascale_train_dataloader,
                        forward_fn=custom_forward,
                        num_iterations=adascale_iterations)


# #### 3.6 Evaluate the AdaScale model

# In[20]:

print("=" * 80)
print("3.6 Evaluate the AdaScale model")
print("=" * 80)

from aimet_torch.v2.utils import remove_activation_quantizers

if enable_adascale and run_ppl_eval:
    with event_marker("AdaScale FP model eval"):
        with place_model(quantsim.model, torch.device('cuda')), remove_activation_quantizers(quantsim.model):
            adascaled_ppl = llm_evaluate_ppl_with_dataloader(model=quantsim.model, dataloader=wikitext_test_dataloader)
    print(f"PPL score of AdaScale model = {adascaled_ppl}")


# #### 3.7 Export model
# 
# This exports new model weights that are more quantization friendly, and stores them in the specified adascale directory. If you want to use these new weights for a quantization pipeline notebook, you should pass in the adascale directory path as the Model ID in the quant notebook.

# In[21]:

print("=" * 80)
print("3.7 Export model")
print("=" * 80)

if enable_adascale:
    with event_marker("Save AdaScale model", flush_ram=True):
        fp_ada_model = QuantizationSimModel.get_original_model(quantsim.model, qdq_weights = True)
        fp_ada_model.save_pretrained(adascale_dir)

    tokenizer_dir = os.path.join(adascale_dir, 'tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    # update model_id to the exported adascale model directory
    model_id = adascale_dir
    
    del model
    del quantsim


# ---
# ### 4.1 Adapt FP32 model definition for inference on HTP.
# - The following adaptations are done to replace default attention module with attention definition that compatible with NSP backend
#   * use conv instead of linear for Q,K,V,O projections
#   * bypass attention and causal mask generation and replace with pre-generated 2D-mask input
#   * output only newly created V and transposed K instead of entire augmented KV sequence
#   * input pre-calculated positional embedding instead of position ids, thus bypass the embedding generation in the model

# In[22]:

print("=" * 80)
print("4.1 Adapt FP32 model definition for inference on HTP")
print("=" * 80)

from transformers.models.qwen2 import modeling_qwen2
from transformers import cache_utils

from genai_lib.llm.dev.model_adaptation.qwen2.adaptation import (
    QcQwen2Attention,
    QcQwen2ForCausalLM,
    adapted_update_causal_mask,
    adapted_RotaryEmbedding,
    DynamicCache_update,
    DynamicCache_get_seq_length,
    update_attr,
    DynamicCache_to_legacy_cache,
)

with event_marker("FP model adaptation configuration"):
    modeling_qwen2.Qwen2Attention = QcQwen2Attention
    modeling_qwen2.Qwen2ForCausalLM = QcQwen2ForCausalLM

    # Bypass attention_mask preparation
    assert hasattr(modeling_qwen2.Qwen2Model, '_update_causal_mask'), \
    "Qwen2Model does not have _update_causal_mask as attribute"
    modeling_qwen2.Qwen2Model._update_causal_mask = adapted_update_causal_mask

    # Bypass rotary_emb module
    assert hasattr(modeling_qwen2.Qwen2RotaryEmbedding, 'forward'), \
    f"Unknown Qwen2RotaryEmbedding definition: {modeling_qwen2.Qwen2RotaryEmbedding}"
    modeling_qwen2.Qwen2RotaryEmbedding.forward = adapted_RotaryEmbedding

    # Adapting KV$ management
    assert update_attr(cache_utils.DynamicCache, 'update', DynamicCache_update), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'get_seq_length', DynamicCache_get_seq_length),  f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
    assert update_attr(cache_utils.DynamicCache, 'to_legacy_cache', DynamicCache_to_legacy_cache), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"


# ---
# ### 4.2 Instantiate adapted FP32 model definition

# In[23]:

print("=" * 80)
print("4.2 Instantiate adapted FP32 model definition")
print("=" * 80)

#======================Fixed setting that should not be changed by users==============
# Auto-regression length: number of tokens to consume and number of logits to produce.
# This value should NOT be changed due to downstream consumption requirements
ARN = get_config_value("ARN", 2073, 'int')
MASK_NEG = get_config_value("MASK_NEG", -3100, 'int')
setattr(llm_config, 'return_new_key_value_only', True)
setattr(llm_config, 'transposed_key_cache', True)
setattr(llm_config, 'use_combined_mask_input', True)
setattr(llm_config, 'use_position_embedding_input', True)
setattr(llm_config, '_attn_implementation', 'eager')
setattr(llm_config, '_attn_implementation_internal', 'eager')
setattr(llm_config, 'return_dict', False)
setattr(llm_config, 'logits_to_keep', 0)
setattr(llm_config, 'enable_r3_hadamard', True)


# In[24]:


from genai_lib.common.debug.recipe_logger import llm_lib_log_property, Property

# Recipe_logger: Log the ARN of the prepared model
llm_lib_log_property({Property.ARN : ARN})


# In[25]:


with event_marker('Adapted FP model creation'):
    model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(model_id, config=llm_config)
    # Must initialize R3 hadamard after `.from_pretrained()`, since those weights are not in the state_dict
    for name, module in model.named_modules():
        if hasattr(module, "initialize_r3_hadamard"):
            module.initialize_r3_hadamard()


# ---
# ### 4.3 Changes to HuggingFace model to work with the Adapted Model or Prepared Model
# - As a result of adapting the model we introduce changes to the types of the model inputs.
# - As a result of model preparation, we make the shapes of the inputs static.
# - adapted_model_forward works with either adapted model dynamic input or prepared model static input model through flag static_shape.
# - Override the 'forward' function and the function 'prepare_inputs_for_generation'. With these overrides, we make the adapted model or prepared model work just like the old model.
# - adapted_model_prepare_inputs_for_dynamic_shapes is utility function for forward pass of adapted model with dynamic shapes.
# - adapted_model_prepare_inputs_for_static_shapes is utility function for forward pass of prepared model with static shapes.

# In[26]:

print("=" * 80)
print("4.3 Changes to HuggingFace model to work with the Adapted Model")
print("=" * 80)

from genai_lib.llm.static_graph_utils import llm_slice_inputs_for_inference, llm_pad_inputs, llm_create_1d_attn_mask, llm_pad_past_kv, \
    llm_trim_pad_logits, llm_get_position_ids_from_attention_mask, llm_pad_input_attn_mask, llm_create_kv_attn_mask, llm_get_dummy_kv
from genai_lib.llm.dev.model_adaptation.qwen2.utils import llm_update_causal_mask, llm_create_position_embeddings
from genai_lib.llm.dev.model_adaptation.common.utils import KEY_CONCAT_AXIS, llm_update_kv_cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import types
import inspect

def prepare_inputs_for_static_shape(model, input_ids_slice, attn_mask_slice, outputs, **kwargs):
    batch_size = input_ids_slice.shape[0]
    pad_token = tokenizer.eos_token_id
    head_dim = model.config.head_dim if hasattr(model.config, 'head_dim') else model.config.hidden_size // model.config.num_attention_heads

    ####### input id preparation #######
    pad_input_ids = llm_pad_inputs(pad_token = pad_token,
                                   max_input_tokens = ARN,
                                   input_ids_slice = input_ids_slice)

    ####### KV input preparation #######
    #TODO: add support for taking in the dtype when creating the tensors
    dummy_kv = llm_get_dummy_kv(batch_size = batch_size,
                                model_context_len = context_length,
                                max_input_tokens = ARN,
                                num_key_value_heads = model.config.num_key_value_heads,
                                head_dim = head_dim,
                                key_concat_axis = KEY_CONCAT_AXIS,
                                device = model.device)

    padded_past_kv_in = llm_pad_past_kv(dummy_past_kv = dummy_kv,
                                        unpadded_past_kv = outputs['past_key_values'],
                                        num_hidden_layers = model.config.num_hidden_layers,
                                        key_concat_axis = KEY_CONCAT_AXIS)


    ######### Attention mask Input preparation #######
    inp_attn_mask = llm_pad_input_attn_mask(attn_mask_slice = attn_mask_slice,
                                            max_input_tokens = ARN)
    past_kv_attn_mask = llm_create_kv_attn_mask(unpadded_past_kv = outputs['past_key_values'],
                                                model_context_len = context_length,
                                                max_input_tokens = ARN,
                                                batch_size = batch_size,
                                                device = model.device)
    prepared_1d_attention_mask = llm_create_1d_attn_mask(attn_mask_past_kv = past_kv_attn_mask,
                                                         attn_mask_input = inp_attn_mask)

    # due to model adaptation
    prepared_causal_mask = llm_update_causal_mask(prepared_1d_attn_mask = prepared_1d_attention_mask,
                                                  input_tensor=  pad_input_ids,
                                                  max_input_tokens = ARN,
                                                  model_context_len = context_length,
                                                  model_id_or_path = model_id,
                                                  mask_neg = MASK_NEG)

    ########### Position ID preparation #######
    position_ids = llm_get_position_ids_from_attention_mask(attention_mask = prepared_1d_attention_mask,
                                                            max_input_tokens = ARN,
                                                            model_context_len = context_length)

    # model adaptation
    prepared_position_embeddings = llm_create_position_embeddings(config = model.config,
                                                                  position_ids = position_ids)

    prepared_inputs = {
        'input_ids': pad_input_ids,
        'attention_mask': prepared_causal_mask,
        'position_ids': prepared_position_embeddings,
        'past_key_values': padded_past_kv_in,
    }
    return prepared_inputs


def prepare_inputs_for_dynamic_shape(model, input_ids_slice, attn_mask_slice, outputs, **kwargs):
    batch_size = input_ids_slice.shape[0]
    pad_token = tokenizer.eos_token_id
    head_dim = model.config.head_dim if hasattr(model.config, 'head_dim') else model.config.hidden_size // model.config.num_attention_heads
    if attn_mask_slice is None:
        attn_mask_slice = torch.ones((input_ids_slice.shape[0], input_ids_slice.shape[1]), dtype = torch.long, device = model.device)
    position_ids = torch.cumsum(attn_mask_slice, dim=1) - 1
    prepared_position_embeddings = llm_create_position_embeddings(config = model.config,
                                                                  position_ids = position_ids)

    prepared_inputs = {
        'input_ids': input_ids_slice,
        'attention_mask': attn_mask_slice,
        'position_ids': prepared_position_embeddings,
        'past_key_values': outputs['past_key_values'],
    }
    return prepared_inputs


# In[27]:


# Redefinition of the forward function to work with model I/O adaptations and static shapes of the tensors that the model consumes as input
def adapted_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    return_dict=False,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    cache_position=None,
    **kwargs
):
    static_shape = hasattr(self, 'num_logits_to_return')

    # create the generator which slices input into chunks of AR (and pads if necessary)
    slice_inputs_gen_obj = llm_slice_inputs_for_inference(max_input_tokens = ARN if static_shape else input_ids.shape[-1],
                                                          model_context_len = context_length,
                                                          input_ids = input_ids)
    num_slices = kwargs.get('num_slices', None)

    # dictionary to store the running output which contains the logits and the useful past kv cache until that execution
    outputs = {}
    outputs['past_key_values'] = past_key_values
    for i, inputs in enumerate(slice_inputs_gen_obj):
        if num_slices is not None and i >= num_slices:
            break
        input_ids_slice = inputs['input_ids_slice']
        attn_mask_slice = inputs['attn_mask_slice']

        if static_shape:
            prepared_inputs = prepare_inputs_for_static_shape(self, input_ids_slice=input_ids_slice, attn_mask_slice=attn_mask_slice, outputs=outputs)
        else:
            prepared_inputs = prepare_inputs_for_dynamic_shape(self, input_ids_slice=input_ids_slice, attn_mask_slice=attn_mask_slice, outputs=outputs)

        cur_outputs = self.model(**prepared_inputs)
        if not static_shape:
            cur_outputs = (self.lm_head(cur_outputs[0]),) + cur_outputs[1:]

        # avoided creating a new tuple of current_key_value to avoid the memory spike, sending slice
        outputs['past_key_values'] = llm_update_kv_cache(unpadded_past_kv = outputs['past_key_values'],
                                                         current_key_values= cur_outputs[-1],
                                                         input_ids_slice = input_ids_slice)

        lm_logits = llm_trim_pad_logits(cur_logits = cur_outputs[0],
                                         input_ids_slice=input_ids_slice)

        bsz, _, dim = lm_logits.shape

        outputs['logits'] = torch.cat(
                (outputs.get('logits', torch.zeros((bsz, 0, dim), device=lm_logits.device)), lm_logits),
                dim=1)

    if return_dict:
        return CausalLMOutputWithPast(
            loss=outputs.get('loss', None),
            logits=outputs.get('logits', None),
            past_key_values=outputs.get('past_key_values', None),
            hidden_states=None,
            attentions=None,
        )
    return (outputs['logits'], outputs['past_key_values'])


# ---
# ### 3.4 Complete the last step(s) of Model Adaptation
# The following model adaptation are enabled for inference:
# - apply linear to conv in attention, MLP and lmhead and arrange linear weights properly for conv

# In[28]:

print("=" * 80)
print("4.4 Complete the last step(s) of Model Adaptation")
print("=" * 80)

from genai_lib.common.dev.model_adaptation.linear_to_conv import replace_linears_with_convs

with event_marker('FP model adaptation for NSP backend completion'):
    model = replace_linears_with_convs(model)

if run_ppl_eval:
    model.forward = types.MethodType(adapted_model_forward, model)
    with event_marker("Adapted FP model eval"):
        with place_model(model, torch.device('cuda')):
            adapted_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader)
    print(f"PPL score of adapted model = {adapted_ppl}")
    model.forward = types.MethodType(QcQwen2ForCausalLM.forward, model)


# In[29]:


if run_ppl_eval:
    llm_lib_log_metric(ModelType.adapted_model, Metric.ppl, adapted_ppl, model_name="base")


# ---
# ## 4.4 Model Sample Input

# In[30]:

print("=" * 80)
print("4.4 Model Sample Input")
print("=" * 80)

from genai_lib.llm.static_graph_utils import llm_get_dummy_kv
from genai_lib.llm.dev.model_adaptation.common.utils import KEY_CONCAT_AXIS

def get_dummy_data(device="cuda", dtype=torch.float32, return_dict=False):
    input_ids = torch.randint(0, len(tokenizer), (1, ARN), device=device)
    attn_mask = torch.ones((1, ARN), device=device)
    dummy_kv = llm_get_dummy_kv(batch_size = 1,
                                max_input_tokens = ARN,
                                model_context_len = context_length,
                                num_key_value_heads = llm_config.num_key_value_heads,
                                head_dim = llm_config.hidden_size // llm_config.num_attention_heads,
                                key_concat_axis = KEY_CONCAT_AXIS,
                                device = device)
    past_kv = tuple(dummy_kv for _ in range(llm_config.num_hidden_layers))
    outputs = {'past_key_values': past_kv}

    dummy_input = prepare_inputs_for_static_shape(model.to(device), input_ids, attn_mask, outputs)

    if return_dict:
        return dummy_input
    return tuple(list(dummy_input.values()))


# ---
# ## 5. Prepare model using QAIRT model preparer

print("=" * 80)
print("5. Prepare model using QAIRT model preparer")
print("=" * 80)

# ---
# ### 5.1 KVCache MHA model preparation

# In[31]:

print("=" * 80)
print("5.1 KVCache MHA model preparation")
print("=" * 80)

# MPP uses same torch to onnx export as AIMET torch.
# So for LLMs, please set EXPORT_TO_ONNX_DIRECT to true, the same way as it is done in AIMET.
from qti.aisw.emitter.utils import onnx_saver
onnx_saver.EXPORT_TO_ONNX_DIRECT = True


# In[32]:


import time
from qti.aisw.emitter.utils.torch_utils import load_torch_model_using_safetensors
from genai_lib.llm.model_preparation_utils import llm_build_preparer_converter_args
from genai_lib.llm.utils import llm_model_input_output_names
from qti.aisw.preparer_api.model_preparer import prepare_model

# Configuring the model for KVCache mode
model.num_logits_to_return = ARN


prepare_path = os.path.join(output_dir, 'prepare')
os.makedirs(prepare_path, exist_ok=True)
prepare_filename = f'{model_name}_kvcache_{llm_config.num_hidden_layers}_layer'

skip_prepare = get_config_value("SKIP_PREPARE", False, 'bool')
if skip_prepare:
    with event_marker(f"KVCache load pre-prepared {prepare_filename}", flush_ram=True):
        prepared_model_path = os.path.join(prepare_path, f'{prepare_filename}.py')
        if not os.path.exists(prepared_model_path):
            raise ValueError(f"prepared artifacts not found in {prepare_path}")
        else:
            print(f'WARNING: preparation skipped for model={prepare_filename}, prepared at {time.ctime(os.path.getmtime(prepared_model_path))}')
            prepared_model = load_torch_model_using_safetensors(path=prepare_path, 
                                                                filename=prepare_filename,
                                                                model_name=prepare_filename)

else:
    dummy_input = get_dummy_data(device=model.model.device, dtype=next(model.parameters()).dtype, return_dict=True)
    dummy_input = tuple(dummy_input.values())
    input_names, output_names = llm_model_input_output_names(llm_config.num_hidden_layers)
    converter_args = llm_build_preparer_converter_args(llm_config.num_hidden_layers, input_names, use_qairt_mpp=True) # Build converter args

    with event_marker("KVCache prepare model", flush_ram=True):
    
        if __name__ == '__main__': # We use the main guard to prevent child processes from re-running the top-level code
            prepared_model = prepare_model(model,
                                           dummy_input,
                                           model_name=prepare_filename,
                                           filename=prepare_filename,
                                           path=prepare_path,
                                           input_names=input_names,
                                           output_names=output_names,
                                           onnx_export_args={"opset_version":17},
                                           converter_args=converter_args,
                                           keep_original_model_structure=False, # Flatten the model to enable weight-sharing by setting `keep_original_model_structure = False
                                           order_inputs = True,
                                           order_outputs = True,
                                           skipped_optimizers=['eliminate_common_subexpression',
                                                               'eliminate_nop_with_unit', 
                                                               'eliminate_duplicate_initializer'
                                                              ],
                                           return_prepare_model=True
                                           )


# ---
# ## 6. Evaluation of prepared model
# Verify if prepared KV cache model generates the same PPL as FP model.

print("=" * 80)
print(" 6. Evaluation of prepared model")
print("=" * 80)

# ---
# ### 6.1 Changes to HuggingFace model to work with the prepared model
# 
# Replace the model inside the HuggingFace model with the prepared model.
# Note that the prepared model already fuses model.model and model.lm_head 
# into one, so here we simply set model.lm_head to None

# In[33]:

print("=" * 80)
print("6.1 Changes to HuggingFace model to work with the prepared model")
print("=" * 80)

del model.model
del model.lm_head

model.model = None
model.lm_head = None

model.forward = types.MethodType(adapted_model_forward, model)


# ---
# ### 6.2 Convert the model to half precision

# In[34]:

print("=" * 80)
print("6.2 Convert the model to half precision")
print("=" * 80)

if enable_fp16:
    torch.set_default_dtype(torch.float16)
    model.half()


# ---
# ### 6.3 Evaluation of perplexity score using prepared model

# In[35]:

print("=" * 80)
print("6.3 Evaluation of perplexity score using prepared model")
print("=" * 80)

if run_ppl_eval:
    with event_marker("KVcache prepared FP eval", flush_ram=True):
        with place_model(prepared_model, torch.device("cuda")):
            model.model = prepared_model
            prepared_kvcache_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader)

    # This should be very close (<1e-4 delta) to original model's perplexity
    # If the perplexity score goes further up, it indicates the AIMET/QNN pair is producing a faulty prepared model
    print(f"ppl score of KVCACHE prepared fp model = {prepared_kvcache_ppl}")
    print(f"Diff between HF orig ppl and prepared ppl = {orig_ppl - prepared_kvcache_ppl}")


# In[36]:


if run_ppl_eval:
    llm_lib_log_metric(ModelType.prepared_model, Metric.ppl, prepared_kvcache_ppl, model_name="base")


# ---
# ## 7. Quantization
# 
# The _Quantization_ step is the primary focus of this notebook, this section could be modified to execute various quantization experiments.

print("=" * 80)
print("7. Quantization")
print("=" * 80)

# ---
# ### 7.1 Create quantsim configured for QNN HTP target 

# In[37]:

print("=" * 80)
print("7.1 Create quantsim configured for QNN HTP target")
print("=" * 80)

from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel

if apply_seqmse:
    # Create copy of fp model defintion for SeqMSE
    from copy import deepcopy
    fp_prepared_model = deepcopy(prepared_model)

dummy_input = get_dummy_data(device = "cuda", dtype = next(model.parameters()).dtype, return_dict = True)
sig = inspect.signature(prepared_model.forward)
dummy_input_sorted = {}
for key in list(sig.parameters.keys()):
    dummy_input_sorted[key] = dummy_input[key]
dummy_input = tuple(dummy_input_sorted.values())

with event_marker("create KVCache Quantsim"):
    with place_model(prepared_model, "cuda"):
        quantsim = QuantizationSimModel(model=prepared_model,
                                        quant_scheme=QuantScheme.post_training_tf,
                                        dummy_input=dummy_input,
                                        default_output_bw=16,
                                        default_param_bw=4,
                                        in_place=True,
                                        config_file=htp_config_file)


# ---
# ### 7.2 Setting 16bit x 8bit matmuls
# To keep key and value tensors as 8 bits, reducing data I/O costs associated with KV-cache orchestration.

# In[38]:

print("=" * 80)
print("7.2 Setting 16bit x 8bit matmuls")
print("=" * 80)


from aimet_torch.v2.experimental.quantsim_utils import set_matmul_second_input_producer_to_8bit_symmetric

if key_value_bitwidth == 8:
    set_matmul_second_input_producer_to_8bit_symmetric(quantsim)


# ---
# ### 7.3 Concat encoding unification
# configuring concat ops to have shared encoding on input and output activations.

# In[39]:

print("=" * 80)
print("7.3 Concat encoding unification")
print("=" * 80)

from aimet_torch.v2.experimental import propagate_output_encodings
from aimet_torch.nn.modules import custom as aimet_ops

propagate_output_encodings(quantsim, aimet_ops.Concat)


# ---
# ### 7.4 Manual Mixed Precision
# applying mixed precision configuration to ops 

# In[40]:

print("=" * 80)
print("7.4 Manual Mixed Precision")
print("=" * 80)

import json

key_cache_module = ".*k_R3.*Conv$"
value_cache_module = ".*v_proj.*Conv$"

with open("./config/mixed_precision_config/exceptions.json", "r") as f_in:
    mixed_precision_config = json.load(f_in)

# Customize mixed precision config based on user parameters
for entry in mixed_precision_config['name_list']:
    if "model_embed_tokens_Gather" in entry['module_name']:
        entry['exceptions']['param_exceptions']['bitwidth'] = embedding_table_bitwidth
    if key_cache_module == entry['module_name'] or value_cache_module == entry['module_name']:
        entry['exceptions']['output_exceptions'][0]['bitwidth'] = key_value_bitwidth


# In[41]:


from llm_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig

quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file = mixed_precision_config)
quantsim_adjuster.apply_exceptions(quantsim)


# In[42]:


from aimet_torch.v2.nn.modules.custom import QuantizedRmsNorm
from aimet_torch.v2.quantization.affine import QuantizeDequantize

# Make RMSNorm encodings per-tensor (they default to per-channel)
for name, qmodule in quantsim.named_qmodules():
    if isinstance(qmodule, QuantizedRmsNorm):
        qmodule.param_quantizers['weight'] = QuantizeDequantize(shape=(), bitwidth=16, symmetric=False).to(qmodule.weight.device)


# ---
# ### 7.5 Apply Block Quantization
# Swapping needed modules' weight quantizers to LPBQ quantizers

# In[43]:

print("=" * 80)
print("7.5 Apply Block Quantization")
print("=" * 80)


from aimet_torch.v2.nn.true_quant import QuantizedConv2d
from aimet_torch.v2.quantsim.config_utils import set_grouped_blockwise_quantization_for_weights

lpbq_conditions = []
r3_modules = [module for name, module in quantsim.model.named_modules() if "R3" in name]

if apply_decoder_lpbq:
    lpbq_conditions.append(lambda module: isinstance(module, QuantizedConv2d) and module.param_quantizers['weight'].bitwidth == 4 and module not in r3_modules)
if apply_lm_head_lpbq:
    lm_head_modules = [qmodule for name, qmodule in quantsim.named_qmodules() if "lm_head" in name]
    lpbq_conditions.append(lambda module: module in lm_head_modules and isinstance(module, QuantizedConv2d) and module not in r3_modules)

arg = (lambda module: any(condition(module) for condition in lpbq_conditions)) if lpbq_conditions else None

if arg:
    set_grouped_blockwise_quantization_for_weights(sim = quantsim,
                                                   arg = arg,
                                                   bitwidth = 4,
                                                   symmetric = True,
                                                   decompressed_bw = 8,
                                                   block_size = 64,
                                                   block_grouping = -1)


# ---
# ### 7.6 Sequential MSE
# applying sequential MSE technique to optimize parameter encodings

# In[44]:

print("=" * 80)
print("7.6 Sequential MSE")
print("=" * 80)

if apply_seqmse:
    from aimet_torch.v2.seq_mse import apply_seq_mse, SeqMseParams

    def _forward_fn(_model, inputs):
        # Always use 'base' adapter for Sequential-MSE algorithm
        model.model = _model
        model(**inputs)
    r3_modules = [module for name, module in fp_prepared_model.named_modules() if "R3" in name]
    params = SeqMseParams(num_batches = num_seqmse_batches,
                          inp_symmetry = 'symqt',
                          num_candidates = num_seqmse_candidates,
                          loss_fn = 'mse',
                          forward_fn = _forward_fn)

    with event_marker("SeqMSE"):
        with place_model(quantsim.model, torch.device("cuda")), place_model(fp_prepared_model, torch.device("cuda")):
            with torch.no_grad():
                apply_seq_mse(fp_prepared_model, quantsim, wikitext_train_dataloader, params, modules_to_exclude=r3_modules)
    del fp_prepared_model


# ---
# ### 7.8 Calibration

# In[45]:

print("=" * 80)
print("7.8 Calibration")
print("=" * 80)

from tqdm import tqdm
from aimet_torch.v2.experimental.quantsim_utils import clip_weights_to_7f7f

def _forward_fn(sim_model, kwargs):
    model.model = sim_model
    data_loader = kwargs['data_loader']
    max_iterations = kwargs['num_batches']
    for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
        if batch_id < max_iterations:
            model(input_ids = change_tensor_device_placement(batch['input_ids'], model.device))
        else:
            break

kwargs = {
    'data_loader': wikitext_train_dataloader,
    'num_batches': num_calibration_batches,
}

with event_marker("compute encoding", flush_ram=True):
     with place_model(quantsim.model, "cuda"):
        quantsim.compute_encodings(_forward_fn, kwargs)

clip_weights_to_7f7f(quantsim)


# ---
# ### 7.9 Apply Activation Clipping

# In[46]:

print("=" * 80)
print("7.9 Apply Activation Clipping")
print("=" * 80)

if apply_clipping:
    from aimet_torch.v2.nn.base import BaseQuantizationMixin as QUANTIZED_MODULE

    clamp_val = 200

    def clip_and_recompute_encodings(quantizer, name, clamp_val):
        if not quantizer.is_initialized():
            return
        qmin = quantizer.min.min()
        qmax = quantizer.max.max()
        if qmin < -clamp_val or qmax > clamp_val:
            quantizer.min.data = torch.clamp(quantizer.min, -clamp_val, clamp_val)
            quantizer.max.data = torch.clamp(quantizer.max, -clamp_val, clamp_val)

            print(f"{name} activation clamping... before: {qmin}, {qmax} | after: {quantizer.min.min().item()}, {quantizer.max.max().item()}")

    for name, module in quantsim.model.named_modules():
        if isinstance(module, QUANTIZED_MODULE):
            for quantizer in module.output_quantizers:
                if quantizer:
                    clip_and_recompute_encodings(quantizer, name + " | output quantizer", clamp_val)
            for quantizer in module.input_quantizers:
                if quantizer:
                    clip_and_recompute_encodings(quantizer, name + " | input quantizer", clamp_val)


# ---
# ### 7.10 Eval KV Cache sim on Base Model

# In[47]:

print("=" * 80)
print("7.10 Eval KV Cache sim on Base Model")
print("=" * 80)

if run_ppl_eval:
    with event_marker("KV cache sim with base model eval", flush_ram=True):
        with place_model(quantsim.model, torch.device("cuda")):
            model.model = quantsim.model
            sim_ppl = llm_evaluate_ppl_with_dataloader(model=model, dataloader=wikitext_test_dataloader)

    print(f"ppl score of KVCACHE sim with base model = {sim_ppl}")
    print(f"Diff between orig ppl and kvcache sim ppl = {orig_ppl - sim_ppl}")


# In[48]:


from genai_lib.common.debug.recipe_logger import dump_logs_to_json

if run_ppl_eval:
    # Recipe_logger: Log the ppl for qsim model and dump the cumulative logs to a JSON file.
    llm_lib_log_metric(ModelType.qsim_model, Metric.ppl, sim_ppl)

dump_logs_to_json()


# ---
# ## 8. Export
# the pipeline call below would export onnx model, encodings and test vector for KVCache model.

# ---
# ### 8.1 Save quantsim model export

# In[49]:

print("=" * 80)
print("8.1 Save quantsim model export")
print("=" * 80)

import pickle as pkl

# Increase recursion depth limit to save full model
sys.setrecursionlimit(100000)

with event_marker("Save quantsim model"), open(f"{output_dir}/quantsim_model.pkl", 'wb') as file:
    pkl.dump(quantsim, file)


# ---
# ### 8.2 Export Onnx and Encodings

# In[50]:

print("=" * 80)
print("8.2 Export Onnx and Encodings")
print("=" * 80)

from aimet_torch import onnx_utils

# Setting this flag to False means that the prepared model will be flattened
onnx_utils.EXPORT_TO_ONNX_DIRECT = True

onnx_dir = os.path.join(output_dir, 'onnx')
os.makedirs(onnx_dir, exist_ok=True)

input_names, output_names = llm_model_input_output_names(llm_config.num_hidden_layers, use_position_embedding_input=True, separate_tuple_input_output=True)

if enable_fp16:
    # Convert FP16 model back to FP32 for ONNX export
    torch.set_default_dtype(torch.float32)
    model.float()

dummy_input = get_dummy_data(device = "cpu", dtype = next(model.parameters()).dtype, return_dict = True)

sig = inspect.signature(prepared_model.forward)
dummy_input_sorted = {}
for key in list(sig.parameters.keys()):
    dummy_input_sorted[key] = dummy_input[key]
dummy_input = dummy_input_sorted
dummy_input = tuple(list(dummy_input.values()))

onnx_api_args = onnx_utils.OnnxExportApiArgs(input_names=input_names, output_names=output_names, opset_version=17)

with event_marker("Model export", flush_ram=True):
    with place_model(quantsim.model, torch.device("cpu")):
        quantsim.export(onnx_dir, model_name, dummy_input, onnx_export_args=onnx_api_args)

tokenizer_dir = os.path.join(output_dir, 'tokenizer')
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save_pretrained(tokenizer_dir)


# ---
# ### 8.3 Generating test vectors for QNN SDK

# In[51]:

print("=" * 80)
print("8.3 Generating test vectors for QNN SDK")
print("=" * 80)

from genai_lib.llm.test_vectors import generate_test_vectors

test_vector_layers = [
    "model_embed_tokens_Gather",
    "model_layers_\\d+_Add_1"
]

num_test_vectors = get_config_value("NUM_TEST_VECTORS", 1, 'int')
with event_marker("generate test vector"):
    with place_model(model, torch.device("cuda")):
        for index, batch in enumerate(wikitext_train_dataloader):
            if index >= num_test_vectors:
                break
            input_ids_slice = batch['input_ids'][..., :ARN].to(device=torch.device('cuda'))
            attention_mask = torch.ones((input_ids_slice.shape[0], ARN), dtype = torch.long, device=torch.device('cuda'))
            outputs = {'past_key_values': None}
            model_inputs = prepare_inputs_for_static_shape(model = model,
                                                           input_ids_slice = input_ids_slice,
                                                           attn_mask_slice = attention_mask,
                                                           outputs = outputs)
            generate_test_vectors(sim=quantsim, model_inputs=model_inputs, output_dir=output_dir, batch_index=index, test_vector_layers=test_vector_layers)


# ---
# ### Summary

# In[52]:

print("=" * 80)
print("Summary")
print("=" * 80)

from genai_lib.common.debug.profiler import EventProfiler
EventProfiler().report()
EventProfiler().json_dump(os.path.join(output_dir, 'profiling_stats.json'))


# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
