# Llama 3.1 8B QAIRT Preparation

This directory contains tools and scripts for preparing the [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) model for deployment on Qualcomm AI Runtime (QAIRT) using post-training quantization (PTQ).

## Overview

The workflow uses a standalone Python script (`llama3_1_script.py`) that handles:
- Model loading and preparation
- Post-training quantization using AIMET
- ONNX export with quantization encodings
- Context binary generation for QAIRT deployment

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support
- **Python**: Python 3.8 or higher
- **QAIRT SDK**: Qualcomm AI Runtime SDK installed
- **Model**: Llama 3.1 8B model files (downloaded from Hugging Face)

## Environment Setup

Setting up the environment is straightforward with these 3 steps:

### 1. Create Python Virtual Environment

```bash
python3 -m venv llama31_venv
```

### 2. Activate the Virtual Environment

```bash
source llama31_venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The new requirements.txt uses more flexible PyTorch version constraints (`torch<2.9`) and does not require special CUDA-specific installations. The environment is shared with other models like DeepSeek for easier maintenance.

## Configuration

The script uses a JSON configuration file instead of environment variables. Before running the script, configure `config_example.json` with your environment-specific paths:

```json
{
  "QNN_SDK_ROOT": "/path/to/qairt/sdk/",
  "MODEL_ID": "/path/to/llama-3.1-8B-model/",
  "MODEL_NAME": "llama3_1",
  "HTP_CONFIG_FILE": "/path/to/venv/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_v73.json"
}
```

### Configuration Parameters

#### Required Parameters

- **QNN_SDK_ROOT**: Path to your QAIRT SDK installation directory
  - Example: `/prj/qct/aisw_scratch/lv/local_dev/shared/sdks/qairt/2.43.0.260123/`
  
- **MODEL_ID**: Path to the directory containing your Llama 3.1 8B model files
  - This should be the directory with `config.json`, model weights, tokenizer files, etc.
  - Example: `/path/to/llama-3.1-8B-model/`

#### Optional Parameters

- **MODEL_NAME**: Name identifier for the model (default: "llama3_1")
  
- **HTP_CONFIG_FILE**: Path to the AIMET HTP quantization configuration file
  - This file is installed with aimet-torch and defines quantization parameters
  - The path will be within your virtual environment's site-packages
  - Example: `<venv_path>/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_v73.json`

- **CONTEXT_LENGTH**: Context length for the model (default: 4173)
- **ENABLE_RIGHT_PADDING**: Enable right padding of kvcache (default: True)
- **APPLY_DECODER_SEQMSE**: Apply SeqMSE to decoder (default: False)
- **APPLY_LM_HEAD_SEQMSE**: Apply SeqMSE to LM head (default: False)
- **APPLY_DECODER_LPBQ**: Apply LPBQ to decoder (default: False)
- **APPLY_LM_HEAD_LPBQ**: Apply LPBQ to LM head (default: False)
- **ACTIVATION_CLIPPING_CLAMP_VAL**: Activation clipping value (default: None)
- **EMBEDDING_TABLE_BITWIDTH**: Embedding table bitwidth: 8 or 16 (default: 8)
- **ENABLE_FP16**: Enable FP16 flow (default: False)
- **RUN_PPL_EVAL**: Run perplexity evaluation (default: True)
- **SKIP_PREPARE**: Skip model preparation (default: False)
- **TARGET_PLATFORM**: Target platform: Windows/Android (default: Windows)
- **PLATFORM_GEN**: Platform generation: 2/4/5 (default: 2)
- **CACHE_DIR**: Cache directory path (default: ./cache_dir)
- **OUTPUT_DIR**: Output directory path (default: ./output_dir)
- **NUM_HIDDEN_LAYERS**: Number of hidden layers, 0=use model default (default: 0)
- **BASE_CALIBRATION_DATASET**: Calibration dataset name (default: WIKITEXT)
- **ARN**: Auto-regression length (default: 2073)

### Finding the HTP Config File Path

After installing the requirements, you can find the HTP config file path with:

```bash
python -c "import aimet_common; import os; print(os.path.join(os.path.dirname(aimet_common.__file__), 'quantsim_config', 'htp_quantsim_config_v73.json'))"
```

## Running the Script

Once your environment is set up and configured, run the script with your config file:

```bash
python llama3_1_script.py --config config_example.json
```

The script will:
1. Load the Llama 3.1 8B model
2. Apply post-training quantization using AIMET
3. Export quantized ONNX models with encodings
4. Generate context binaries for QAIRT deployment

### Configuration Priority

The script supports a 3-tier configuration priority system:
1. **JSON config file** (highest priority) - values specified in the config file
2. **Environment variables** (medium priority) - if not in config file
3. **Default values** (lowest priority) - if not specified anywhere

This allows flexible configuration while maintaining backward compatibility.

### Getting Help

To see all available configuration options:

```bash
python llama3_1_script.py --help
```

## Output

The script generates optimized model artifacts in the working directory:

- **ONNX Models**: Quantized ONNX models with separate files for different context lengths
- **Encodings**: Quantization encoding files (`.encodings`)
- **Context Binaries**: QAIRT-ready `.bin` files for deployment
- **DLC Files**: Deep Learning Container files for QAIRT runtime

Output files are organized by:
- Autoregressive length (e.g., `ar1`, `ar128`)
- Context length (e.g., `cl4096`)
- Model partition (e.g., `1_of_6`, `2_of_6`, etc.)

## Key Updates in This Version

This version includes several improvements over the previous setup:

- **Simplified Installation**: No special PyTorch CUDA installation required
- **Updated Transformers**: Now uses transformers 4.50.1 with compatibility fixes
- **Flexible PyTorch**: Uses `torch<2.9` instead of specific CUDA versions
- **JSON Configuration**: Cleaner configuration via JSON files instead of environment variables
- **Shared Environment**: Compatible with other models (e.g., DeepSeek) for easier maintenance

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce batch size in the script configuration
- Use a GPU with more memory
- Close other GPU-intensive applications

### Missing Dependencies

If you get import errors:
- Ensure you activated the virtual environment
- Verify all packages installed correctly: `pip list`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### QAIRT SDK Not Found

If the script cannot find QAIRT SDK:
- Verify `QNN_SDK_ROOT` path in your config JSON is correct
- Ensure the SDK directory contains the expected structure
- Check that you have read permissions for the SDK directory

### Model Loading Errors

If the model fails to load:
- Verify `MODEL_ID` path points to a valid Llama 3.1 8B model directory
- Ensure the directory contains all required files (`config.json`, weights, tokenizer)
- Check that you have read permissions for the model directory

### HTP Config File Not Found

If the quantization config file is not found:
- Run the command in "Finding the HTP Config File Path" section
- Update `HTP_CONFIG_FILE` in your config JSON with the correct path
- Ensure aimet-torch is properly installed

### Transformers 4.50 Compatibility

The script includes compatibility fixes for transformers 4.50. If you encounter issues:
- Check that you're using the correct transformers version: `pip show transformers`
- The script will print warnings if certain attributes are not found
- These warnings are informational and usually don't affect functionality

## Additional Resources

- [QAIRT Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html)
- [AIMET Documentation](https://quic.github.io/aimet-pages/index.html)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review QAIRT SDK documentation
- Consult AIMET documentation for quantization-specific issues
- Check transformers documentation for model-related questions
