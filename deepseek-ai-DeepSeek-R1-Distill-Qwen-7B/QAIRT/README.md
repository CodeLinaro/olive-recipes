# DeepSeek-R1-Distill-Qwen-7B QAIRT Preparation

This directory contains tools and scripts for preparing the [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) model for deployment on Qualcomm AI Runtime (QAIRT) using post-training quantization (PTQ).

## Overview

The workflow uses a standalone Python script (`deepseek_r1_qwen2.py`) that handles:
- Model loading and preparation
- Optional AdaScale weight optimization
- Post-training quantization using AIMET
- ONNX export with quantization encodings
- Context binary generation for QAIRT deployment

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support
- **Python**: Python 3.8 or higher
- **QAIRT SDK**: Qualcomm AI Runtime SDK installed
- **Model**: DeepSeek-R1-Distill-Qwen-7B model (downloaded from Hugging Face)

## Environment Setup

Setting up the environment is straightforward with these 3 steps:

### 1. Create Python Virtual Environment

```bash
python3 -m venv deepseek_venv
```

### 2. Activate the Virtual Environment

```bash
source deepseek_venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt uses flexible PyTorch version constraints (`torch<2.9`) and shares the same dependencies with other models like Llama 3.1 for easier maintenance.

## Configuration

The script uses a JSON configuration file instead of environment variables. Before running the script, configure `config_example.json` with your environment-specific paths:

```json
{
  "QNN_SDK_ROOT": "/path/to/qairt/sdk/",
  "MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "MODEL_NAME": "deepseek_r1_qwen2",
  "HTP_CONFIG_FILE": "/path/to/venv/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_v73.json"
}
```

### Configuration Parameters

#### Required Parameters

- **QNN_SDK_ROOT**: Path to your QAIRT SDK installation directory
  - Example: `/prj/qct/aisw_scratch/lv/local_dev/shared/sdks/qairt/2.43.0.260123/`
  
- **MODEL_ID**: Model identifier or path
  - For Hugging Face: `"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"` (downloads automatically)
  - For local model: `/path/to/local/model/directory/`

#### Optional Parameters

##### Feature Configuration
- **CONTEXT_LENGTH**: Context length for the model (default: 4096)

##### Quantization Configuration
- **APPLY_SEQMSE**: Apply SeqMSE optimization (default: True)
- **APPLY_DECODER_LPBQ**: Apply LPBQ to decoder (default: True)
- **APPLY_LM_HEAD_LPBQ**: Apply LPBQ to LM head (default: False)
- **APPLY_CLIPPING**: Apply activation clipping (default: False)
- **EMBEDDING_TABLE_BITWIDTH**: Embedding table bitwidth: 8 or 16 (default: 16)
- **KEY_VALUE_BITWIDTH**: Key-value cache bitwidth: 8 or 16 (default: 8)
- **NUM_CALIBRATION_BATCHES**: Number of calibration batches (default: 200)
- **NUM_SEQMSE_BATCHES**: Number of SeqMSE batches (default: 20)
- **NUM_SEQMSE_CANDIDATES**: Number of SeqMSE candidates (default: 20)

##### Speed Configuration
- **ENABLE_FP16**: Enable FP16 flow (default: False)
- **RUN_PPL_EVAL**: Run perplexity evaluation (default: True)

##### AdaScale Configuration (Advanced)
- **ENABLE_ADASCALE**: Enable AdaScale optimization (default: False)
- **ADASCALE_ITERATIONS**: AdaScale iterations (default: 1500)
- **C4_DATASET_PATH**: Path to C4 dataset for AdaScale (required if AdaScale enabled)
- **BATCH_SIZE**: Batch size for AdaScale (default: 2)
- **PERCENT_DATASET_TO_LOAD**: Percent of dataset to load (default: 3)
- **NUM_SAMPLES**: Number of samples for AdaScale (default: 1000)

##### QNN SDK Configuration
- **LD_LIBRARY_PATH**: Library path for QNN SDK (default: None)

##### NSP Target Configuration
- **TARGET_PLATFORM**: Target platform: Windows/Android (default: Android)
- **PLATFORM_GEN**: Platform generation: 2/4/5 (default: 5)
- **HTP_CONFIG_FILE**: Path to HTP quantsim config file (default: hardcoded path)

##### Model Configuration
- **MODEL_NAME**: Model name identifier (default: deepseek_r1_qwen2)
- **CACHE_DIR**: Cache directory path (default: ./cache_dir)
- **OUTPUT_DIR**: Output directory path (default: computed from model_id)
- **ADASCALE_DIR**: AdaScale directory path (default: computed from model_id)
- **NUM_HIDDEN_LAYERS**: Number of hidden layers, 0=use model default (default: 0)

##### ARN Configuration
- **ARN**: Auto-regression length (default: 2073)
- **MASK_NEG**: Mask negative value (default: -3100)

##### Prepare Configuration
- **SKIP_PREPARE**: Skip model preparation (default: False)

##### Test Vector Configuration
- **NUM_TEST_VECTORS**: Number of test vectors to generate (default: 1)

### Finding the HTP Config File Path

After installing the requirements, you can find the HTP config file path with:

```bash
python -c "import aimet_common; import os; print(os.path.join(os.path.dirname(aimet_common.__file__), 'quantsim_config', 'htp_quantsim_config_v73.json'))"
```

## Running the Script

Once your environment is set up and configured, run the script with your config file:

```bash
python deepseek_r1_qwen2.py --config config_example.json
```

The script will:
1. Load the DeepSeek-R1-Distill-Qwen-7B model
2. Optionally apply AdaScale weight optimization (if enabled)
3. Apply post-training quantization using AIMET
4. Export quantized ONNX models with encodings
5. Generate context binaries for QAIRT deployment

### Configuration Priority

The script supports a 3-tier configuration priority system:
1. **JSON config file** (highest priority) - values specified in the config file
2. **Environment variables** (medium priority) - if not in config file
3. **Default values** (lowest priority) - if not specified anywhere

This allows flexible configuration while maintaining backward compatibility.

### Getting Help

To see all available configuration options:

```bash
python deepseek_r1_qwen2.py --help
```

## Output

The script generates optimized model artifacts in the output directory:

- **ONNX Models**: Quantized ONNX models
- **Encodings**: Quantization encoding files (`.encodings`)
- **Test Vectors**: Test vectors for validation
- **Quantsim Model**: Saved quantization simulation model (`.pkl`)

## About AdaScale (Advanced Feature)

**AdaScale** is an optional weight optimization technique that can improve quantization quality by adjusting model weights to be more quantization-friendly.

### When to Use AdaScale

- **Default: DISABLED** - The script works well without AdaScale for most use cases
- **Enable for**: Maximum accuracy when you have the C4 dataset available
- **Trade-off**: Requires additional time and the C4 dataset for calibration

### Enabling AdaScale

To enable AdaScale, add to your config JSON:

```json
{
  "ENABLE_ADASCALE": true,
  "C4_DATASET_PATH": "/path/to/c4/dataset.json",
  "ADASCALE_ITERATIONS": 1500
}
```

**Note**: AdaScale requires a C4 dataset in JSON format. If you don't have this dataset, keep AdaScale disabled (the default).

## Key Differences from Llama 3.1

This DeepSeek implementation has several differences from the Llama 3.1 setup:

- **Architecture**: Uses Qwen2 architecture instead of Llama
- **Context Length**: Default 4096 (vs 4173 for Llama)
- **Platform Generation**: Default GEN5 (vs GEN2 for Llama)
- **Quantization**: LPBQ enabled by default for decoder
- **AdaScale**: Optional weight optimization feature (not in Llama)
- **Embedding Bitwidth**: Default 16-bit (vs 8-bit for Llama)

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
- Verify `MODEL_ID` is correct for Hugging Face download
- Or verify local model path contains all required files
- Check internet connection if downloading from Hugging Face
- Ensure you have read permissions for local model directory

### HTP Config File Not Found

If the quantization config file is not found:
- Run the command in "Finding the HTP Config File Path" section
- Update `HTP_CONFIG_FILE` in your config JSON with the correct path
- Ensure aimet-torch is properly installed

### AdaScale Errors

If you encounter errors with AdaScale enabled:
- Verify `C4_DATASET_PATH` points to a valid JSON dataset
- Ensure the dataset is in the correct format
- Try disabling AdaScale if you don't need the extra optimization

## Additional Resources

- [QAIRT Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html)
- [AIMET Documentation](https://quic.github.io/aimet-pages/index.html)
- [DeepSeek-R1 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Qwen2 Model Documentation](https://huggingface.co/docs/transformers/model_doc/qwen2)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review QAIRT SDK documentation
- Consult AIMET documentation for quantization-specific issues
- Check transformers documentation for model-related questions
- Review Qwen2 architecture documentation for model-specific details
