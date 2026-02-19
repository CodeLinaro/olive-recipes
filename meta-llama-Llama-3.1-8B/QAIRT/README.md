# Llama 3.1 8B QAIRT Preparation

This directory contains tools and scripts for preparing the [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) model for deployment on Qualcomm AI Runtime (QAIRT) using post-training quantization (PTQ).

## Overview

The workflow uses a standalone Python script (`llama3_1_script.py`) that handles:
- Model loading and preparation
- Post-training quantization using AIMET
- ONNX export with quantization encodings
- Context binary generation for QAIRT deployment

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA 12.1 support
- **Python**: Python 3.8 or higher
- **QAIRT SDK**: Qualcomm AI Runtime SDK installed
- **Model**: Llama 3.1 8B model files (downloaded from Hugging Face)

## Environment Setup

Setting up the environment is straightforward with these 4 steps:

### 1. Create Python Virtual Environment

```bash
python3 -m venv llama_venv
```

### 2. Activate the Virtual Environment

```bash
source llama_venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.1 Support

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Before running the script, you need to configure `config_example.json` with your environment-specific paths:

```json
{
  "QNN_SDK_ROOT": "/path/to/qairt/sdk/",
  "MODEL_ID": "/path/to/llama-3.1-8B-model/",
  "MODEL_NAME": "llama3_1",
  "HTP_CONFIG_FILE": "/path/to/venv/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_v73.json"
}
```

### Configuration Parameters

- **QNN_SDK_ROOT**: Path to your QAIRT SDK installation directory
  - Example: `/prj/qct/aisw_scratch/lv/local_dev/shared/sdks/qairt/2.43.0.260123/`
  
- **MODEL_ID**: Path to the directory containing your Llama 3.1 8B model files
  - This should be the directory with `config.json`, model weights, tokenizer files, etc.
  - Example: `/path/to/llama-3.1-8B-model/`
  
- **MODEL_NAME**: Name identifier for the model (typically "llama3_1")
  
- **HTP_CONFIG_FILE**: Path to the AIMET HTP quantization configuration file
  - This file is installed with aimet-torch and defines quantization parameters
  - The path will be within your virtual environment's site-packages
  - Example: `<venv_path>/lib/python3.10/site-packages/aimet_common/quantsim_config/htp_quantsim_config_v73.json`

### Finding the HTP Config File Path

After installing the requirements, you can find the HTP config file path with:

```bash
python -c "import aimet_common; import os; print(os.path.join(os.path.dirname(aimet_common.__file__), 'quantsim_config', 'htp_quantsim_config_v73.json'))"
```

## Running the Script

### Option 1: Direct Execution (Recommended)

Once your environment is set up and configured, run the script directly:

```bash
python llama3_1_script.py --config config_example.json
```

The script will:
1. Load the Llama 3.1 8B model
2. Apply post-training quantization using AIMET
3. Export quantized ONNX models with encodings
4. Generate context binaries for QAIRT deployment

### Option 2: Via Olive QairtPreparationPass

This workflow can also be integrated into Olive pipelines using the `QairtPreparationPass`. The pass wraps the standalone script and can be configured in an Olive workflow JSON file.

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
- Verify `QNN_SDK_ROOT` path in `config_example.json` is correct
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
- Update `HTP_CONFIG_FILE` in `config_example.json` with the correct path
- Ensure aimet-torch is properly installed

## Additional Resources

- [QAIRT Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html)
- [AIMET Documentation](https://quic.github.io/aimet-pages/index.html)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review QAIRT SDK documentation
- Consult AIMET documentation for quantization-specific issues
