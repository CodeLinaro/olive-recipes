***

# Artifacts Preparation and Model Build Instructions

## 1. Verify Downloaded Artifacts

Ensure the following files are present in the current folder:

1.  `build_ort_genai_model.ipynb`
2.  `config.json`
3.  `gen_qnn_ctx_onnx_model.py`
4.  `genai_config.json`
5.  `preprocessor_config.json`
6.  `processor_config.json`
7.  `special_tokens_map.json`
8.  `tokenizer.json`
9.  `tokenizer_config.json`

> **Next:** Copy all downloaded files from example1 and example2 into the current folder.

***

# Artifacts Generation Setup

## 2. Create Artifacts Generation Environment

```bash
python -m venv gemma_env
gemma_env\Scripts\activate
```

***

## 3. Install Required Packages

### Install Provided Wheel File

⚠️ Build wheel file using [onnxruntime-genai](https://github.com/CodeLinaro/onnxruntime-genai/commit/1b73c328bdc2ded86304312721d7051c1f9b95f0) commit

```bash
pip install --force-reinstall onnxruntime_genai-0.10.0.dev0-cp312-cp312-win_arm64.whl
```

### Install Supporting Dependencies

```bash
pip install jupyter notebook transformers==4.52.4 torch onnx==1.20.1 onnxscript==0.6.2 --extra-index-url https://download.pytorch.org/whl/cpu
```

***

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

***

# Building ONNX & QNN Context Models

## 5. Edit Notebook Configuration

Open **`build_ort_genai_model.ipynb`** and update:

*   **`QNN_SDK_DIR`** → Set this to the folder path of your **QNN-SDK** installation (folder contain `bin/` directory).

***

## 6. Run the Notebook

Execute **all cells** in `build_ort_genai_model.ipynb`.

Once complete, the following artifacts must be generated along with few additional files

***

# Final Artifacts List

The build process will generate the following files:

    ar1_cl8192_all_of_4_qnn_ctx.onnx
    ar128_cl8192_all_of_4_qnn_ctx.onnx
    config.json
    embed_fp32_mod.data
    embed_fp32_mod.onnx
    genai_config.json
    input-processor_swa_cl1_oga_rs.onnx
    input-processor_swa_cl128_oga_rs.onnx
    output_processor.onnx
    position-processor_swa_same_attn_quant.onnx
    preprocessor_config.json
    processor_config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    veg.serialized.bin
    veg_qnn_ctx_fp32_io.onnx
    weight_sharing_model_1_of_4.serialized.bin
    weight_sharing_model_2_of_4.serialized.bin
    weight_sharing_model_3_of_4.serialized.bin
    weight_sharing_model_4_of_4.serialized.bin

***

# Inference Environment Setup

## 7. Create Inference Environment

```bash
python -m venv gemma_test_env
gemma_test_env\Scripts\activate
```

### Install Required Wheel File


⚠️ Build wheel file using [onnxruntime-genai](https://github.com/CodeLinaro/onnxruntime-genai/commit/1b73c328bdc2ded86304312721d7051c1f9b95f0) commit

```bash
pip install --force-reinstall onnxruntime_genai-0.10.0.dev0-cp312-cp312-win_arm64.whl
```

***
