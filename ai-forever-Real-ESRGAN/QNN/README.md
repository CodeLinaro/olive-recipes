# Exporting Real-ESRGAN for QNN

## Install Python Dependencies

```
pip install onnxruntime onnxruntime-qnn olive-ai
pip install git+https://github.com/ai-forever/Real-ESRGAN.git@6dfc3ec9352a273dd7aa9cf241dafee473702d0d --no-build-isolation
```

## Export to ONNX

### QNN GPU

```
olive run --config config_gpu_fp32.json
```

✅ Optimized model saved in: `output/`
