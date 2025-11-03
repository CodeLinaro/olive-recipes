## Whisper-large-v3-turbo Optimization with ONNX Runtime QNN EP

### Prerequisites
```bash
python -m pip install -r requirements_qnn.txt
```
### Generate data for static quantization

To get better results, we need to generate real data from original model instead of using random data for static quantization.

First generate fp32 onnx models:
1. Encoder fp32 model

    `olive run --config whisper_encoder_fp32.json`
1. Decoder fp32 model

    `olive run --config whisper_decoder_fp32.json`


Then download and generate data:
1. `python download_librispeech_asr.py --save_dir .\data`

2. `python .\demo.py --audio-path .\data\librispeech_asr_clean_test --encoder "models\whisper_encoder_fp32\model\model.onnx" --decoder "models\whisper_decoder_fp32\model.onnx" --model_id "openai/whisper-large-v3-turbo" --save_data .\data\quantization_data`

### Generate QDQ models

1. `olive run --config whisper_encoder_qdq.json`
2. `olive run --config whisper_decoder_qdq.json`

### Evaluation

`python .\evaluate_whisper.py --encoder "models\whisper_encoder_qdq\model.onnx --decoder "models\whisper_decoder_qdq\model.onnx --model_id "openai/whisper-large-v3-turbo" --execution_provider QNNExecutionProvider`

### To transcribe a single sample:

`python .\demo.py --audio-path .\data\librispeech_asr_clean_test\1320-122617-0000.npy --encoder "models\whisper_encoder_qdq\model.onnx" --decoder "models\whisper_decoder_qdq\model.onnx" --model_id "openai/whisper-large-v3-turbo" --execution_provider QNNExecutionProvider`