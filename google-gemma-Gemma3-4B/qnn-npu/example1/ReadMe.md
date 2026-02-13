# Setup Instructions

## Platform requirements
This notebook is intended to run on a machine with:
  * Ubuntu 22.04 on a x86_64 system
  * NVIDIA driver version equivalent to 525.60.13
  * NVIDIA A100 GPU
  * AIMET version 2.13.0
  * QAIRT version 2.40

## 1. Create and Activate Python Virtual Environment

```bash
cd example1
python3.10 -m venv venv_gemma
source venv_gemma/bin/activate
pip install uv
uv pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn
uv pip install -r example1_env_req.txt
```

***

## 2. Install Qualcomm Package Manager (QPM)

1.  Download QPM for Linux from:  
    **<https://qpm.qualcomm.com/#/main/tools/details/QPM3>**

2.  Copy the downloaded `.deb` package to the current folder.

3.  Install QPM:

```bash
sudo dpkg -i QualcommPackageManager*.Linux-x86.deb
which qpm-cli
```

4.  Log in and activate license:

```bash
qpm-cli --login <username>
qpm-cli --license-activate Qualcomm_AI_Runtime_SDK
```

5.  Extract the QNN SDK:

```bash
qpm-cli --extract Qualcomm_AI_Runtime_SDK -v "2.40.1.251119" --config /tmp/qnn
```

***

## 3. Set Up QNN SDK Environment

```bash
export QNN_SDK_ROOT="/tmp/qnn"
source $QNN_SDK_ROOT/bin/envsetup.sh
sudo $QNN_SDK_ROOT/bin/check-linux-dependency.sh
$QNN_SDK_ROOT/bin/check-python-dependency
sudo apt-get install -y libtinfo5
```

***

## 4. Setup and Launch Jupyter Lab

### Login to Hugging Face
To access models, you'll need to log-in to Hugging Face with a user [access token](https://huggingface.co/docs/hub/security-tokens). The following command will run you through the steps to login:
```bash
huggingface-cli login --token <>
```

### Launch Jupyter Lab

```bash
jupyter lab --ip=$HOSTNAME --no-browser --allow-root
```

***

# Inside Jupyter Lab

## 5. Update Dataset Path in Notebooks

For the notebooks:

*   `gemma3_4b.ipynb`
*   `gemma3_veg.ipynb`

Replace:

```python
dataset_path = "<path to folder containing the coco dataset root folder>"
```

With the actual path where COCO dataset root folder is located.

### Verify dataset structure:

```bash
tree <path to folder containing the coco dataset root folder> -L 2 | head -n 10
```

Example expected structure:

    <path to folder containing the coco dataset root folder>
    └── coco
        └── train2017
            ├── 000000000009.jpg
            ├── 000000000025.jpg
            ├── 000000000030.jpg
            ├── 000000000034.jpg
            ├── 000000000036.jpg

***

## 6. Run Notebooks in Order

⚠️ **Important:** After each notebook completes, **kill the kernel** to free GPU memory before starting the next one.

1.  `spinquant.ipynb`
2.  `gemma3_4b.ipynb`
3.  `gemma3_veg.ipynb`
4.  `gemma_4b_embed.ipynb`

***

## 7. Download Generated Artifacts

After running the notebooks, download:

*   `embed_fp32/embed_fp32.onnx`
*   `embedding_layer.weight`

***
