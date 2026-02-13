***

# Example 2 — QNN Model Preparation

## Platform requirements
This notebook is intended to run on a machine with:
  * Ubuntu 22.04 on a x86_64 system
  * QNN version 2.31.0
  * QAIRT version 2.40

## 1. Create and Activate Python Virtual Environment

```bash
python3.10 -m venv venv_gemma
source venv_gemma/bin/activate
pip install uv
uv pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn
uv pip install -r example2_env_req.txt
```

***

## 2. Set Up QNN SDK Environment

⚠️ **Important:** For QNN SDK installation refer to **example1\ReamMe.md**

```bash
export QNN_SDK_ROOT="/tmp/qnn"
source $QNN_SDK_ROOT/bin/envsetup.sh
sudo $QNN_SDK_ROOT/bin/check-linux-dependency.sh
$QNN_SDK_ROOT/bin/check-python-dependency
sudo apt-get install -y libtinfo5
```

***

## 3. Launch Jupyter Lab

```bash
jupyter lab --ip=$HOSTNAME --no-browser --allow-root
```

***

# Inside Jupyter Lab

## 4. Run the Following Notebooks

1.  `Example2A/host_linux/qnn_model_prepare.ipynb`
2.  `Example2B/host_linux/qnn_model_prepare_for_llm.ipynb`

***

## 5. Download Generated Artifacts

After notebook execution, download:

### From Example 2A:

    Example2A/host_linux/exports/serialized_binaries/veg.serialized.bin

### From Example 2B:

    Example2B/host_linux/assets/artifacts/ar128-ar1-cl8192/*.serialized.bin

***
