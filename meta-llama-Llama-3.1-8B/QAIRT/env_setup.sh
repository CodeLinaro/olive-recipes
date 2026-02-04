
#!/usr/bin/env bash

# Check if "dev" argument is passed
DEV_MODE=false
if [[ "$1" == "dev" ]]; then
    DEV_MODE=true
fi

# Deactivate existing virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating virtual environment: $VIRTUAL_ENV"
    deactivate
else
    echo "No virtual environment is currently active."
fi

# Remove old virtual environment
VENV_DIR="llama31-venv"
if [[ -d $VENV_DIR ]]; then
    echo "Deleting virtual environment directory: $VENV_DIR"
    rm -rf $VENV_DIR
else
    echo "Virtual environment directory DNE: $VENV_DIR"
fi

# Create new virtual environment
uv venv -p 3.10 $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install ../../../../olive/Olive/dist/olive_ai-0.11.0.dev0-py3-none-any.whl
uv pip install qairt-dev[onnx]

# Set QAIRT SDK root
export QAIRT_SDK_ROOT=/local/mnt2/workspace2/kromero/sdks/2.44.0.260115-genie-custom/

# Handle PYTHONPATH based on dev mode
if $DEV_MODE; then
    echo "Dev mode enabled: Setting PYTHONPATH"
    export PYTHONPATH=/local/mnt2/workspace2/kromero/qairt-tools/QAIRT_Tools/core/src/python/
else
    echo "Dev mode not enabled: Unsetting PYTHONPATH"
    unset PYTHONPATH
fi
