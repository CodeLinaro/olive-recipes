
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating virtual environment: $VIRTUAL_ENV"
    deactivate
else
    echo "No virtual environment is currently active."
fi

VENV_DIR="phi35-venv"
if [[ -d $VENV_DIR ]]; then
    echo "Deleting virtual environment directory: $VENV_DIR"
    rm -rf $VENV_DIR
else
    echo "Virtual environment directory DNE: $VENV_DIR"
fi

echo "Creating venv..."
uv venv -p 3.10 --seed $VENV_DIR
echo "Activating venv..."
source $VENV_DIR/bin/activate

export OLIVE_LOCATION=/local/mnt2/workspace/kromero/olive/Olive
echo "Removing old olive installation"
rm -rf $OLIVE_LOCATION/build
rm -rf $OLIVE_LOCATION/dist

cd $OLIVE_LOCATION
echo "Building Olive..."
python setup.py bdist_wheel

cd -

uv pip install $OLIVE_LOCATION/dist/*.whl

export QAIRT_PIPELINE_LOCATION=/local/mnt2/workspace/kromero/qairt-tools/qairt-tools/core/experimental

if [[ ":$PYTHONPATH:" != *":$QAIRT_PIPELINE_LOCATION:"* ]]; then
    export PYTHONPATH="$QAIRT_PIPELINE_LOCATION:$PYTHONPATH"
fi

echo "Installing QAIRT pipeline API dependencies..."
uv pip install -r $QAIRT_PIPELINE_LOCATION/pipeline/requirements.txt

uv pip install qairt_dev-0.1.0.dev0-py3-none-manylinux2014_x86_64.whl[onnx]

uv pip install -r requirements.txt
