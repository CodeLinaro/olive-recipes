

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

uv venv -p 3.12 $VENV_DIR
source $VENV_DIR/bin/activate

uv pip install -r requirements.txt

uv pip install --no-build-isolation git+https://github.com/ModelCloud/GPTQModel.git@558449bed3ef2653c36041650d30da6bbbca440d

uv pip install ../../../../olive/Olive/dist/olive_ai-0.10.0.dev0-py3-none-any.whl[qairt] 

