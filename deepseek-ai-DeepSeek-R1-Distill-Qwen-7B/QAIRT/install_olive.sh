
cd ../../../../olive/Olive

python setup.py build
python setup.py bdist_wheel
uv pip install dist/olive_ai-0.11.0.dev0-py3-none-any.whl

cd -
