#!/usr/bin/env bash

VENVNAME=Computer_Vision04

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install tensorflow
pip install matplotlib
pip install opencv-python
pip install pydot
pip install tqdm
python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install requirements.txt

deactivate
echo "build $VENVNAME"