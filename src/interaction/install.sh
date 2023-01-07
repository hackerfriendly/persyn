#!/bin/bash
set -e

if [ $(uname -s) = "Darwin" ]; then
  brew install cmake
fi

python3 -m venv --prompt='interact-env' env
. env/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_lg
