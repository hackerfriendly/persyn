#!/bin/bash
set -e

virtualenv --python=python3.8 env --prompt='(slack-env) '
. env/bin/activate

pip install -r requirements.txt
