#!/bin/bash

if [ -z "$VIRTUAL_ENV" ]; then
  echo "Run this script from inside a python 3 virtualenv:"
  echo "  $ virtualenv --python=python3 ../env --prompt='(dreams-env) '"
  echo "  $ . ../env/bin/activate"
  exit 1
fi

set -e

# git clone https://github.com/openai/CLIP
# git clone https://github.com/CompVis/taming-transformers

git clone https://github.com/hackerfriendly/CLIP
git clone https://github.com/hackerfriendly/taming-transformers

pip install -r requirements.txt
pip install -e ./taming-transformers

curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' -o vqgan_imagenet_f16_1024.yaml -C -
curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1' -o vqgan_imagenet_f16_1024.ckpt -C -
curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' -o vqgan_imagenet_f16_16384.yaml -C -
curl -L 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' -o vqgan_imagenet_f16_16384.ckpt -C -
