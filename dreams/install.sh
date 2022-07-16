#!/bin/bash
set -e

virtualenv --python=python3.8 env --prompt='(dreams-env) '
. env/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

# VQGAN by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
cd vqgan/ && ./install.sh && cd -

# v-diffusion-pytorch, also by Katherine Crowson
git clone https://github.com/hackerfriendly/v-diffusion-pytorch
cd v-diffulsion-pytorch/ && ./install.sh && cd -

# StyleGAN2, https://github.com/NVlabs/stylegan2
git clone https://github.com:hackerfriendly/stylegan2
cd stylegan2/ && ./install.sh && cd -

