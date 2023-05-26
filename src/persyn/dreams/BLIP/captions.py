#!/usr/bin/env python3
'''
BLIP, https://github.com/salesforce/BLIP

Adapted to FastAPI by hackerfriendly
'''
# pylint: disable=import-error, no-name-in-module, no-member, wrong-import-position
from persyn.dreams.BLIP.models.blip import blip_decoder
import os
import base64
import argparse

from io import BytesIO

import requests
import torch
import uvicorn

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from pydantic import BaseModel

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

CHUNK_SIZE = 2048
BLIP_IMAGE_EVAL_SIZE = 384

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

med_config = {
    "architectures": [
        "BertModel"
    ],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "type_vocab_size": 2,
    "vocab_size": 30524,
    "encoder_width": 768,
    "add_cross_attention": True
}

print("Loading BLIP model...")
# model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
# vit = 'base'
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
vit = 'large'

blip_model = blip_decoder(pretrained=model_url, image_size=BLIP_IMAGE_EVAL_SIZE,
                          vit=vit, med_config=med_config)
blip_model.eval()
blip_model = blip_model.to(device)


def load_image(raw_image):

    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def generate_caption(img_url):
    image = load_image(img_url)
    with torch.no_grad():
        # beam search
        #caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        cap = blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        log.info("🖼️ ", cap[0])
        return cap[0]

app = FastAPI()

class ImageToCaption(BaseModel):
    data: str

@app.get("/", status_code=302)
async def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.post("/caption/")
async def caption(img: ImageToCaption):
    ''' Generate a caption with BLIP '''

    if img.data.startswith('http://') or img.data.startswith('https://'):
        image = Image.open(requests.get(img.data, stream=True, timeout=30).raw).convert('RGB')
    else:
        buf = BytesIO()
        buf.write(base64.b64decode(img.data))
        buf.seek(0)

        image = Image.open(buf).convert('RGB')

    return { "caption": generate_caption(image) }


def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Picture captions by BLIP.'''
    )
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    # parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    persyn_config = load_config(args.config_file)

    log.info("🖼  Caption server starting up")

    uvicorn.run(
        'persyn.dreams.BLIP.captions:app',
        host=persyn_config.dreams.captions.hostname,
        port=persyn_config.dreams.captions.port,
        workers=persyn_config.dreams.captions.workers,
        reload=False,
    )

if __name__ == '__main__':
    main()
