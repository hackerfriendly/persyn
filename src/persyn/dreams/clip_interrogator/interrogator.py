#!/usr/bin/env python3
'''
clip-interrigator by pharmapsychotic, https://github.com/pharmapsychotic/clip-interrogator

Adapted to FastAPI by hackerfriendly
'''
# pylint: disable=import-error, no-name-in-module, no-member, wrong-import-position
import hashlib
import math
import os
import pickle
import base64
import argparse

from io import BytesIO

import numpy as np
import requests
import torch
import uvicorn

from models.blip import blip_decoder
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from pydantic import BaseModel

import clip

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

CHUNK_SIZE = 2048
BLIP_IMAGE_EVAL_SIZE = 384
CLIP_MODEL_NAME = 'ViT-L/14'
# ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading BLIP model...")
blip_model = blip_decoder(
    pretrained='env/src/blip/models/model_large_caption.pth',
    image_size=BLIP_IMAGE_EVAL_SIZE,
    vit='large',
    med_config='env/src/blip/configs/med_config.json'
)
blip_model.eval()
blip_model = blip_model.to(device)

print("Loading CLIP model...")

clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device="cuda")
clip_model.cuda().eval()

DATA_PATH = 'data'

class LabelTable():
    ''' LabelTable. This is horrible and needs a complete rewrite. '''
    def __init__(self, labels, desc):
        self.labels = labels
        self.embeds = []

        sha = hashlib.sha256(",".join(labels).encode()).hexdigest()

        os.makedirs('./cache', exist_ok=True)
        cache_filepath = f"./cache/{desc}.pkl"
        if desc is not None and os.path.exists(cache_filepath):
            with open(cache_filepath, 'rb') as f:
                data = pickle.load(f)
                if data.get('hash') == sha and data.get('model') == CLIP_MODEL_NAME:
                    self.labels = data['labels']
                    self.embeds = data['embeds']

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/CHUNK_SIZE))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None):
                text_tokens = clip.tokenize(chunk).cuda()
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            with open(cache_filepath, 'wb') as f:
                pickle.dump({"labels":self.labels, "embeds":self.embeds, "hash":sha, "model":CLIP_MODEL_NAME}, f)

    def _rank(self, image_features, text_embeds, top_count=1):
        top_count = min(top_count, len(text_embeds))
        sim = torch.zeros((1, len(text_embeds))).to(device)
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).float().to(device)
        for i in range(image_features.shape[0]):
            sim += (image_features[i].unsqueeze(0) @ text_embeds.T).softmax(dim=-1)
        _, top_labels = sim.cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features, top_count=1):
        ''' rank labels '''
        if len(self.labels) <= CHUNK_SIZE:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/CHUNK_SIZE))
        keep_per_chunk = int(CHUNK_SIZE / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks)):
            start = chunk_idx*CHUNK_SIZE
            stop = min(start+CHUNK_SIZE, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]

def generate_caption(pil_image):
    gpu_image = transforms.Compose([
        transforms.Resize((BLIP_IMAGE_EVAL_SIZE, BLIP_IMAGE_EVAL_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

def rank_top(image_features, text_array):
    text_tokens = clip.tokenize([text for text in text_array]).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    sim = torch.zeros((1, len(text_array)), device=device)
    for i in range(image_features.shape[0]):
        sim += (image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)

    _, top_labels = sim.cpu().topk(1, dim=-1)
    return text_array[top_labels[0][0].numpy()]

def similarity(image_features, text):
    text_tokens = clip.tokenize([text]).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity[0][0]

def load_list(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def interrogate(image):
    caption = generate_caption(image)

    images = clip_preprocess(image).unsqueeze(0).cuda()
    with torch.no_grad():
        image_features = clip_model.encode_image(images).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # flaves = flavors.rank(image_features, 2048)
    best_medium = mediums.rank(image_features, 1)[0]
    # best_artist = artists.rank(image_features, 1)[0]
    # best_trending = trendings.rank(image_features, 1)[0]
    # best_movement = movements.rank(image_features, 1)[0]

    best_prompt = caption
    best_sim = similarity(image_features, best_prompt)

    def check(addition):
        nonlocal best_prompt, best_sim
        prompt = best_prompt + ", " + addition
        sim = similarity(image_features, prompt)
        if sim > best_sim:
            best_sim = sim
            best_prompt = prompt
            return True
        return False

    def check_multi_batch(opts):
        nonlocal best_prompt, best_sim
        prompts = []
        for i in range(2**len(opts)):
            prompt = best_prompt
            for bit in range(len(opts)):
                if i & (1 << bit):
                    prompt += ", " + opts[bit]
            prompts.append(prompt)

        t = LabelTable(prompts, None)
        best_prompt = t.rank(image_features, 1)[0]
        best_sim = similarity(image_features, best_prompt)

    # check_multi_batch([best_medium, best_artist, best_trending, best_movement])
    check_multi_batch([best_medium])

    return best_prompt

    # extended_flavors = set(flaves)
    # for _ in tqdm(range(25), desc="Flavor chain"):
    #     try:
    #         best = rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
    #         flave = best[len(best_prompt)+2:]
    #         if not check(flave):
    #             break
    #         extended_flavors.remove(flave)
    #     except:
    #         # exceeded max prompt length
    #         break

    # return best_prompt

trending_list = [
    "Artstation",
    "behance",
    "cg society",
    "cgsociety",
    "deviantart",
    "dribble",
    "flickr",
    "instagram",
    "pexels",
    "pinterest",
    "pixabay",
    "pixiv",
    "polycount",
    "reddit",
    "shutterstock",
    "tumblr",
    "unsplash",
    "zbrush central"
]
trending_list.extend(["trending on "+site for site in trending_list])
trending_list.extend(["featured on "+site for site in trending_list])
trending_list.extend([site+" contest winner" for site in trending_list])

raw_artists = load_list(f'{DATA_PATH}/artists.txt')
artists = [f"by {a}" for a in raw_artists]
artists.extend([f"inspired by {a}" for a in raw_artists])

artists = LabelTable(artists, "artists")
flavors = LabelTable(load_list(f'{DATA_PATH}/flavors.txt'), "flavors")
mediums = LabelTable(load_list(f'{DATA_PATH}/mediums.txt'), "mediums")
movements = LabelTable(load_list(f'{DATA_PATH}/movements.txt'), "movements")
trendings = LabelTable(trending_list, "trendings")

app = FastAPI()

class ImageToCaption(BaseModel):
    data: str

@app.get("/", status_code=302)
async def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.post("/caption/")
async def caption(img: ImageToCaption):
    ''' Generate a caption with clip-interrogator '''

    if img.data.startswith('http://') or img.data.startswith('https://'):
        image = Image.open(requests.get(img.data, stream=True, timeout=30).raw).convert('RGB')
    else:
        buf = BytesIO()
        buf.write(base64.b64decode(img.data))
        buf.seek(0)

        image = Image.open(buf).convert('RGB')

    return { "caption": interrogate(image) }


def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Picture captions by interrogator.py.'''
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

    log.info("🖼 Caption server starting up")

    uvicorn.run(
        'dreams.interrogator:app',
        host=persyn_config.dreams.captions.hostname,
        port=persyn_config.dreams.captions.port,
        workers=persyn_config.dreams.captions.workers,
        reload=False,
    )

if __name__ == '__main__':
    main()
