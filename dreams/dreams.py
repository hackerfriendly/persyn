'''
dreams.py

A REST API for generating chat bot hallucinations.
'''
import json
import os
import random
import tempfile
import uuid

from pathlib import Path
from subprocess import run, CalledProcessError
from threading import Lock

import urllib
import urllib.request
import requests

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response

from dalle2 import Dalle2

app = FastAPI()

# Every GPU device that can be used for image generation
GPUS = {
    "0": {"name": "TITAN X", "lock": Lock()},
    # "1": {"name": "TITAN X", "lock": Lock()}
}

SCRIPT_PATH = Path(__file__).resolve().parent

def post_to_slack(channel, prompt, image_id, slack_bot_token, bot_name):
    ''' Post the image URL to Slack '''
    blocks = [
        {
            "type": "image",
            "title": {
                "type": "plain_text",
                "text": prompt
            },
            "image_url" : f"{os.environ['BASEURL']}/{image_id}.jpg",
            "alt_text": prompt
        }
    ]
    req = {
        "token": slack_bot_token,
        "channel": channel,
        "username": bot_name,
        "text": prompt,
        "blocks": json.dumps(blocks)
    }
    reply = requests.post('https://slack.com/api/chat.postMessage', data=req)
    print(reply.status_code, reply.text)

def upload_files(files):
    ''' scp files to SCPDEST. Expects a Path glob generator. '''
    scpopts = os.environ.get('SCPOPTS', None)
    if scpopts:
        run(['/usr/bin/scp', scpopts] + [str(f) for f in files] + [os.environ['SCPDEST']], check=True)
    else:
        run(['/usr/bin/scp'] + [str(f) for f in files] + [os.environ['SCPDEST']], check=True)

def wait_for_gpu():
    ''' Return the device name of first available GPU. Blocks until one is available and sets the lock. '''
    while True:
        gpu = random.choice(list(GPUS))
        if GPUS[gpu]['lock'].acquire(timeout=1):
            return gpu

def process_prompt(cmd, channel, prompt, image_id, tmpdir, slack_bot_token, bot_name):
    ''' Generate the image files, upload them, post to Slack, and clean up. '''
    try:
        gpu = wait_for_gpu()
        try:
            print("GPU:", gpu)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            run(cmd, check=True, env=env)
        finally:
            GPUS[gpu]['lock'].release()

        upload_files(Path(tmpdir).glob('*'))

    except CalledProcessError:
        return

    if channel:
        post_to_slack(channel, prompt, image_id, slack_bot_token, bot_name)

def vdiff_cfg(channel, prompt, model, image_id, steps, slack_bot_token, bot_name):
    ''' https://github.com/crowsonkb/v-diffusion-pytorch classifier-free guidance '''

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/v-diffusion-pytorch/cfg_sample.py',
            '--out', f'{tmpdir}/{image_id}.jpg',
            '--steps', f'{steps}',
            # Bigger is nice but quite slow (~40 minutes for 500 steps)
            # '--size', '768', '768',
            '--size', '512', '512',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            '--model', model,
            '--style', 'random',
            prompt[:250]
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir, slack_bot_token, bot_name)

def vdiff_clip(channel, prompt, model, image_id, steps, slack_bot_token, bot_name):
    ''' https://github.com/crowsonkb/v-diffusion-pytorch '''

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/v-diffusion-pytorch/clip_sample.py',
            '--out', f'{tmpdir}/{image_id}.jpg',
            '--model', model,
            '--steps', f'{steps}',
            '--size', '384', '512',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            prompt[:250]
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir, slack_bot_token, bot_name)

def vqgan(channel, prompt, model, image_id, steps, slack_bot_token, bot_name):
    ''' https://colab.research.google.com/drive/15UwYDsnNeldJFHJ9NdgYBYeo6xPmSelP '''
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/vqgan/vqgan.py',
            '--out', f'{tmpdir}/{image_id}.jpg',
            '--steps', f'{steps}',
            '--size', '720', '480',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            '--vqgan-config', f'models/{model}.yaml',
            '--vqgan-checkpoint', f'models/{model}.ckpt',
            prompt[:250]
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir, slack_bot_token, bot_name)

def stylegan2(channel, prompt, model, image_id, slack_bot_token, bot_name):
    ''' https://github.com/NVlabs/stylegan2 '''
    psi = random.uniform(0.6, 0.9)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/stylegan2/go-stylegan2',
            f'models/{model}',
            str(random.randint(0, 2**32 - 1)),
            str(psi),
            f'{tmpdir}/{image_id}.jpg'
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir, slack_bot_token, bot_name)

def dalle2(channel, prompt, model, image_id, slack_bot_token, bot_name): # pylint: disable=unused-argument
    ''' https://openai.com/dall-e-2/ (pre-release) '''
    with tempfile.TemporaryDirectory() as tmpdir:

        token = os.environ.get('DALLE2_TOKEN', None)
        if not token:
            raise HTTPException(
                status_code=400,
                detail="No DALLE2_TOKEN defined. Check your config."
            )

        dalle = Dalle2(token)
        generations = dalle.generate(prompt)

        if not generations:
            raise HTTPException(
                status_code=400,
                detail="No DALLE2 generations returned"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = generations[0] # TODO: download everything in the batch
            imageurl = gen["generation"]["image_path"]

            urllib.request.urlretrieve(imageurl, f"{tmpdir}/{image_id}.jpg")
            upload_files(Path(tmpdir).glob('*'))

        if channel:
            post_to_slack(channel, prompt, image_id, slack_bot_token, bot_name)

def latent_diffusion(channel, prompt, model, image_id, slack_bot_token, bot_name): # pylint: disable=unused-argument
    ''' https://github.com/hackerfriendly/latent-diffusion '''
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/latent-diffusion/go-ld',
            f'{tmpdir}/{image_id}.jpg',
            prompt[:250]
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir, slack_bot_token, bot_name)

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "dreams server. Try /docs"}

@app.get("/view/{image_id}")
async def image_url(image_id):
    ''' Redirect to BASEURL '''
    response = Response(status_code=301)
    response.headers['Location'] = f"{os.environ['BASEURL']}/{image_id}.jpg"
    return response

@app.post("/generate/")
async def generate(
    prompt: str,
    background_tasks: BackgroundTasks,
    engine: str = 'v-diffusion-pytorch-cfg',
    model: str = None,
    channel: str = None,
    slack_bot_token: str = None,
    bot_name: str = None
    ):
    ''' Make an image and post it '''
    image_id = uuid.uuid4()

    engines = {
        'v-diffusion-pytorch-cfg': vdiff_cfg,
        'v-diffusion-pytorch-clip': vdiff_clip,
        'vqgan': vqgan,
        'stylegan2': stylegan2,
        'latent-diffusion': latent_diffusion,
        'dalle2': dalle2
    }

    models = {
        'stylegan2': {
            'ffhq': {'name': 'stylegan2-ffhq-config-f.pkl'},
            'car': {'name': 'stylegan2-car-config-f.pkl'},
            'cat': {'name': 'stylegan2-cat-config-f.pkl'},
            'church': {'name': 'stylegan2-church-config-f.pkl'},
            'horse': {'name': 'stylegan2-horse-config-f.pkl'},
            'waifu': {'name': '2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl'},
            'default': {'name': 'stylegan2-ffhq-config-f.pkl'}
        },
        'dalle2': {
            'default': {'name': 'default'}
        },
        'vqgan': {
            'vqgan_imagenet_f16_1024': {'name': 'vqgan_imagenet_f16_1024', 'steps': 500},
            'vqgan_imagenet_f16_16384': {'name': 'vqgan_imagenet_f16_16384', 'steps': 500},
            'default': {'name': 'vqgan_imagenet_f16_16384', 'steps': 500}
        },
        'v-diffusion-pytorch-cfg': {
            'cc12m_1_cfg': {'name': 'cc12m_1_cfg', 'steps': 50},
            'default': {'name': 'cc12m_1_cfg', 'steps': 50}
        },
        'v-diffusion-pytorch-clip': {
            'cc12m_1': {'name': 'cc12m_1', 'steps': 300},
            'yfcc_1': {'name': 'yfcc_1', 'steps': 300},
            'yfcc_2': {'name': 'yfcc_2', 'steps': 300},
            'default': {'name': 'yfcc_2', 'steps': 300}
        },
        'latent-diffusion': {
            'default': {'name': 'text2img-large'}
        }
    }

    if engine not in engines:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid engine {engine}. Choose one of: {', '.join(list(engines))}"
        )

    if model and model not in models[engine]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model {model} for {engine}. Choose one of: {', '.join(list(models[engine]))}"
        )

    if not model:
        model = 'default'

    prompt = prompt.strip().replace('\n', ' ').replace(':', ' ')

    if not prompt:
        prompt = "Untitled"

    if engine in ['stylegan2', 'latent-diffusion', 'dalle2']:
        background_tasks.add_task(
            engines[engine],
            channel=channel,
            prompt=prompt,
            model=models[engine][model]['name'],
            image_id=image_id,
            slack_bot_token=slack_bot_token,
            bot_name=bot_name
        )
    else:
        background_tasks.add_task(
            engines[engine],
            channel=channel,
            prompt=prompt,
            model=models[engine][model]['name'],
            image_id=image_id,
            steps=models[engine][model]['steps'],
            slack_bot_token=slack_bot_token,
            bot_name=bot_name
        )

    return {
        "engine": engine,
        "model": model,
        "image_id": image_id
    }
