'''
dreams.py

A REST API for generating chat bot hallucinations.

Run with
'''
import json
import os
import random
import tempfile
import uuid

from pathlib import Path
from subprocess import run, CalledProcessError
from threading import Lock

import requests

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response

app = FastAPI()

# Maximum iterations
DEFAULT_STEPS = 1000

# Every GPU device that can be used for image generation
GPUS = {
    "0": {"name": "TITAN X", "lock": Lock()},
    "1": {"name": "TITAN X", "lock": Lock()}
}

SCRIPT_PATH = Path(__file__).resolve().parent

def post_to_slack(channel, prompt, image_id):
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
        "token": os.environ['SLACK_BOT_TOKEN'],
        "channel": channel,
        "username": os.environ['BOT_NAME'],
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

def process_prompt(cmd, channel, prompt, image_id, tmpdir):
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
        post_to_slack(channel, prompt, image_id)

def vdiff_cfg(channel, prompt, model, image_id):
    ''' https://github.com/crowsonkb/v-diffusion-pytorch classifier-free guidance '''

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/v-diffusion-pytorch/cfg_sample.py',
            '--out', f'{tmpdir}/{image_id}.jpg',
            '--steps', f'{DEFAULT_STEPS}',
            # Bigger is nice but quite slow (~40 minutes for 500 steps)
            # '--size', '768', '768',
            '--size', '512', '512',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            '--model', model,
            prompt
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir)

def vdiff_clip(channel, prompt, model, image_id):
    ''' https://github.com/crowsonkb/v-diffusion-pytorch '''

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/v-diffusion-pytorch/clip_sample.py',
            '--out', f'{tmpdir}/{image_id}.jpg',
            '--model', model,
            '--steps', f'{DEFAULT_STEPS}',
            '--size', '512', '512',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            prompt
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir)

def vqgan(channel, prompt, model, image_id):
    ''' https://colab.research.google.com/drive/15UwYDsnNeldJFHJ9NdgYBYeo6xPmSelP '''
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/vqgan/vqgan.py',
            '--out', f'{tmpdir}/{image_id}.jpg',
            '--steps', f'{DEFAULT_STEPS}',
            '--size', '720', '480',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            '--vqgan-config', f'models/{model}.yaml',
            '--vqgan-checkpoint', f'models/{model}.ckpt',
            prompt
        ]
        process_prompt(cmd, channel, prompt, image_id, tmpdir)

def stylegan2(channel, prompt, model, image_id):
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
        process_prompt(cmd, channel, prompt, image_id, tmpdir)

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
    channel: str = None
    ):
    ''' Make an image and post it '''
    image_id = uuid.uuid4()

    engines = {
        'v-diffusion-pytorch-cfg': vdiff_cfg,
        'v-diffusion-pytorch-clip': vdiff_clip,
        'vqgan': vqgan,
        'stylegan2': stylegan2
    }

    models = {
        'stylegan2': {
            'ffhq': 'stylegan2-ffhq-config-f.pkl',
            'car': 'stylegan2-car-config-f.pkl',
            'cat': 'stylegan2-cat-config-f.pkl',
            'church': 'stylegan2-church-config-f.pkl',
            'horse': 'stylegan2-horse-config-f.pkl',
            'waifu': '2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl',
            'default': 'stylegan2-ffhq-config-f.pkl'
        },
        'vqgan': {
            'vqgan_imagenet_f16_1024': 'vqgan_imagenet_f16_1024',
            'vqgan_imagenet_f16_16384': 'vqgan_imagenet_f16_16384',
            'default': 'vqgan_imagenet_f16_16384'
        },
        'v-diffusion-pytorch-cfg': {
            'cc12m_1': 'cc12m_1_cfg',
            'default': 'cc12m_1_cfg'
        },
        'v-diffusion-pytorch-clip': {
            'cc12m_1': 'cc12m_1',
            'yfcc_1': 'yfcc_1',
            'yfcc_2': 'yfcc_2',
            'default': 'yfcc_2'
        }
    }

    if engine not in engines:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid engine. Choose one of: {', '.join(list(engines))}"
        )

    if model and model not in models[engine]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model for {engine}. Choose one of: {', '.join(list(models[engine]))}"
        )

    if not model:
        model = 'default'

    background_tasks.add_task(
        engines[engine],
        channel=channel,
        prompt=prompt.strip(),
        model=models[engine][model],
        image_id=image_id
    )

    return {
        "engine": engine,
        "model": model,
        "image_id": image_id
    }
