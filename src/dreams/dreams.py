'''
dreams.py

A REST API for generating chat bot hallucinations.

TODO:
This daemon was conceived in an earlier time (spring 2022), when many image engines roamed the earth,
before Stable Diffusion rose to dominate them all.

It needs a complete overhaul to make it easier to support local engines and external APIs.

Notably, all local rendering is currently broken.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, no-member, invalid-name
import json
import os
import random
import tempfile
import uuid
import argparse

from pathlib import Path
from subprocess import run, CalledProcessError

import boto3
import requests
import uvicorn

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# These are defined in main()
persyn_config = None
sqs = None

app = FastAPI()

SCRIPT_PATH = Path(__file__).resolve().parent

def post_to_queue(service, channel, queue, prompt, images, bot_name):
    ''' Post the completed image notification to SQS '''
    if queue is None:
        log.warning("No queue provided, not posting image.")
        return

    data = {
        "event_type": "image-ready",
        "service": service,
        "channel": channel,
        "images": images,
        "caption": prompt,
        "bot_name": bot_name,
    }
    qurl = sqs.get_queue_by_name(QueueName=queue)
    response = qurl.send_message(MessageBody=json.dumps(data))
    log.info(f"Posted {response['MessageId']} to queue {queue}")

def upload_files(files):
    ''' scp files to SCPDEST. Expects a Path glob generator. '''
    scpopts = getattr(persyn_config.dreams.upload, 'opts', None)
    if scpopts:
        run(['/usr/bin/scp', scpopts] + [str(f) for f in files] + [persyn_config.dreams.upload.dest_path], check=True)
    else:
        run(['/usr/bin/scp'] + [str(f) for f in files] + [persyn_config.dreams.upload.dest_path], check=True)

def wait_for_gpu():
    ''' Return the device name of first available GPU. Blocks until one is available and sets the lock. '''
    while True:
        gpu = random.choice(list(persyn_config.dreams.gpus))
        if persyn_config.dreams.gpus[gpu]['lock'].acquire(timeout=1):
            return gpu

def process_prompt(cmd, service, channel, queue, prompt, images, tmpdir, bot_name):
    ''' Generate the image files, upload them, post to Slack, and clean up. '''
    try:
        gpu = wait_for_gpu()
        try:
            print("GPU:", gpu)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            run(cmd, check=True, env=env)
        finally:
            persyn_config.dreams.gpus[gpu]['lock'].release()

        upload_files(Path(tmpdir).glob('*'))

    except CalledProcessError:
        return

    if service:
        post_to_queue(service, channel, queue, prompt, images, bot_name)

def vdiff_cfg(service, channel, prompt, queue, model, image_id, steps, bot_name):
    ''' https://github.com/crowsonkb/v-diffusion-pytorch classifier-free guidance '''

    image = f"{image_id}.jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/v-diffusion-pytorch/cfg_sample.py',
            '--out', f'{tmpdir}/{image}',
            '--steps', f'{steps}',
            # Bigger is nice but quite slow (~40 minutes for 500 steps)
            # '--size', '768', '768',
            '--size', '512', '512',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            '--model', model,
            '--style', 'random',
            prompt[:250]
        ]
        process_prompt(cmd, service, channel, queue, prompt, [image], tmpdir, bot_name)

def vdiff_clip(service, channel, prompt, queue, model, image_id, steps, bot_name):
    ''' https://github.com/crowsonkb/v-diffusion-pytorch '''

    image = f"{image_id}.jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/v-diffusion-pytorch/clip_sample.py',
            '--out', f'{tmpdir}/{image}',
            '--model', model,
            '--steps', f'{steps}',
            '--size', '384', '512',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            prompt[:250]
        ]
        process_prompt(cmd, service, channel, queue, prompt, [image], tmpdir, bot_name)

def vqgan(service, channel, prompt, queue, model, image_id, steps, bot_name):
    ''' https://colab.research.google.com/drive/15UwYDsnNeldJFHJ9NdgYBYeo6xPmSelP '''
    image = f"{image_id}.jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/vqgan/vqgan.py',
            '--out', f'{tmpdir}/{image}',
            '--steps', f'{steps}',
            '--size', '720', '480',
            '--seed', f'{random.randint(0, 2**64 - 1)}',
            '--vqgan-config', f'models/{model}.yaml',
            '--vqgan-checkpoint', f'models/{model}.ckpt',
            prompt[:250]
        ]
        process_prompt(cmd, service, channel, queue, prompt, [image], tmpdir, bot_name)

def stylegan2(service, channel, prompt, queue, model, image_id, bot_name, style): # pylint: disable=unused-argument
    ''' https://github.com/NVlabs/stylegan2 '''
    image = f"{image_id}.jpg"
    psi = random.uniform(0.6, 0.9)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/stylegan2/go-stylegan2',
            f'models/{model}',
            str(random.randint(0, 2**32 - 1)),
            str(psi),
            f'{tmpdir}/{image}'
        ]
        process_prompt(cmd, service, channel, queue, prompt, [image], tmpdir, bot_name)

def sdd(service, channel, prompt, queue, model, image_id, bot_name, style, steps, seed, width, height, guidance): # pylint: disable=unused-argument
    ''' Fetch images from stable_diffusion.py '''
    url = getattr(persyn_config.dreams.stable_diffusion, 'url', None)
    if not url:
        raise HTTPException(
            status_code=400,
            detail="dreams.stable_diffusion.url not defined. Check your config."
        )

    req = {
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "guidance": guidance
    }

    response = requests.post(f"{url}/generate/", params=req, stream=True, timeout=120)

    if not response.ok:
        raise HTTPException(
            status_code=400,
            detail="Generate failed!"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = str(Path(tmpdir)/f"{image_id}.jpg")
        with open(fname, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        upload_files([fname])

    if service:
        post_to_queue(service, channel, queue, prompt, [f"{image_id}.jpg"], bot_name)


def stable_diffusion(service, channel, prompt, queue, model, image_id, bot_name, style): # pylint: disable=unused-argument
    ''' https://github.com/hackerfriendly/stable-diffusion '''
    image = f"{image_id}.jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/stable-diffusion/go-sd',
            '-o', f'{tmpdir}/{image}',
            '-p', prompt,
            '-t', style
        ]
        process_prompt(cmd, service, channel, queue, prompt, [image], tmpdir, bot_name)

def latent_diffusion(service, channel, prompt, queue, model, image_id, bot_name, style): # pylint: disable=unused-argument
    ''' https://github.com/hackerfriendly/latent-diffusion '''
    image = f"{image_id}.jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            f'{SCRIPT_PATH}/latent-diffusion/go-ld',
            f'{tmpdir}/{image}',
            prompt[:250]
        ]
        process_prompt(cmd, service, channel, queue, prompt, [image], tmpdir, bot_name)

@app.get("/", status_code=302)
async def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.get("/view/{image_id}")
async def image_url(image_id):
    ''' Redirect to BASEURL '''
    response = Response(status_code=301)
    response.headers['Location'] = f"{persyn_config.dreams.upload.url_base}/{image_id}.jpg"
    return response

@app.post("/generate/")
def generate(
    prompt: str,
    background_tasks: BackgroundTasks,
    engine: str = 'sdd',
    model: str = None,
    service: str = None,
    channel: str = None,
    queue: str = None,
    bot_name: str = None,
    style: str = None,
    seed: int = -1,
    steps: int = 50,
    width: int = 512,
    height: int = 512,
    guidance: int = 10
    ):
    ''' Make an image and post it '''
    image_id = uuid.uuid4()

    engines = {
        'v-diffusion-pytorch-cfg': vdiff_cfg,
        'v-diffusion-pytorch-clip': vdiff_clip,
        'vqgan': vqgan,
        'stylegan2': stylegan2,
        'latent-diffusion': latent_diffusion,
        'stable-diffusion': sdd,
        'sdd': sdd
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
        },
        'stable-diffusion': {
            'default': {'name': 'stable-diffusion'}
        },
        'sdd': {
            'default': {'name': 'sdd'}
        }
    }

    if engine == "dalle2":
        engine = "stable-diffusion"

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
        prompt = ""

    if style is None:
        style = ""

    prompt = prompt[:max(len(prompt) + len(style), 300)]

    if engine in ['stylegan2', 'latent-diffusion', 'dalle2']:
        background_tasks.add_task(
            engines[engine],
            service=service,
            channel=channel,
            queue=queue,
            prompt=prompt,
            model=models[engine][model]['name'],
            image_id=image_id,
            bot_name=bot_name,
            style=style
        )
    else:
        background_tasks.add_task(
            engines[engine],
            service=service,
            channel=channel,
            queue=queue,
            prompt=prompt,
            model=models[engine][model]['name'],
            image_id=image_id,
            bot_name=bot_name,
            style=style,
            steps=steps,
            seed=seed,
            width=width,
            height=height,
            guidance=guidance
        )

    return {
        "engine": engine,
        "model": model,
        "image_id": image_id
    }

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Persyn dreams server. Do persyns dream of electric sheep?'''
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

    global persyn_config
    global sqs
    persyn_config = load_config(args.config_file)
    sqs = boto3.resource('sqs', region_name=getattr(persyn_config.id, 'aws_region', None))

    log.info("😴 Dreams server starting up")

    uvicorn.run(
        'dreams.dreams:app',
        host=persyn_config.dreams.hostname,
        port=persyn_config.dreams.port,
        workers=persyn_config.dreams.workers,
        reload=False,
    )

if __name__ == "__main__":
    main()
