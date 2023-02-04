'''
dreams.py

A REST API for generating chat bot hallucinations.

Dreams acts as a multiplexer for image generation engines. A request is made to an external API
to generate an image (optionally based on a prompt). When the image is ready, it is copied to
a public facing webserver via scp, and a chat containing the image is posted to the autobus.

Currently only stable diffusion is supported (via stable_diffusion.py), but any number of engines
can easily be added (Stable Diffusion API, DALL-E, Midjourney, etc.)
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, no-member, invalid-name
import os
import random
import tempfile
import uuid
import argparse
import asyncio
import autobus

from pathlib import Path
from subprocess import run

import requests
import uvicorn

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse

from interaction.messages import SendChat

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# This is defined in main()
persyn_config = None

app = FastAPI()

def post_to_autobus(service, channel, prompt, images, bot_name, bot_id):
    ''' Post the completed image notification to autobus '''
    event = SendChat(
        service=service,
        channel=channel,
        images=images,
        msg=prompt,
        bot_name=bot_name,
        bot_id=bot_id
    )
    autobus.publish(event)
    log.info(f"🚌 Image post: {len(images)} sent to autobus")

def upload_files(files):
    ''' scp files to SCPDEST. Expects a Path glob generator. '''
    scpopts = getattr(persyn_config.dreams.upload, 'opts', None)
    if scpopts:
        run(['/usr/bin/scp', scpopts] + [str(f) for f in files] + [persyn_config.dreams.upload.dest_path], check=True)
    else:
        run(['/usr/bin/scp'] + [str(f) for f in files] + [persyn_config.dreams.upload.dest_path], check=True)

def sdd(service, channel, prompt, model, image_id, bot_name, bot_id, style, steps, seed, width, height, guidance): # pylint: disable=unused-argument
    ''' Fetch images from stable_diffusion.py '''
    if not persyn_config.dreams.stable_diffusion.url:
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

    response = requests.post(f"{persyn_config.dreams.stable_diffusion.url}/generate/",
                             params=req, stream=True, timeout=120)

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
        post_to_autobus(service, channel, prompt, [f"{image_id}.jpg"], bot_name, bot_id)

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
    engine: str = None,
    model: str = None,
    service: str = None,
    channel: str = None,
    bot_name: str = None,
    bot_id: str = None,
    style: str = None,
    seed: int = -1,
    steps: int = 50,
    width: int = 512,
    height: int = 512,
    guidance: int = 10
    ):
    ''' Make an image and post it '''
    image_id = uuid.uuid4()

    prompt = prompt.strip().replace('\n', ' ').replace(':', ' ')

    if not prompt:
        prompt = ""

    if style is None:
        style = ""

    prompt = prompt[:max(len(prompt) + len(style), 300)]

    background_tasks.add_task(
        sdd,
        service=service,
        channel=channel,
        prompt=prompt,
        model=model,
        image_id=image_id,
        bot_name=bot_name,
        bot_id=bot_id,
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

async def main():
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
    persyn_config = load_config(args.config_file)

    log.info("🐑 Dreams server starting up")

    uvicorn_config = uvicorn.Config(
        'dreams.dreams:app',
        host=persyn_config.dreams.hostname,
        port=persyn_config.dreams.port,
        workers=persyn_config.dreams.workers,
        reload=False,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    try:
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        await uvicorn_server.serve()
    finally:
        await autobus.stop()

def launch():
    ''' asyncio wrapper to allow launching from pyproject.toml scripts '''
    asyncio.run(main())

if __name__ == "__main__":
    launch()
