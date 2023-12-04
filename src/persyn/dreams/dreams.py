'''
dreams.py

A REST API for generating chat bot hallucinations.

Dreams acts as a multiplexer for image generation engines. A request is made to an external API
to generate an image (optionally based on a prompt). When the image is ready, it is copied to
a public facing webserver via scp or to an S3 bucket, and a chat containing the image is posted
to the autobus.

Currently only DALL-E is supported.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, no-member, invalid-name
import os
import tempfile
import uuid
import argparse
import asyncio

from pathlib import Path
from subprocess import run

import requests
import uvicorn
import boto3

from PIL import Image
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse

from openai import OpenAI

from persyn import autobus

from persyn.interaction.messages import SendChat

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

# This is defined in main()
persyn_config = None

oai_client = None

rs = requests.Session()

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

def upload_files(files, config=None):
    ''' Upload files via scp or s3. Expects a Path glob generator. '''
    if config is None:
        config = persyn_config

    scpopts = getattr(config.dreams.upload, 'opts', None)
    bucket = getattr(config.dreams.upload, 'bucket', None)
    prefix = getattr(config.dreams.upload, 'dest_path', '')

    if bucket:
        for file in files:
            s3_client = boto3.client('s3')
            try:
                s3_client.upload_file(file, bucket, f'{prefix}{os.path.basename(file)}')
            except ClientError as e:
                log.error(e)
            continue
        return

    # no bucket. Use scp instead.
    if scpopts:
        run(['/usr/bin/scp', scpopts] + [str(f) for f in files] + [config.dreams.upload.dest_path], check=True)
    else:
        run(['/usr/bin/scp'] + [str(f) for f in files] + [config.dreams.upload.dest_path], check=True)

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

    response = rs.post(f"{persyn_config.dreams.stable_diffusion.url}/generate/",
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

def dalle(service, channel, prompt, model, image_id, bot_name, bot_id, style, steps, seed, width, height, guidance): # pylint: disable=unused-argument
    ''' Fetch images from OpenAI '''

    if not model:
        model = "dall-e-3"
    if not width:
        width = persyn_config.dreams.dalle.width
    if not height:
        height = persyn_config.dreams.dalle.height
    if not style:
        style = "standard"

    response = oai_client.images.generate(
        model=model,
        prompt=prompt,
        size=f"{width}x{height}",
        quality=style,
        n=1,
    )

    response = rs.get(response.data[0].url, stream=True)
    response.raise_for_status()

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = str(Path(tmpdir)/f"{image_id}.jpg")
        with open(fname, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        image = Image.open(fname)

        # Set the EXIF data. See PIL.ExifTags.TAGS to map numbers to names.
        exif = image.getexif()
        exif[271] = prompt # exif: Make
        exif[272] = model # exif: Model
        exif[305] = f'width={width}, height={height}, quality={style}' # exif: Software

        image.save(fname, format="JPEG", quality=85, exif=exif)

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
    engine: str = 'dall-e',
    model: str = None,
    service: str = None,
    channel: str = None,
    bot_name: str = None,
    bot_id: str = None,
    style: str = None,
    seed: int = -1,
    steps: int = 50,
    width: int = None,
    height: int = None,
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

    engines = {
        'dall-e': dalle,
        'stable-diffusion': sdd
    }

    background_tasks.add_task(
        engines[engine],
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
    global oai_client

    persyn_config = load_config(args.config_file)

    log.info("🐑 Dreams server starting up")

    oai_client = OpenAI(
        api_key=persyn_config.completion.api_key,
        organization=persyn_config.completion.openai_org
    )

    uvicorn_config = uvicorn.Config(
        'persyn.dreams.dreams:app',
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
