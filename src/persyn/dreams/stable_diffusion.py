'''
stable_diffusion.py: Stable Diffusion daemon. Pre-load the model and serve image prompts via FastAPI.

This fetches SD from Hugging Face, so huggingface-cli login first.

Rendering time is about 10 it/s (~4 seconds for 40 steps) for 512x512 on an RTX 2080.
'''
# pylint: disable=no-member
import random
import argparse
import gc
import os

from threading import Lock
from typing import Optional
from io import BytesIO

import torch
import numpy as np

import uvicorn

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse

from transformers import AutoFeatureExtractor, logging
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

app = FastAPI()

# defined after the config is loaded
MODEL = None

PIPE = None

# One lock for each available GPU (only one supported for now)
GPUS = {}
GPUS[0] = Lock()

if not GPUS:
    raise RuntimeError("No GPUs detected. Check your config and try again.")

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# The CompVis safety model.
safety_feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")


def naughty(image):
    ''' Returns True if naughty bits are detected, else False. '''
    imgarray = np.asarray(image)
    safety_checker_input = safety_feature_extractor([imgarray], return_tensors="pt")
    _, has_nsfw_concept = safety_checker(images=[imgarray], clip_input=safety_checker_input.pixel_values)
    return has_nsfw_concept[0]

def wait_for_gpu():
    ''' Return the device name of first available GPU. Blocks until one is available and sets the lock. '''
    while True:
        gpu = random.choice(list(GPUS))
        if GPUS[gpu].acquire(timeout=0.5):
            return gpu

def clear_cuda_mem():
    ''' Try to recover from CUDA OOM '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception as e:
            pass

    gc.collect()
    torch.cuda.empty_cache()

def round_to(num, mult=8):
    ''' Round to the nearest multiple of mult '''
    ret = num + (mult - 1)
    return ret - (ret % mult)

def generate_image(prompt, seed, steps, width, height, guidance, negative_prompt):
    ''' Generate and return an image array using the first available GPU '''
    gpu = wait_for_gpu()

    try:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        return PIPE(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps,
            height=round_to(height),
            width=round_to(width),
            guidance_scale=guidance
        ).images[0]

    except RuntimeError:
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        raise HTTPException(
            status_code=507,
            detail="Out of CUDA memory. Try smaller values for width and height."
        )

    finally:
        clear_cuda_mem()
        GPUS[gpu].release()


def safe_generate_image(prompt, seed, steps, width, height, guidance, safe=True, negative_prompt=""):
    ''' Generate an image and check NSFW. Returns a FastAPI StreamingResponse. '''

    image = generate_image(prompt, seed, steps, width, height, guidance, negative_prompt)

    if safe and naughty(image):
        print("🍆 detected!!!1!")
        prompt = "An adorable teddy bear running through a grassy field, early morning volumetric lighting"
        image = generate_image(prompt, seed, steps, width, height, guidance, negative_prompt)

    # Set the EXIF data. See PIL.ExifTags.TAGS to map numbers to names.
    exif = image.getexif()
    exif[271] = prompt # exif: Make
    exif[272] = MODEL # exif: Model
    exif[305] = f'seed={seed}, steps={steps}, negative_prompt={negative_prompt}' # exif: Software

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85, exif=exif)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg", headers={
        'Content-Disposition': 'inline; filename="synthesis.jpg"'}
    )

@app.get("/", status_code=302)
async def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.post("/generate/")
def generate(
    prompt: Optional[str] = Query(""),
    seed: Optional[int] = Query(-1),
    steps: Optional[int] = Query(ge=1, le=100, default=50),
    width: Optional[int] = Query(1024),
    height: Optional[int] = Query(512),
    guidance: Optional[float] = Query(ge=2, le=20, default=6),
    safe: Optional[bool] = Query(True),
    negative_prompt: Optional[str] = Query(
            "text, logo, words, worst quality, low quality, deformed iris, deformed pupils, bad eyes, cross eyed, poorly drawn face, cloned face, extra fingers, mutated hands, fused fingers, too many fingers, missing arms, missing legs, extra arms, extra legs, poorly drawn hands, bad anatomy, bad proportions, cropped, lowres, jpeg artifacts, signature, watermark, username, artist name, trademark, watermark, title, multiple view, Reference sheet, long neck, Out of Frame"
        ),
    ):
    ''' Generate an image with Stable Diffusion '''

    if seed < 0:
        seed = random.randint(0,2**64-1)

    prompt = prompt.strip().replace('\n', ' ')

    torch.cuda.empty_cache()

    return safe_generate_image(prompt, seed, steps, width, height, guidance, safe, negative_prompt)

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''stable_diffusion server.'''
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

    global MODEL
    global PIPE

    MODEL = persyn_config.dreams.stable_diffusion.model

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")

    if MODEL.startswith("stabilityai/stable-diffusion-xl"):
        PIPE = DiffusionPipeline.from_pretrained(
            MODEL,
            # scheduler=scheduler,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        PIPE.enable_model_cpu_offload()

    # use fp16 for ~3x speedup (if available)
    elif MODEL.startswith("stabilityai/stable-diffusion"):
        PIPE = StableDiffusionPipeline.from_pretrained(
            MODEL,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            # vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        )
        PIPE = PIPE.to("cuda")

    else:
        PIPE = StableDiffusionPipeline.from_pretrained(MODEL, scheduler=scheduler)
        PIPE = PIPE.to("cuda")

    log.info(f"🎨 Stable Diffusion server starting up, serving model: {MODEL}")

    uvicorn.run(
        'persyn.dreams.stable_diffusion:app',
        host=persyn_config.dreams.stable_diffusion.hostname,
        port=persyn_config.dreams.stable_diffusion.port,
        workers=persyn_config.dreams.stable_diffusion.workers,
        reload=False,
    )

if __name__ == "__main__":
    main()
