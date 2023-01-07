'''
Prompt Partot by Stephen Young, https://replicate.com/kyrick/prompt-parrot

I'm allergic to Docker and cog didn't work, so here it is via FastAPI + virtualenv.
'''
# pylint: disable=invalid-name

import itertools
import torch

# Color logging
from color_logging import ColorLog

from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse

from transformers import AutoModelForCausalLM, AutoTokenizer

start_token = "<BOP>"
pad_token = "<PAD>"
end_token = "<EOP>"

app = FastAPI()
log = ColorLog()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pylint: disable=no-member

model = AutoModelForCausalLM.from_pretrained("./model").to(device)
log.warning(f"Using device: {model.device}")

tokenizer = AutoTokenizer.from_pretrained(
    "distilgpt2", cache_dir="./model", bos_token=start_token, eos_token=end_token, pad_token=pad_token
)

@app.get("/", status_code=302)
async def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.post("/generate/")
async def generate(
    prompt: str = Query(..., min_length=1, max_length=255)
    ):
    ''' Generate a fancier prompt '''

    log.warning(f"👈 {prompt}")
    max_prompt_length = 50
    min_prompt_length = 30
    temperature = 1.0
    # top_k = 70
    top_p = 0.7

    encoded_prompt = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    encoded_prompt = encoded_prompt.to(model.device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_prompt_length,
        min_length=min_prompt_length,
        temperature=temperature,
        # top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,  # gets rid of warning
    )

    tokenized_start_token = tokenizer.encode(start_token)
    generated_prompts = []
    for generated_sequence in output_sequences:
        # precision is a virtue
        tokens = []
        for i, s in enumerate(generated_sequence):
            if s in tokenized_start_token and i != 0:
                if len(tokens) >= min_prompt_length:
                    break
            tokens.append(s)

        text = tokenizer.decode(
            tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        text = (
            text.strip().replace("\n", " ").replace("/", ",").replace("8 k", "8k").replace("4 k", "4k").replace("2 d", "2d").replace("3 d", "3d").replace("f 1. 8", "f 1.8")
        )  # / remove slash. It causes problems in namings
        # remove repeated adjacent words from `text`. For example: "lamma lamma is cool cool" -> "lamma is cool"
        text = " ".join([k for k, g in itertools.groupby(text.split())])
        generated_prompts.append(text)

    log.warning(f"👉 {generated_prompts[0]}")

    return {
        "prompt": prompt,
        "parrot": generated_prompts[0]
    }
