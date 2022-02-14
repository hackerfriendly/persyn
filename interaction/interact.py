'''
interact.py

A REST API for tying together all of the other components.
'''
import os
import random

from typing import Optional

# import humanize

from fastapi import FastAPI, HTTPException, Query

# Prompt completion
from gpt import GPT

# text-to-speech
from voice import tts

# Emotions courtesy of Dr. Noonian Soong
from feels import get_feels

# Long and short term memory
from memory import Recall

# Time handling
from chrono import natural_time

# Color logging
from color_logging import ColorLog

log = ColorLog()

# These are all defined in config/*.conf
BOT_NAME = os.environ["BOT_NAME"]
BOT_ID = os.environ["BOT_ID"]
BOT_VOICE = os.environ.get('BOT_VOICE', 'USA')

# Minimum completion reply quality. Lower numbers get more dark + sleazy.
MINIMUM_QUALITY_SCORE = float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))

# Temperature. 0.0 == repetitive, 1.0 == chaos
TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.99))

# How are we feeling today?
feels = {'current': get_feels("")}

# GPT-3 for completion
completion = GPT(bot_name=BOT_NAME, min_score=MINIMUM_QUALITY_SCORE)

# FastAPI
app = FastAPI()

# Elasticsearch memory
recall = Recall(
    bot_name=BOT_NAME,
    bot_id=BOT_ID,
    url=os.environ['ELASTIC_URL'],
    auth_name=os.environ["BOT_NAME"],
    auth_key=os.environ.get('ELASTIC_KEY', None),
    index_prefix=os.environ.get('ELASTIC_INDEX_PREFIX', BOT_NAME.lower()),
    conversation_interval=600, # ten minutes
    verify_certs=True
)

def summarize_convo(service, channel, save=True):
    '''
    Generate a GPT summary of the current conversation for this channel.
    If save == True, save it to long term memory.
    Returns the text summary.
    '''
    summaries, convo = recall.load(service, channel, summaries=1)
    if not convo:
        return '\n'.join(summaries)

    summary = completion.get_summary(
        text='\n'.join(convo),
        summarizer="To briefly summarize this conversation, ",
        max_tokens=200
    )
    if save:
        recall.summary(service, channel, summary)
    return summary

def choose_reply(prompt, convo):
    ''' Choose the best reply from a list of possibilities '''

    # TODO: If no replies survive, try again?
    scored = completion.get_replies(
        prompt=prompt,
        convo=convo,
        temperature=TEMPERATURE
    )

    if not scored:
        return ":shrug:"

    for item in sorted(scored.items()):
        log.warning(f"{item[0]:0.2f}:", item[1])

    idx = random.choices(list(sorted(scored)), weights=list(sorted(scored)))[0]
    reply = scored[idx]
    log.info(f"✅ Choice: {idx:0.2f}", reply)

    return reply

def get_reply(service, channel, msg, speaker_name, speaker_id):
    ''' Get the best reply for the given channel. '''
    if recall.expired(service, channel):
        summarize_convo(service, channel, save=True)

    if msg != '...':
        recall.save(service, channel, msg, speaker_name, speaker_id)
        tts(msg)

    summaries, convo = recall.load(service, channel, summaries=3)
    narration = [f"Narrator: {line}" for line in summaries]
    prefix = "" # TODO: more contextual motivations go here

    # Load summaries and conversation
    newline = '\n'

    prompt = f"""{prefix}It is {natural_time()}. {BOT_NAME} is feeling {feels['current']['text']}.

{newline.join(narration)}
{newline.join(convo)}
{BOT_NAME}:"""

    reply = choose_reply(prompt, convo)

    recall.save(service, channel, reply, BOT_NAME, BOT_ID)
    tts(reply, voice=BOT_VOICE)
    feels['current'] = get_feels(f'{prompt} {reply}')

    log.warning("😄 Feeling:", feels['current'])

    return reply

def get_status(service, channel):
    ''' status report '''
    paragraph = '\n\n'
    newline = '\n'
    summaries, convo = recall.load(service, channel, summaries=3)
    return f'''It is {natural_time()}. {BOT_NAME} is feeling {feels['current']['text']}.

{paragraph.join(summaries)}

{newline.join(convo)}
'''

def amnesia(service, channel):
    ''' forget it '''
    return recall.forget(service, channel)

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Interact server. Try /docs"}

@app.post("/reply/")
async def handle_reply(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    msg: str = Query(..., min_length=1, max_length=5000),
    speaker_id: str = Query(..., min_length=1, max_length=255),
    speaker_name: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''

    if not msg.strip():
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least one non-space character."
        )

    return {
        "reply": get_reply(service, channel, msg, speaker_name, speaker_id)
    }

@app.post("/summary/")
async def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    save: Optional[bool] = Query(True)
    ):
    ''' Return the reply '''
    return {
        "summary": summarize_convo(service, channel, save)
    }

@app.post("/status/")
async def handle_status(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "status": get_status(service, channel)
    }

@app.post("/amnesia/")
async def handle_amnesia(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "amnesia": amnesia(service, channel)
    }
