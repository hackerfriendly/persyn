'''
interact.py

A REST API for tying together all of the other components.
'''
import datetime as dt
import os

from typing import Optional

import humanize

from fastapi import FastAPI, HTTPException, Query

# Prompt completion
from gpt import GPT

# text-to-speech
from voice import tts

# Emotions courtesy of Dr. Noonian Soong
from feels import get_feels

# Long and short term memory
from memory import LongTermMemory

# Entity IDs
from entities import EntityMapper

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

# How are we feeling today?
feels = {'current': get_feels("")}

# GPT-3 for completion
completion = GPT(bot_name=BOT_NAME, bot_id=BOT_ID, min_score=MINIMUM_QUALITY_SCORE)

# Entity mapper
em = EntityMapper(BOT_ID)

# FastAPI
app = FastAPI()

# Elasticsearch memory
ltm = LongTermMemory(
    url=os.environ['ELASTIC_URL'],
    auth_name=os.environ["BOT_NAME"],
    auth_key=os.environ.get('ELASTIC_KEY', None),
    convo_index=os.environ.get('ELASTIC_CONVO_INDEX', 'bot-conversations-v0'),
    summary_index=os.environ.get('ELASTIC_SUMMARY_INDEX', 'bot-summaries-v0'),
    conversation_interval=600, # New conversation every 10 minutes
    verify_certs=False
)

def summarize_convo(service, channel, convo_id=None, save=True):
    ''' Generate a GPT summary of a conversation chosen by id '''
    if not convo_id:
        last_message = ltm.get_last_message(service, channel)
        if not last_message:
            return ""
        convo_id = last_message['_source']['convo_id']

    summary = completion.get_summary(
        text='\n'.join(ltm.get_convo_by_id(convo_id)),
        summarizer="To briefly summarize this conversation, ",
        max_tokens=200
    )
    if save:
        ltm.save_summary(service, channel, convo_id, summary)
    return summary

def get_reply(service, channel, msg, speaker_id, speaker_name):
    ''' Get the best reply for the given channel. '''
    entity_id = em.name_to_id(service, channel, speaker_id)

    if msg != '...':
        ltm.save_convo(service, channel, msg, entity_id, speaker_name)
        tts(msg)

    last_message = ltm.get_last_message(service, channel)

    if last_message:
        # Have we moved on?
        if ltm.time_to_move_on(last_message['_source']['@timestamp']):
            # Summarize the previous conversation for later.
            # TODO: move this to a separate thread. No need to wait for summaries here.
            summarize_convo(service, channel, last_message['_source']['convo_id'])

        then = dt.datetime.fromisoformat(last_message['_source']['@timestamp']).replace(tzinfo=None)
        delta = f"They last spoke {humanize.naturaltime(dt.datetime.now() - then)}."
        prefix = f"This is a conversation between {BOT_NAME} and friends. {delta}"
    else:
        prefix = f"{BOT_NAME} has something to say."

    # Load summaries and conversation
    convo = ltm.load_convo(service, channel)
    newline = '\n'

    prompt = f"""{prefix}
It is {natural_time()}. {BOT_NAME} is feeling {feels['current']['text']}.

{newline.join(convo)}
{BOT_NAME}:"""

    reply = completion.get_best_reply(
        prompt=prompt,
        convo=convo
    )
    ltm.save_convo(service, channel, reply, entity_id=BOT_ID, speaker_name=BOT_NAME)
    tts(reply, voice=BOT_VOICE)
    feels['current'] = get_feels(f'{prompt} {reply}')

    log.warning("😄 Feeling:", feels['current'])

    return reply

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
        "reply": get_reply(service, channel, msg, speaker_id, speaker_name)
    }

@app.post("/summary/")
async def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    convo_id: Optional[str] = Query(None, min_length=36, max_length=36),
    save: Optional[bool] = Query(True)
    ):
    ''' Return the reply '''
    return {
        "summary": summarize_convo(service, channel, convo_id, save)
    }
