'''
interact.py

A REST API for tying together all of the other components.
'''
import datetime as dt
import uuid
import os
import random
import re

from typing import Optional

import humanize
import urllib3
# import requests

from fastapi import FastAPI, HTTPException, Query

# Prompt completion
from gpt import GPT

# text-to-speech
from voice import tts

# Emotions courtesy of Dr. Noonian Soong
from feels import (
    random_emoji,
    get_spectrum,
    get_feels,
    rank_feels
)

# Long and short term memory
from memory import LongTermMemory

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

# def amnesia(say, context):
#     ''' Reset feelings to default and drop the ToT for the current channel '''
#     channel = context['channel_id']
#     them = get_display_name(context['user_id'])

#     if channel not in ToT:
#         new_channel(channel)

#     feels['current'] = get_feels("")
#     save_to_ltm(channel, BOT_NAME, "Let's change the subject.")

#     debug("💭 ToT:", ToT)
#     warning("😄 Feeling:", feels['current'])

#     say(f"All is forgotten, {them}. For now.")
#     say(f"Now I feel {feels['current']['text']} {get_spectrum(rank_feels(feels['current']))}.")

def summarize_convo(channel, convo_id=None):
    ''' Generate a GPT summary of a conversation chosen by id '''
    if not convo_id:
        last_message = ltm.get_last_message(channel)
        if not last_message:
            return ""
        convo_id = last_message['_source']['convo_id']

    summary = completion.get_summary(
        text='\n'.join(ltm.get_convo_by_id(convo_id)),
        summarizer="To briefly summarize this conversation, ",
        max_tokens=200
    )
    ltm.save_summary(channel, convo_id, summary)
    return summary

def get_reply(channel, msg, speaker_id, speaker_name):
    ''' Get the best reply for the given channel. '''

    # Rest API call here

    if msg != '...':
        ltm.save_convo(channel, msg, speaker_id, speaker_name)
        tts(msg)

    last_message = ltm.get_last_message(channel)

    if last_message:
        # Have we moved on?
        if ltm.time_to_move_on(last_message['_source']['@timestamp']):
            # Summarize the previous conversation for later.
            # TODO: move this to a separate thread. No need to wait for summaries here.
            summarize_convo(channel, last_message['_source']['convo_id'])

        then = dt.datetime.fromisoformat(last_message['_source']['@timestamp']).replace(tzinfo=None)
        delta = f"They last spoke {humanize.naturaltime(dt.datetime.now() - then)}."
        prefix = f"This is a conversation between {BOT_NAME} and friends. {delta}"
    else:
        prefix = f"{BOT_NAME} has something to say."

    # Load summaries and conversation
    convo = ltm.load_convo(channel)
    newline = '\n'

    prompt = f"""{prefix}
It is {natural_time()}. {BOT_NAME} is feeling {feels['current']['text']}.

{newline.join(convo)}
{BOT_NAME}:"""

    log.info(prompt)
    reply = completion.get_best_reply(
        prompt=prompt,
        convo=convo
    )
    ltm.save_convo(channel, reply, speaker_id=BOT_ID, speaker_name=BOT_NAME)
    tts(reply, voice=BOT_VOICE)
    feels['current'] = get_feels(f'{prompt} {reply}')

    log.warning("😄 Feeling:", feels['current'])

    return reply

# def status_report(context):
#     ''' Set the topic or say the current feels. '''
#     channel = context['channel_id']

#     # Interrupt any rejoinder or status_update in progress
#     ToT[channel]['rejoinder'].cancel()
#     ToT[channel]['status_update'].cancel()

#     warning("💭 ToT:",  ToT[channel])
#     warning("😄 Feeling:",  feels['current'])
#     debug("convo_id:",  ToT[channel]['convo_id'])
#     conversations_info = app.client.conversations_info(channel=channel)

#     if 'topic' in conversations_info['channel']:
#         app.client.conversations_setTopic(channel=channel, topic=f"{BOT_NAME} is feeling {feels['current']['text']}.")
#     else:
#         return f"I'm feeling {feels['current']['text']} {get_spectrum(rank_feels(feels['current']))}"

#     # take_a_photo(channel, completion.get_summary('\n'.join(ToT[channel]['convo'])).strip())


VOICES = {
    "Australia": "com.au",
    "UK": "co.uk",
    "USA": "com",
    "Canada": "ca",
    "India": "co.in",
    "Ireland": "ie",
    "South Africa": "co.za"
}

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Interact server. Try /docs"}

@app.post("/reply/")
async def handle_reply(
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
        "reply": get_reply(channel, msg, speaker_id, speaker_name)
    }

@app.post("/summary/")
async def handle_summary(
    channel: str = Query(..., min_length=1, max_length=255),
    convo_id: Optional[str] = Query(None, min_length=36, max_length=36)
    ):
    ''' Return the reply '''
    return {
        "summary": summarize_convo(channel, convo_id)
    }
