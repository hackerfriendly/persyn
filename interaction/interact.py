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

# Color logging
from color_logging import debug, info, warning, error, critical # pylint: disable=unused-import

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
from ltm import LongTermMemory

# These are all defined in config/*.conf
ME = os.environ["BOT_NAME"]

# Minimum completion reply quality. Lower numbers get more dark + sleazy.
MINIMUM_QUALITY_SCORE = float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))

# How are we feeling today?
feels = {}

# TODO: remove this
ToT = {}

# GPT-3 for completion
completion = GPT(bot_name=ME, min_score=MINIMUM_QUALITY_SCORE)

# FastAPI
app = FastAPI()

# Elasticsearch memory
ltm = LongTermMemory(
    url=os.environ['ELASTIC_URL'],
    auth_name=ME,
    auth_key=os.environ.get('ELASTIC_KEY', None),
    convo_index=os.environ.get('ELASTIC_CONVO_INDEX', 'bot-conversations-v0'),
    summary_index=os.environ.get('ELASTIC_SUMMARY_INDEX', 'bot-summaries-v0'),
    conversation_interval=600, # New conversation every 10 minutes
    verify_certs=False
)

def amnesia(say, context):
    ''' Reset feelings to default and drop the ToT for the current channel '''
    channel = context['channel_id']
    them = get_display_name(context['user_id'])

    if channel not in ToT:
        new_channel(channel)

    feels['current'] = get_feels("")
    save_to_ltm(channel, ME, "Let's change the subject.")

    debug("💭 ToT:", ToT)
    warning("😄 Feeling:", feels['current'])

    say(f"All is forgotten, {them}. For now.")
    say(f"Now I feel {feels['current']['text']} {get_spectrum(rank_feels(feels['current']))}.")

def natural_time(hour=dt.datetime.now().hour):
    ''' Natural time of the day '''
    day_times = ("late at night", "early morning", "morning", "afternoon", "evening", "night")
    return day_times[hour // 4]

def get_convo_summaries(channel, size=1):
    ''' Return the last n conversation summaries seen on this channel '''
    try:
        ret = es.search( # pylint: disable=unexpected-keyword-arg
            index=ELASTIC_SUMMARY_INDEX,
            query={
                "term": {"channel.keyword": channel}
            },
            sort=[{"@timestamp":{"order":"desc"}}],
            size=size
        )['hits']['hits']
    except KeyError:
        return None

    return [convo['_source']['summary'] for convo in ret[::-1]]

def get_last_message(channel):
    ''' Return the last message seen on this channel '''
    try:
        return es.search( # pylint: disable=unexpected-keyword-arg
            index=ELASTIC_CONVO_INDEX,
            query={
                "term": {"channel.keyword": channel}
            },
            sort=[{"@timestamp":{"order":"desc"}}],
            size=1
        )['hits']['hits'][0]
    except KeyError:
        return None

def summarize_convo(channel, convo_id):
    ''' Generate a GPT summary of a conversation chosen by id '''
    summary = completion.get_summary(
        text='\n'.join(get_convo_by_id(convo_id)),
        summarizer="To briefly summarize this conversation, ",
        max_tokens=200
    )
    doc = {
        "convo_id": convo_id,
        "summary": summary,
        "channel": channel,
        "@timestamp": get_cur_ts()
    }
    _id = es.index(index=ELASTIC_SUMMARY_INDEX, document=doc)["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter


def get_reply(channel, them, msg):
    ''' Track the Train of Thought, and send an appropriate response. '''

    # Rest API call here

    if msg != '...':
        save_to_ltm(channel, them, msg)
        ToT[channel]['convo'].append(f"{them}: {msg}")

    last_message = get_last_message(channel)

    if last_message:
        # Have we moved on?
        if last_message['_source']['convo_id'] != ToT[channel]['convo_id']:
            # Summarize the previous conversation for later
            summarize_convo(channel, last_message['_source']['convo_id'])
            for line in get_convo_summaries(channel, 3):
                ToT[channel]['convo'].append(line.strip() + '\n')
            ToT[channel]['convo'].append(f"{them}: {msg}")

        then = dt.datetime.fromisoformat(last_message['_source']['@timestamp']).replace(tzinfo=None)
        delta = f"They last spoke {humanize.naturaltime(dt.datetime.now() - then)}."
        prefix = f"This is a conversation between {ME} and friends. {delta}"
    else:
        prefix = f"{ME} has something to say."

    convo = '\n'.join(ToT[channel]['convo'])

    if not convo:
        convo = '\n\n'.join(get_convo_summaries(channel, 3))

    prompt = f"""{prefix}
It is {natural_time()}. {ME} is feeling {feels['current']['text']}.

{convo}
{ME}:"""

    info(prompt)
    reply = completion.get_best_reply(
        prompt=prompt,
        convo=ToT[channel]['convo']
        # stop=[f"{u}:" for u in get_channel_members(channel)[:3]]
    )
    ToT[channel]['convo'].append(f"{ME}: {reply}")
    save_to_ltm(channel, ME, reply)

    feels['current'] = get_feels(f'{prompt} {reply}')

    debug("💭 ToT:", ToT)
    warning("😄 Feeling:", feels['current'])

    return reply

def status_report(say, context):
    ''' Set the topic or say the current feels. '''
    channel = context['channel_id']

    if channel not in ToT:
        new_channel(channel)

    # Interrupt any rejoinder or status_update in progress
    ToT[channel]['rejoinder'].cancel()
    ToT[channel]['status_update'].cancel()

    warning("💭 ToT:",  ToT[channel])
    warning("😄 Feeling:",  feels['current'])
    debug("convo_id:",  ToT[channel]['convo_id'])
    conversations_info = app.client.conversations_info(channel=channel)

    if 'topic' in conversations_info['channel']:
        app.client.conversations_setTopic(channel=channel, topic=f"{ME} is feeling {feels['current']['text']}.")
    else:
        say(f"I'm feeling {feels['current']['text']} {get_spectrum(rank_feels(feels['current']))}")

    take_a_photo(channel, completion.get_summary('\n'.join(ToT[channel]['convo'])).strip())


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
    return {"message": "Google Text To Speech server. Try /docs"}

@app.get("/voices/")
async def voices():
    ''' List all available voices '''
    return {
        "voices": list(VOICES),
        "success": True
    }

@app.post("/say/")
async def say(
    text: str = Query(..., min_length=1, max_length=5000),
    voice: Optional[str] = Query("USA", max_length=32)):
    ''' Generate with gTTS and pipe to audio. '''

    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Choose one of: {', '.join(list(VOICES))}"
        )

    if not any(c.isalnum() for c in text):
        raise HTTPException(
            status_code=400,
            detail=f"Text must contain at least one alphanumeric character."
        )
