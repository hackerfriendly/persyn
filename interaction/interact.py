'''
interact.py

A REST API for tying together all of the other components.
'''

import collections
import datetime as dt
import uuid

from subprocess import run, CalledProcessError
from typing import Optional
from io import BytesIO

import humanize

from fastapi import FastAPI, HTTPException, Query

# Color logging
from color_logging import debug, info, warning, error, critical # pylint: disable=unused-import

# long term memory
from elasticsearch import Elasticsearch

# Prompt completion
from gpt import GPT

# Emotions courtesy of Dr. Noonian Soong
from feels import (
    random_emoji,
    nope_emoji,
    get_spectrum,
    get_feels,
    rank_feels
)

# These are all defined in config/*.conf
ME = os.environ["BOT_NAME"]

ELASTIC_URL = os.environ['ELASTIC_URL']
ELASTIC_KEY = os.environ.get('ELASTIC_KEY', None)
ELASTIC_CONVO_INDEX = os.environ.get('ELASTIC_CONVO_INDEX', 'bot-conversations-v0')
ELASTIC_SUMMARY_INDEX = os.environ.get('ELASTIC_SUMMARY_INDEX', 'bot-summaries-v0')

# Minimum completion reply quality. Lower numbers get more dark + sleazy.
MINIMUM_QUALITY_SCORE = float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))

# Voice support
VOICE_SERVER = os.environ['GTTS_SERVER_URL']
DEFAULT_VOICE = os.environ.get('DEFAULT_VOICE', 'UK')
VOICES = []

# Length of the Short Term Memory. Bigger == more coherent == $$$
STM = 16

# How are we feeling today?
feels = {}

# Long-term memory
es = Elasticsearch([ELASTIC_URL], http_auth=(ME, ELASTIC_KEY), verify_certs=False, timeout=30)

# GPT-3 for completion
completion = GPT(bot_name=ME, min_score=MINIMUM_QUALITY_SCORE)

# FastAPI
app = FastAPI()

def tts(message, voice=None):
    ''' Send a message to a voice server '''

    # Skip continuation messages
    if message == "...":
        return

    global VOICE_SERVER, VOICES

    if not VOICE_SERVER:
        error("No tts voice server found, voice disabled.")
        return

    if not VOICES:
        reply = requests.get(f'{VOICE_SERVER}/voices/')
        if not reply.ok:
            error("Could not fetch tts voices:", reply.text)
            VOICE_SERVER = None
            return
        VOICES = [v for v in reply.json()['voices'] if v != DEFAULT_VOICE]
        warning("📣 Available voices:", VOICES)

    if voice is None:
        voice = random.choice(VOICES)

    req = {
        "text": message,
        "voice": voice
    }
    reply = requests.post(f'{VOICE_SERVER}/say/', params=req)

    if reply.ok:
        info(f"📣 Sent to tts: ({voice})", message)
    else:
        error("📣 Connect to tts failed:", reply.text)

def load_from_ltm(channel):
    ''' Load the last conversation from LTM. '''

    clear_stm(channel)

    history = es.search( # pylint: disable=unexpected-keyword-arg
        index=ELASTIC_CONVO_INDEX,
        query={
            "term": {"channel.keyword": channel}
        },
        sort=[{"@timestamp":{"order":"desc"}}],
        size=STM
    )['hits']['hits']

    if not history:
        return

    last_ts = history[-1]['_source']['@timestamp']

    for line in history[::-1]:
        src = line['_source']
        time_lag = elapsed(last_ts, src['@timestamp'])
        last_ts = src['@timestamp']

        # Clear the short term memory after CONVERSATION_INTERVAL
        if time_lag > CONVERSATION_INTERVAL:
            clear_stm(channel)

        ToT[channel]['convo'].append(f"{src['speaker']}: {src['msg']}")
        ToT[channel]['convo_id'] = src['convo_id']

def get_convo_by_id(convo_id):
    ''' Extract a full conversation by its convo_id. Returns a list of strings. '''
    history = es.search( # pylint: disable=unexpected-keyword-arg
        index=ELASTIC_CONVO_INDEX,
        query={
            "term": {"convo_id.keyword": convo_id}
        },
        sort=[{"@timestamp":{"order":"asc"}}],
        size=1000
    )['hits']['hits']

    ret = []
    for line in history:
        ret.append(f"{line['_source']['speaker']}: {line['_source']['msg']}")

    debug(f"get_convo_by_id({convo_id}):", ret)
    return ret

def amnesia(say, context):
    ''' Reset feelings to default and drop the ToT for the current channel '''
    channel = context['channel_id']
    them = get_display_name(context['user_id'])

    if channel not in ToT:
        new_channel(channel)

    clear_stm(channel)
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

def save_to_ltm(channel, them, msg):
    ''' Save convo to ElasticSearch '''
    cur_ts = get_cur_ts()
    last_message = get_last_message(channel)

    if last_message:
        prev_ts = last_message['_source']['@timestamp']
        time_lag = elapsed(prev_ts, cur_ts)

        # Clear the short term memory after CONVERSATION_INTERVAL
        if time_lag > CONVERSATION_INTERVAL:
            clear_stm(channel)
        else:
            ToT[channel]['convo_id'] = last_message['_source']['convo_id']
    else:
        prev_ts = cur_ts
        clear_stm(channel)

    doc = {
        "@timestamp": cur_ts,
        "channel": channel,
        "speaker": them,
        "msg": msg,
        "elapsed": elapsed(prev_ts, cur_ts),
        "convo_id": ToT[channel]['convo_id']
    }
    _id = es.index(index=ELASTIC_CONVO_INDEX, document=doc)["_id"] # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    debug("doc:", _id)

    return _id

def elapsed(ts1, ts2):
    ''' Elapsed seconds between two timestamps (str in isoformat) '''
    return abs((dt.datetime.fromisoformat(ts2) - dt.datetime.fromisoformat(ts1)).total_seconds())

def get_cur_ts():
    ''' Return a properly formatted timestamp string '''
    return str(dt.datetime.now(dt.timezone.utc).astimezone().isoformat())

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
            clear_stm(channel)
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

@app.message(re.compile(r"^status$", re.I))
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

    # Keep the conversation going hours later
    say_something_later(
        say,
        channel,
        context,
        when=random.randint(12 * 3600, 48 * 3600)
    )

    take_a_photo(channel, completion.get_summary('\n'.join(ToT[channel]['convo'])).strip())

@app.message(re.compile(r"^:camera:$"))
def photo_summary(say, context): # pylint: disable=unused-argument
    ''' Take a photo of this conversation '''

    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{ME} takes out a camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=60,
        what=f"_{ME} takes a picture of this conversation._ It will take a few minutes to develop."
    )
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

