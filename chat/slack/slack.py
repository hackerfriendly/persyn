#!/usr/bin/env python3
"""
slackbolt.py

A Slack bot based on GPT-3.
"""
import collections
import logging
import os
import random
import re
import sys
import tempfile
import uuid

import datetime as dt
import threading as th

import urllib3
import requests
import humanize
import tweepy

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

# long term memory
from elasticsearch import Elasticsearch

# Emotions courtesy of Dr. Noonian Soong
from feels import (
    random_emoji,
    nope_emoji,
    get_spectrum,
    get_feels,
    rank_feels
)

# Prompt completion
from gpt import GPT

# Color logging
from color_logging import debug, info, warning, error, critical # pylint: disable=unused-import

# Disable SSL warnings for Elastic
urllib3.disable_warnings()

# These are all defined in config/*.conf
ME = os.environ["BOT_NAME"]

ELASTIC_URL = os.environ['ELASTIC_URL']
ELASTIC_KEY = os.environ.get('ELASTIC_KEY', None)
ELASTIC_CONVO_INDEX = os.environ.get('ELASTIC_CONVO_INDEX', 'bot-conversations-v0')
ELASTIC_SUMMARY_INDEX = os.environ.get('ELASTIC_SUMMARY_INDEX', 'bot-summaries-v0')

# Minimum reply quality. Lower numbers get more dark + sleazy.
MINIMUM_QUALITY_SCORE = float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))

IMAGE_ENGINES = ["v-diffusion-pytorch-cfg", "vqgan", "stylegan2"]
IMAGE_MODELS = {
    "stylegan2": ["ffhq", "car", "cat", "church", "horse", "waifu"]
}
IMAGE_ENGINE_WEIGHTS = [0.4, 0.4, 0.2]

# New conversation every 10 minutes
CONVERSATION_INTERVAL = 600

# GPT-3 for completion
completion = GPT(bot_name=ME, min_score=MINIMUM_QUALITY_SCORE)

# Long-term memory
es = Elasticsearch([ELASTIC_URL], http_auth=(ME, ELASTIC_KEY), verify_certs=False, timeout=30)

# Twitter
twitter_auth = tweepy.OAuthHandler(os.environ['TWITTER_CONSUMER_KEY'], os.environ['TWITTER_CONSUMER_SECRET'])
twitter_auth.set_access_token(os.environ['TWITTER_ACCESS_TOKEN'], os.environ['TWITTER_ACCESS_TOKEN_SECRET'])

twitter = tweepy.API(twitter_auth)

BASEURL = os.environ.get('BASEURL', None)

# Voice support
VOICE_SERVER = os.environ['GTTS_SERVER_URL']
DEFAULT_VOICE = os.environ.get('DEFAULT_VOICE', 'UK')
VOICES = []

# Slack bolt App
app = App(token=os.environ['SLACK_BOT_TOKEN'])

# How are we feeling today?
feels = {}

# Length of the Short Term Memory. Bigger == more coherent == $$$
STM = 16

# Train of Thought (Short Term Memory): one deque per channel
ToT = {}

# Username cache
known_users = {}

def natural_time(hour=dt.datetime.now().hour):
    ''' Natural time of the day '''
    day_times = ("late at night", "early morning", "morning", "afternoon", "evening", "night")
    return day_times[hour // 4]

def setup_logging(stream=sys.stderr, log_format="%(message)s", debug_mode=False):
    ''' Basic logging '''
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(stream=stream, level=level, format=log_format)

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

def new_channel(channel):
    ''' Initialize a new channel. '''
    ToT[channel] = {
        'rejoinder': th.Timer(0, warning, ["New channel:", channel]),
        'status_update': th.Timer(0, debug, ["status_update_thread:", channel]),
        'convo': collections.deque(maxlen=STM)
    }
    load_from_ltm(channel)
    ToT[channel]['rejoinder'].start()
    ToT[channel]['status_update'].start()

def get_display_name(user_id):
    """ Return the user's first name if available, otherwise the display name """
    if user_id not in known_users:
        users_info = app.client.users_info(user=user_id)['user']
        try:
            known_users[user_id] = users_info['profile']['first_name']
        except KeyError:
            known_users[user_id] = users_info['profile']['display_name']

    return known_users[user_id]

def substitute_names(text):
    """ Substitute all <@XYZ> in text with the equivalent display name. """
    for user_id in set(re.findall(r'<@(\w+)>', text)):
        text = re.sub(f'<@{user_id}>', get_display_name(user_id), text)
    return text

def take_a_photo(channel, prompt, engine=None, model=None):
    ''' Pick an image engine and generate a photo '''
    if not engine:
        engine = random.choices(
            IMAGE_ENGINES,
            weights=IMAGE_ENGINE_WEIGHTS
        )[0]

    if engine == "stylegan2":
        req = {
            "engine": engine,
            "channel": channel,
            "prompt": prompt,
            "model": model or random.choice(IMAGE_MODELS["stylegan2"])
        }
    else:
        req = {
            "engine": engine,
            "channel": channel,
            "prompt": prompt
        }
    reply = requests.post(f"{os.environ['DREAM_SERVER_URL']}/generate/", params=req)
    warning(f"{os.environ['DREAM_SERVER_URL']}/generate/", f"{prompt}: {reply.status_code}")
    return reply.status_code

def elapsed(ts1, ts2):
    ''' Elapsed seconds between two timestamps (str in isoformat) '''
    return abs((dt.datetime.fromisoformat(ts2) - dt.datetime.fromisoformat(ts1)).total_seconds())

def clear_stm(channel):
    ''' Clear the short term memory for this channel '''
    if channel not in ToT:
        return

    ToT[channel]['rejoinder'].cancel()
    ToT[channel]['convo'].clear()
    ToT[channel]['convo_id'] = uuid.uuid4()

def get_cur_ts():
    ''' Return a properly formatted timestamp string '''
    return str(dt.datetime.now(dt.timezone.utc).astimezone().isoformat())

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

def get_reply(channel, them, msg):
    ''' Track the Train of Thought, and send an appropriate response. '''
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

@app.message(re.compile(r"^help$", re.I))
def help_me(say, context): # pylint: disable=unused-argument
    ''' TODO: These should really be / commands. '''
    say(f"""Commands:
  `...`: Let {ME} keep talking without interrupting
  `forget it`: Clear the conversation history for this channel
  `status`: How is {ME} feeling right now?
  `summary`: Explain it all to me in a single sentence.
  :camera: : Generate a picture summarizing this conversation
  :camera: _prompt_ : Generate a picture of _prompt_
  :selfie: Take a selfie
""")

@app.message(re.compile(r"^forget it$", re.I))
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

@app.message(re.compile(r"^:camera:(.+)$"))
def picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    say(f"OK, {them}.\n_{ME} takes out a camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=60,
        what=f"*{ME} takes a picture of _{prompt}_* It will take a few minutes to develop."
    )
    take_a_photo(channel, prompt)

@app.message(re.compile(r"^:selfie:$"))
def selfie(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{ME} takes out a camera and smiles awkwardly_.")
    say_something_later(
        say,
        channel,
        context,
        when=8,
        what=":cheese_wedge: *CHEESE!* :cheese_wedge:"
    )
    take_a_photo(
        channel,
        context['matches'][0].strip(),
        engine="stylegan2",
        model=random.choice(["ffhq", "waifu", "cat"])
    )

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

@app.message(re.compile(r"^:\w+:"))
def wink(say, context): # pylint: disable=unused-argument
    ''' Every single emoji gets a emoji one back. '''
    say(random_emoji())

@app.message(re.compile(r"^summary$", re.I))
def summarize(say, context):
    ''' Say a condensed summary of the ToT for this channel '''
    debug(app.client.bots_info())
    channel = context['channel_id']
    if channel not in ToT:
        say(":shrug:")
        return

    recent = '\n'.join(ToT[channel]['convo'])
    summary = completion.get_summary(recent)
    say(summary or ":shrug:")

def say_something_later(say, channel, context, when, what=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    if channel not in ToT:
        new_channel(channel)

    ToT[channel]['rejoinder'].cancel()

    if what:
        ToT[channel]['rejoinder'] = th.Timer(when, say, [what])
    else:
        # yadda yadda yadda
        yadda = {
            'channel_id': channel,
            'user_id': context['user_id'],
            'matches': ['...']
        }
        ToT[channel]['rejoinder'] = th.Timer(when, catch_all, [say, yadda])

    ToT[channel]['rejoinder'].start()

@app.message(re.compile(r"(.*)", re.I))
def catch_all(say, context):
    ''' Default message handler. Prompt GPT and randomly arm a Timer for later reply. '''
    channel = context['channel_id']

    if channel not in ToT:
        new_channel(channel)

    # Interrupt any rejoinder in progress
    ToT[channel]['rejoinder'].cancel()
    ToT[channel]['status_update'].cancel()

    them = get_display_name(context['user_id'])
    msg = substitute_names(context['matches'][0]).strip()

    tts(msg)

    # 5% of the time, say nothing (for now).
    if random.random() < 0.95:
        the_reply = get_reply(channel, them, msg)
        say(the_reply)
        tts(the_reply, voice=DEFAULT_VOICE)

    # Status update (and photo) in 2 minutes. Can be interrupted.
    ToT[channel]['status_update'] = th.Timer(120, status_report, [say, context])
    ToT[channel]['status_update'].start()

    interval = None
    # Long response
    if random.random() < 0.1:
        interval = [9,12]
    # Medium response
    elif random.random() < 0.2:
        interval = [6,8]
    # Quick response
    elif random.random() < 0.3:
        interval = [4,5]

    if interval:
        say_something_later(
            say,
            channel,
            context,
            when=random.randint(interval[0], interval[1])
        )

@app.event("app_mention")
def handle_app_mention_events(body, client, say): # pylint: disable=unused-argument
    ''' Reply to @mentions '''
    channel = body['event']['channel']
    them = get_display_name(body['event']['user'])
    msg = substitute_names(body['event']['text'])

    if channel not in ToT:
        new_channel(channel)

    say(get_reply(channel, them, msg))

@app.event("reaction_added")
def handle_reaction_added_events(body, logger): # pylint: disable=unused-argument
    '''
    Handle reactions: post images to Twitter.
    '''
    if not BASEURL:
        error("Twitter posting is not enabled in the config.")
        return

    try:
        result = app.client.conversations_history(
            channel=body['event']['item']['channel'],
            inclusive=True,
            oldest=body['event']['item']['ts'],
            limit=1
        )

        messages = result.get('messages', [])
        for msg in messages:
            # only post on the first reaction
            if 'reactions' in msg and len(msg['reactions']) == 1:
                print(msg['reactions'][0]['name'])
                if msg['reactions'][0]['name'] in nope_emoji:
                    warning("🐦 Not posting:", {msg['reactions'][0]['name']})
                    return

                if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                    blk = msg['blocks'][0]
                    if not blk['image_url'].startswith(BASEURL):
                        warning("🐦 Not my image, so not posting it to Twitter.")
                        return
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            resp = requests.get(blk['image_url'])
                            resp.raise_for_status()
                            fname = f"{tmpdir}/{blk['image_url'].split('/')[-1]}"
                            with open(fname, "wb") as f:
                                for chunk in resp.iter_content():
                                    f.write(chunk)
                            media = twitter.media_upload(fname)
                            twitter.update_status(blk['alt_text'], media_ids=[media.media_id])
                        info(f"🐦 Uploaded {blk['image_url']}")
                    except Exception as err:
                        error(f"🐦 Could not post {blk['image_url']}: {err}")
                else:
                    error(f"🐦 Non-image posting not implemented yet: {msg['text']}")

    except SlackApiError as err:
        print(f"Error: {err}")

@app.event("reaction_removed")
def handle_reaction_removed_events(body, logger): # pylint: disable=unused-argument
    ''' Skip for now '''
    info("Reaction removed event")

# @app.command("/echo")
# def repeat_text(ack, respond, command):
#     # Acknowledge command request
#     ack()
#     respond(f"_{command['text']}_")


if __name__ == "__main__":
    setup_logging(debug_mode=('DEBUG' in os.environ))
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    feels['current'] = get_feels("")
    try:
        handler.start()
    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        raise SystemExit(0) from kbderr
