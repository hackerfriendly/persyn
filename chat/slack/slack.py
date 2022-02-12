#!/usr/bin/env python3
"""
slack.py

A Slack chat plugin for Persyn. Sends Slack events to interact.py.
"""
import os
import random
import re
import tempfile

import threading as th

import requests
import tweepy

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

# Color logging
from color_logging import ColorLog

log = ColorLog()

# These are all defined in config/*.conf
BOT_NAME = os.environ["BOT_NAME"]
BOT_ID = os.environ["BOT_ID"]

IMAGE_ENGINES = ["v-diffusion-pytorch-cfg"]#, "vqgan", "stylegan2"]
IMAGE_MODELS = {
    "stylegan2": ["ffhq", "car", "cat", "church", "horse", "waifu"]
}
IMAGE_ENGINE_WEIGHTS = [1]#[0.4, 0.4, 0.2]

# Twitter
twitter_auth = tweepy.OAuthHandler(os.environ['TWITTER_CONSUMER_KEY'], os.environ['TWITTER_CONSUMER_SECRET'])
twitter_auth.set_access_token(os.environ['TWITTER_ACCESS_TOKEN'], os.environ['TWITTER_ACCESS_TOKEN_SECRET'])

twitter = tweepy.API(twitter_auth)

BASEURL = os.environ.get('BASEURL', None)

# Slack bolt App
app = App(token=os.environ['SLACK_BOT_TOKEN'])
# Saved to the service field in ltm
SLACK_SERVICE = app.client.auth_test().data['url']

# Reminders: container for delayed response threads
reminders = {}

# Username cache
known_users = {}

# TODO: callback thread to poll(?) interact, or inbound API call for push notifications

def new_channel(channel):
    ''' Initialize a new channel. '''
    reminders[channel] = {
        'rejoinder': th.Timer(0, log.warning, ["New channel:", channel]),
        'count': 0
    }
    reminders[channel]['rejoinder'].start()

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
    log.warning(f"{os.environ['DREAM_SERVER_URL']}/generate/", f"{prompt}: {reply.status_code}")
    return reply.status_code

def get_reply(channel, msg, speaker_name=None, speaker_id=None):
    ''' Ask interact for an appropriate response. '''
    if msg != '...':
        log.info(f"[{channel}] {speaker_name}: {msg}")

    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
        "msg": msg,
        "speaker_name": speaker_name,
        "speaker_id": speaker_id
    }
    try:
        response = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/reply/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not get_reply(): {err}")
        return ":shrug:"

    reply = response.json()['reply']
    log.warning(f"[{channel}] {BOT_NAME}: {reply}")
    return reply

def get_summary(channel, save=False):
    ''' Ask interact for a channel summary. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
        "save": save
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/summary/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not get_reply(): {err}")
        return ":shrug:"

    log.warning(f"∑ {reply.json()['summary']}")
    return reply.json()['summary'] or ":shrug:"

def get_status(channel):
    ''' Ask interact for status. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/status/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not get_reply(): {err}")
        return ":shrug:"

    return reply.json()['status']

def forget_it(channel):
    ''' There is no antimemetics division. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
    }
    try:
        response = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/amnesia/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not forget_it(): {err}")
        return ":shrug:"

    return ":exploding_head:"

@app.message(re.compile(r"^help$", re.I))
def help_me(say, context): # pylint: disable=unused-argument
    ''' TODO: These should really be / commands. '''
    say(f"""Commands:
  `...`: Let {BOT_NAME} keep talking without interrupting
  `summary`: Explain it all to me in a single sentence.
  `status`: Say exactly what is on {BOT_NAME}'s mind.
  :camera: : Generate a picture summarizing this conversation
  :camera: _prompt_ : Generate a picture of _prompt_
  :selfie: Take a selfie
""")

@app.message(re.compile(r"^:camera:(.+)$"))
def picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    say(f"OK, {them}.\n_{BOT_NAME} takes out a camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=60,
        what=f"*{BOT_NAME} takes a picture of _{prompt}_* It will take a few minutes to develop."
    )
    take_a_photo(channel, prompt)

@app.message(re.compile(r"^:selfie:$"))
def selfie(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{BOT_NAME} takes out a camera and smiles awkwardly_.")
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

    say(f"OK, {them}.\n_{BOT_NAME} takes out a camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=60,
        what=f"*click* _{BOT_NAME} shakes it like a polaroid picture_"
    )
    take_a_photo(channel, get_summary(channel))

@app.message(re.compile(r"^summary(\!)?$", re.I))
def summarize(say, context):
    ''' Say a condensed summary of this channel '''
    save = bool(context['matches'][0])
    channel = context['channel_id']
    say("💭 " + get_summary(channel, save))

@app.message(re.compile(r"^status$", re.I))
def status(say, context):
    ''' Say a condensed summary of this channel '''
    channel = context['channel_id']
    say("\n".join([f"> {line}" for line in get_status(channel).split("\n")]))

def say_something_later(say, channel, context, when, what=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    if channel not in reminders:
        new_channel(channel)

    reminders[channel]['rejoinder'].cancel()

    if what:
        reminders[channel]['rejoinder'] = th.Timer(when, say, [what])
    else:
        # Only 3 autoreplies max
        if reminders[channel]['count'] >= 3:
            reminders[channel]['count'] = 0
            return

        reminders[channel]['count'] = reminders[channel]['count'] + 1

        # yadda yadda yadda
        yadda = {
            'channel_id': channel,
            'user_id': context['user_id'],
            'matches': ['...']
        }
        reminders[channel]['rejoinder'] = th.Timer(when, catch_all, [say, yadda])

    reminders[channel]['rejoinder'].start()

@app.message(re.compile(r"(.*)", re.I))
def catch_all(say, context):
    ''' Default message handler. Prompt GPT and randomly arm a Timer for later reply. '''
    channel = context['channel_id']

    if channel not in reminders:
        new_channel(channel)

    # Interrupt any rejoinder in progress
    reminders[channel]['rejoinder'].cancel()

    speaker_id = context['user_id']
    speaker_name = get_display_name(speaker_id)
    msg = substitute_names(context['matches'][0]).strip()

    the_reply = get_reply(channel, msg, speaker_name, speaker_id)

    if the_reply == ":shrug:":
        return

    say(the_reply)

    if the_reply.endswith('…') or the_reply.endswith('...'):
        say_something_later(
            say,
            channel,
            context,
            when=1
        )
        return

    interval = None
    # Long response
    if random.random() < 0.1:
        interval = [7,10]
    # Medium response
    elif random.random() < 0.2:
        interval = [4,6]
    # Quick response
    elif random.random() < 0.3:
        interval = [2,3]

    if interval:
        say_something_later(
            say,
            channel,
            context,
            when=random.randint(interval[0], interval[1])
        )
    else:
        reminders[channel]['rejoinder'].cancel()
        reminders[channel]['rejoinder'] = th.Timer(20, take_a_photo, [channel, get_summary(channel)])

@app.event("app_mention")
def handle_app_mention_events(body, client, say): # pylint: disable=unused-argument
    ''' Reply to @mentions '''
    channel = body['event']['channel']
    speaker_id = body['event']['user']
    speaker_name = get_display_name(speaker_id)
    msg = substitute_names(body['event']['text'])

    if channel not in reminders:
        new_channel(channel)

    say(get_reply(channel, msg, speaker_name, speaker_id))

@app.event("reaction_added")
def handle_reaction_added_events(body, logger): # pylint: disable=unused-argument
    '''
    Handle reactions: post images to Twitter.
    '''
    channel = body['event']['item']['channel']
    try:
        result = app.client.conversations_history(
            channel=channel,
            inclusive=True,
            oldest=body['event']['item']['ts'],
            limit=1
        )

        messages = result.get('messages', [])
        for msg in messages:
            # only post on the first reaction
            if 'reactions' in msg and len(msg['reactions']) == 1:
                if msg['reactions'][0]['name'] in ['-1', 'hankey', 'no_entry', 'no_entry_sign', 'hand']:
                    if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                        log.warning("🐦 Not posting:", {msg['reactions'][0]['name']})
                        return
                    log.warning("🤯 All is forgotten.")
                    forget_it(channel)
                    return
                try:
                    req = { "service": SLACK_SERVICE, "channel": channel }
                    response = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/amnesia/", params=req)
                    response.raise_for_status()
                except requests.exceptions.RequestException as err:
                    log.critical(f"🤖 Could not get_reply(): {err}")
                    return

                if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                    if not BASEURL:
                        log.error("Twitter posting is not enabled in the config.")
                        return

                    blk = msg['blocks'][0]
                    if not blk['image_url'].startswith(BASEURL):
                        log.warning("🐦 Not my image, so not posting it to Twitter.")
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
                            if len(blk['alt_text']) > 277:
                                caption = blk['alt_text'][:277] + '...'
                            else:
                                caption = blk['alt_text']
                            twitter.update_status(caption, media_ids=[media.media_id])
                        log.info(f"🐦 Uploaded {blk['image_url']}")
                    except requests.exceptions.RequestException as err:
                        log.error(f"🐦 Could not post {blk['image_url']}: {err}")
                else:
                    log.error(f"🐦 Unhandled reaction {msg['reactions'][0]['name']} to: {msg['text']}")

    except SlackApiError as err:
        log.error(f"Error: {err}")

@app.event("reaction_removed")
def handle_reaction_removed_events(body, logger): # pylint: disable=unused-argument
    ''' Skip for now '''
    log.info("Reaction removed event")

# @app.command("/echo")
# def repeat_text(ack, respond, command):
#     # Acknowledge command request
#     ack()
#     respond(f"_{command['text']}_")


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    try:
        handler.start()
    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        raise SystemExit(0) from kbderr
