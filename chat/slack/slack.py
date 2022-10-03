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

# Artist names
from art import artists

log = ColorLog()

# These are all defined in config/*.conf
BOT_NAME = os.environ["BOT_NAME"]
BOT_ID = os.environ["BOT_ID"]

IMAGE_ENGINES = ["latent-diffusion", "v-diffusion-pytorch-cfg", "v-diffusion-pytorch-clip"] # "vqgan", "stylegan2"
IMAGE_MODELS = {
    "stylegan2": ["ffhq", "waifu"], #, "cat", "car", "church", "horse"
    "v-diffusion-pytorch-cfg": ["cc12m_1_cfg"],
    "v-diffusion-pytorch-clip": ["yfcc_2", "cc12m_1"],
    "latent-diffusion": ["default"],
    "stable-diffusion": ["default"],
    "dalle2": ["default"]
}

# Twitter support
twitter = None
if os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', None):
    twitter_auth = tweepy.OAuthHandler(os.environ['TWITTER_CONSUMER_KEY'], os.environ['TWITTER_CONSUMER_SECRET'])
    twitter_auth.set_access_token(os.environ['TWITTER_ACCESS_TOKEN'], os.environ['TWITTER_ACCESS_TOKEN_SECRET'])

    twitter = tweepy.API(twitter_auth)

BASEURL = os.environ.get('BASEURL', None)

# Slack bolt App
app = App(token=os.environ['SLACK_BOT_TOKEN'])
# Saved to the service field in ltm
SLACK_SERVICE = app.client.auth_test().data['url']
log.warning(f"SLACK_SERVICE: {SLACK_SERVICE}")

# Reminders: container for delayed response threads
reminders = {}

# Username cache
known_users = {}

# Known bots
known_bots = {}

# TODO: callback thread to poll(?) interact, or inbound API call for push notifications

def new_channel(channel):
    ''' Initialize a new channel. '''
    reminders[channel] = {
        'rejoinder': th.Timer(0, log.warning, ["New channel rejoinder:", channel]),
        'summarizer': th.Timer(1, log.warning, ["New channel summarizer:", channel]),
        'count': 0
    }
    reminders[channel]['rejoinder'].start()
    reminders[channel]['summarizer'].start()

def is_bot(user_id):
    """ Returns true if the user_id is a Slack bot """
    get_display_name(user_id)
    return known_bots[user_id]

def get_display_name(user_id):
    """ Return the user's first name if available, otherwise the display name """
    if user_id not in known_users:
        users_info = app.client.users_info(user=user_id)['user']
        try:
            profile = users_info['profile']
            known_users[user_id] = profile.get('first_name') or profile.get('display_name') or profile.get('real_name')
        except KeyError:
            known_users[user_id] = user_id

    if user_id not in known_bots:
        known_bots[user_id] = users_info['is_bot']

    return known_users[user_id]

def substitute_names(text):
    """ Substitute all <@XYZ> in text with the equivalent display name. """
    for user_id in set(re.findall(r'<@(\w+)>', text)):
        text = re.sub(f'<@{user_id}>', get_display_name(user_id), text)
    return text

def speakers():
    ''' Everyone speaking in any channel '''
    return [BOT_NAME] + list(known_users.values())

def take_a_photo(channel, prompt, engine=None, model=None, style=None):
    ''' Pick an image engine and generate a photo '''
    if not engine:
        engine = random.choice(IMAGE_ENGINES)

    req = {
        "engine": engine,
        "channel": channel,
        "prompt": prompt,
        "model": model or random.choice(IMAGE_MODELS[engine]),
        "slack_bot_token": os.environ['SLACK_BOT_TOKEN'],
        "bot_name": os.environ['BOT_NAME'],
        "style": style
    }
    reply = requests.post(f"{os.environ['DREAM_SERVER_URL']}/generate/", params=req)
    if reply.ok:
        log.warning(f"{os.environ['DREAM_SERVER_URL']}/generate/", f"{prompt}: {reply.status_code}")
    else:
        log.error(f"{os.environ['DREAM_SERVER_URL']}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
    return reply.ok

def get_reply(channel, msg, speaker_name, speaker_id):
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
        log.critical(f"🤖 Could not post /reply/ to interact: {err}")
        return " :speech_balloon: :interrobang: "

    reply = response.json()['reply']
    log.warning(f"[{channel}] {BOT_NAME}: {reply}")
    return reply

def get_summary(channel, save=False, photo=False, max_tokens=200, include_keywords=False, context_lines=0):
    ''' Ask interact for a channel summary. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
        "save": save,
        "max_tokens": max_tokens,
        "include_keywords": include_keywords,
        "context_lines": context_lines
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/summary/", params=req)
        reply.raise_for_status()
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
        log.critical(f"🤖 Could not post /summary/ to interact: {err}")
        return " :writing_hand: :interrobang: "

    summary = reply.json()['summary']
    log.warning(f"∑ {reply.json()['summary']}")

    if summary:
        if photo:
            take_a_photo(channel, summary, engine="stable-diffusion", style=f"{random.choice(artists)}")
        return summary

    return " :spiral_note_pad: :interrobang: "

def get_nouns(text):
    ''' Ask interact for all the nouns in text, excluding the speakers. '''
    req = {
        "text": text
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/nouns/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /nouns/ to interact: {err}")
        return []

    return [e for e in reply.json()['nouns'] if e not in speakers()]

def get_entities(text):
    ''' Ask interact for all the entities in text, excluding the speakers. '''
    req = {
        "text": text
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/entities/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /entities/ to interact: {err}")
        return []

    return [e for e in reply.json()['entities'] if e not in speakers()]

def get_daydream(channel):
    ''' Ask interact to daydream about this channel. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/daydream/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /daydream/ to interact: {err}")
        return []

    return reply.json()['daydream']

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
        log.critical(f"🤖 Could not post /status/ to interact: {err}")
        return " :moyai: :interrobang: "

    return reply.json()['status']

def get_opinions(channel, topic, condense=True):
    ''' Ask interact for its opinions on a topic in this channel. If summarize == True, merge them all. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
        "topic": topic,
        "summarize": condense
    }
    try:
        reply = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/opinion/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
        return []

    ret = reply.json()
    if 'opinions' in ret:
        return ret['opinions']

    return []
    # return [e for e in reply.json()['nouns'] if e not in speakers()]

def inject_idea(channel, idea):
    ''' Directly inject an idea into the stream of consciousness. '''
    req = {
        "service": SLACK_SERVICE,
        "channel": channel,
        "idea": idea
    }
    try:
        response = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/inject/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /inject/ to interact: {err}")
        return " :syringe: :interrobang: "

    return response.json()['status']

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
        return " :jigsaw: :interrobang: "

    return " :exploding_head: "

@app.message(re.compile(r"^help$", re.I))
def help_me(say, context): # pylint: disable=unused-argument
    ''' TODO: These should really be / commands. '''
    say(f"""*Commands:*
  `...`: Let {BOT_NAME} keep talking without interrupting
  `summary`: Explain it all to me very briefly.
  `status`: Say exactly what is on {BOT_NAME}'s mind.
  `nouns`: Some things worth thinking about.
  `reflect`: {BOT_NAME}'s opinion of those things.
  `daydream`: Let {BOT_NAME}'s mind wander on the convo.

  *Image generation:*
  :art: _prompt_ : Generate a picture of _prompt_ using stable-diffusion
  :magic_wand: _prompt_ : Generate a *fancy* picture of _prompt_ using stable-diffusion
  :selfie: Take a selfie
""")

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
        model=random.choice(["ffhq", "waifu"])
    )

@app.message(re.compile(r"^:art:$"))
def photo_stable_diffusion_summary(say, context): # pylint: disable=unused-argument
    ''' Take a stable diffusion photo of this conversation '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{BOT_NAME} takes out a shiny new camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=10,
        what=f"*click* _{BOT_NAME} shakes it like a polaroid picture_"
    )
    take_a_photo(channel, get_summary(channel, max_tokens=30), engine="stable-diffusion")

@app.message(re.compile(r"^:art:(.+)$"))
def stable_diffusion_picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture with stable diffusion '''
    speaker_id = context['user_id']
    speaker_name = get_display_name(speaker_id)
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    say(f"OK, {speaker_name}.\n_{BOT_NAME} takes out a shiny new camera and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=10,
        what=f"*{BOT_NAME} takes a picture of _{prompt}_*."
    )
    take_a_photo(channel, prompt, engine="stable-diffusion")

    ents = get_entities(prompt)
    if ents:
        inject_idea(channel, ents)
        msg = "..."
    else:
        msg = prompt

    the_reply = get_reply(channel, msg, speaker_name, speaker_id)

    say(the_reply)
    summarize_later(channel)

@app.message(re.compile(r"^:magic_wand:(.+)$"))
def prompt_parrot_picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture with stable diffusion '''
    speaker_id = context['user_id']
    speaker_name = get_display_name(speaker_id)
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    say(f"OK, {speaker_name}.\n_{BOT_NAME} takes out a magic wand and frames the scene_")
    say_something_later(
        say,
        channel,
        context,
        when=10,
        what=f"*{BOT_NAME} takes a picture of _{prompt}_*."
    )
    parrot = prompt_parrot(prompt)
    log.warning(f"🦜 {parrot}")
    take_a_photo(channel, prompt, engine="stable-diffusion", style=parrot)

    ents = get_entities(prompt)
    if ents:
        inject_idea(channel, ents)
        msg = "..."
    else:
        msg = prompt

    the_reply = get_reply(channel, msg, speaker_name, speaker_id)

    say(the_reply)
    summarize_later(channel)

@app.message(re.compile(r"^summary(\!)?$", re.I))
def summarize(say, context):
    ''' Say a condensed summary of this channel '''
    save = bool(context['matches'][0])
    channel = context['channel_id']
    say("💭 " + get_summary(channel, save, include_keywords=True, photo=True))

@app.message(re.compile(r"^status$", re.I))
def status(say, context):
    ''' Say a condensed summary of this channel '''
    channel = context['channel_id']
    say("\n".join([f"> {line}" for line in get_status(channel).split("\n")]))

@app.message(re.compile(r"^nouns$", re.I))
def nouns(say, context):
    ''' Say the nouns mentioned on this channel '''
    channel = context['channel_id']
    say("> " + ", ".join(get_nouns(get_status(channel))))

@app.message(re.compile(r"^entities$", re.I))
def entities(say, context):
    ''' Say the entities mentioned on this channel '''
    channel = context['channel_id']
    say("> " + ", ".join(get_entities(get_status(channel))))

@app.message(re.compile(r"^daydream$", re.I))
def daydream(say, context):
    ''' Let your mind wander '''
    channel = context['channel_id']
    say(f"_{BOT_NAME}'s mind starts to wander..._")

    ideas = get_daydream(channel)

    for idea in random.sample(list(ideas), min(len(ideas), 5)):
        # skip anyone speaking in the channel
        if idea in speakers():
            continue

        # skip eg. "4 months ago"
        if 'ago' in str(idea):
            continue

        inject_idea(channel, ideas[idea])
        say(f"💭 *{idea}*: _{ideas[idea]}_")

    for noun in random.sample(get_nouns(get_status(channel)), 8):
        opinion = get_opinions(channel, noun.lower(), condense=True)
        if not opinion:
            opinion = [judge(channel, noun.lower())]
        if opinion:
            inject_idea(channel, opinion[0])
            say(f"🤔 *{noun}*: _{opinion[0]}_")

    say(f"_{BOT_NAME} blinks and looks around._")
    summarize_later(channel, when=1)

@app.message(re.compile(r"^opinions (.*)$", re.I))
def opine_all(say, context):
    ''' Fetch our opinion on a topic '''
    topic = context['matches'][0]
    channel = context['channel_id']

    opinions = get_opinions(channel, topic, condense=False)
    if opinions:
        say('\n'.join(opinions))
    else:
        say('I have no opinion on that topic.')

@app.message(re.compile(r"^opinion (.*)$", re.I))
def opine(say, context):
    ''' Fetch our opinion on a topic '''
    topic = context['matches'][0]
    channel = context['channel_id']

    opinion = get_opinions(channel, topic, condense=True)
    if opinion:
        say(opinion[0])
    else:
        say('I have no opinion on that topic.')

def prompt_parrot(prompt):
    ''' Fetch a prompt from the parrot '''
    try:
        req = { "prompt": prompt }
        response = requests.post(f"{os.environ['PARROT_SERVER_URL']}/generate/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /generate/ to Prompt Parrot: {err}")
        return ""
    return response.json()['parrot']

def judge(channel, topic):
    ''' Form an opinion on topic '''
    try:
        req = { "service": SLACK_SERVICE, "channel": channel, "topic": topic }
        response = requests.post(f"{os.environ['INTERACT_SERVER_URL']}/judge/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /judge/ to interact: {err}")
        return ""
    return response.json()['opinion']

@app.message(re.compile(r"^reflect$", re.I))
def reflect(say, context):
    ''' Fetch our opinion on all the nouns in the channel '''
    channel = context['channel_id']

    for noun in get_nouns(get_status(channel)):
        opinion = get_opinions(channel, noun.lower(), condense=True)
        if not opinion:
            opinion = [judge(channel, noun.lower())]

        say(f"{noun}: {opinion[0]}")

@app.message(re.compile(r"^:bulb:$"))
def lights(say, context): # pylint: disable=unused-argument
    ''' Are the lights on? '''
    them = get_display_name(context['user_id'])

    if os.path.isfile('/home/rob/.u16-lights'):
        say(f'The lights are on, {them}.')
    else:
        say(f'The lights are off, {them}.')

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

def summarize_later(channel, when=None):
    '''
    Summarize the train of thought later. When is in seconds.

    Every time this thread executes, a new convo summary is saved. Only one
    can run at a time.
    '''
    if channel not in reminders:
        new_channel(channel)

    if not when:
        when = 120 + random.randint(20,80)

    reminders[channel]['summarizer'].cancel()
    reminders[channel]['summarizer'] = th.Timer(when, get_summary, [channel, True, True, 50, False, 0])
    reminders[channel]['summarizer'].start()

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
    msg = substitute_names(' '.join(context['matches'])).strip()

    if is_bot(speaker_id):
        log.warning(f'🤖 BOT DETECTED ({speaker_name})')
        # 95% chance to just ignore them
        if random.random() < 0.95:
            return

    the_reply = get_reply(channel, msg, speaker_name, speaker_id)

    say(the_reply)
    summarize_later(channel)

    if the_reply.endswith('…') or the_reply.endswith('...'):
        say_something_later(
            say,
            channel,
            context,
            when=1
        )
        return

    # 5% chance of random interjection later
    if random.random() < 0.05:
        say_something_later(
            say,
            channel,
            context,
            when=random.randint(2, 5)
        )


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
    if twitter is None:
        log.error("🐦 Twitter not configured, check your config.")
        return

    channel = body['event']['item']['channel']
    try:
        result = app.client.conversations_history(
            channel=channel,
            inclusive=True,
            oldest=body['event']['item']['ts'],
            limit=1
        )

        for msg in result.get('messages', []):
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
                    log.critical(f"🤖 Could not post /amnesia/ to interact: {err}")
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
                        if len(blk['alt_text']) > 277:
                            caption = blk['alt_text'][:277] + '...'
                        else:
                            caption = blk['alt_text']
                        with tempfile.TemporaryDirectory() as tmpdir:
                            media_ids = []
                            for blk in msg['blocks']:
                                response = requests.get(blk['image_url'])
                                response.raise_for_status()
                                fname = f"{tmpdir}/{blk['image_url'].split('/')[-1]}"
                                with open(fname, "wb") as f:
                                    for chunk in response.iter_content():
                                        f.write(chunk)
                                media = twitter.media_upload(fname)
                                media_ids.append(media.media_id)
                            twitter.update_status(caption, media_ids=media_ids)
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
        print()
        raise SystemExit(0) from kbderr
