#!/usr/bin/env python3
"""
slack.py


"""
# pylint: disable=import-error, wrong-import-position
import base64
import os
import random
import re
import sys
import tempfile

import threading as th

from pathlib import Path
from hashlib import sha256

import requests
import yaml

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from mastodon import Mastodon, MastodonError

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../../').resolve()))

# Color logging
from utils.color_logging import log

# Artist names
from utils.art import artists

# Bot config
from utils.config import load_config

if len(sys.argv) != 2:
    raise SystemExit("Usage: slack.py [config.yaml]")

CFG = load_config(sys.argv[1])

# import json
# raise SystemExit(json.dumps(CFG.dreams))

# Mastodon support for image posting
# mastodon = os.environ.get('MASTODON_INSTANCE', None)
# if mastodon:
#     masto_secret = Path(os.environ.get('MASTODON_SECRET', ''))
#     if not masto_secret.is_file():
#         raise RuntimeError(
#             f"Mastodon instance specified but secret file '{masto_secret}' does not exist.\nCheck your config."
#         )
#     try:
#         mastodon = Mastodon(
#             access_token = masto_secret,
#             api_base_url = mastodon
#         )
#     except MastodonError:
#         raise SystemExit("Invalid credentials, run masto-login.py and try again.") from MastodonError

mastodon = None

# Slack bolt App
app = App(token=CFG.chat.slack.bot_token)

CFG.chat.service = app.client.auth_test().data['url']
log.warning(f"Logged into chat service: {CFG.chat.service}")

raise SystemExit()
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
    return [CFG.id.name] + list(known_users.values())

def take_a_photo(channel, prompt, engine=None, model=None, style=None):
    ''' Pick an image engine and generate a photo '''
    if not engine:
        engine = random.choice(CFG.dreams.engines)

    req = {
        "engine": engine,
        "channel": channel,
        "prompt": prompt,
        "model": model or random.choice(IMAGE_MODELS[engine]),
        "slack_bot_token": CFG.chat.slack.bot_token,
        "bot_name": CFG.id.name,
        "style": style
    }
    reply = requests.post(f"{CFG.dreams.url}/generate/", params=req)
    if reply.ok:
        log.warning(f"{CFG.dreams.url}/generate/", f"{prompt}: {reply.status_code}")
    else:
        log.error(f"{CFG.dreams.url}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
    return reply.ok

def get_reply(channel, msg, speaker_name, speaker_id):
    ''' Ask interact for an appropriate response. '''
    if msg != '...':
        log.info(f"[{channel}] {speaker_name}: {msg}")

    req = {
        "service": CFG.chat.service,
        "channel": channel,
        "msg": msg,
        "speaker_name": speaker_name,
        "speaker_id": speaker_id
    }
    try:
        response = requests.post(f"{CFG.interact.url}/reply/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /reply/ to interact: {err}")
        return " :speech_balloon: :interrobang: "

    resp = response.json()
    reply = resp['reply']
    goals_achieved = resp['goals_achieved']

    log.warning(f"[{channel}] {CFG.id.name}: {reply}")
    if goals_achieved:
        log.warning(f"[{channel}] {CFG.id.name}: 🏆 {goals_achieved}")

    if any(verb in reply for verb in ['look', 'see', 'show', 'imagine', 'idea', 'memory', 'remember']):
        take_a_photo(channel, get_summary(channel, max_tokens=30), engine="stable-diffusion")

    return (reply, goals_achieved)

def get_summary(channel, save=False, photo=False, max_tokens=200, include_keywords=False, context_lines=0):
    ''' Ask interact for a channel summary. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel,
        "save": save,
        "max_tokens": max_tokens,
        "include_keywords": include_keywords,
        "context_lines": context_lines
    }
    try:
        reply = requests.post(f"{CFG.interact.url}/summary/", params=req)
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
        reply = requests.post(f"{CFG.interact.url}/nouns/", params=req)
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
        reply = requests.post(f"{CFG.interact.url}/entities/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /entities/ to interact: {err}")
        return []

    return [e for e in reply.json()['entities'] if e not in speakers()]

def get_daydream(channel):
    ''' Ask interact to daydream about this channel. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel,
    }
    try:
        reply = requests.post(f"{CFG.interact.url}/daydream/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /daydream/ to interact: {err}")
        return []

    return reply.json()['daydream']

def get_status(channel):
    ''' Ask interact for status. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel,
    }
    try:
        reply = requests.post(f"{CFG.interact.url}/status/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /status/ to interact: {err}")
        return " :moyai: :interrobang: "

    return reply.json()['status']

def get_opinions(channel, topic, condense=True):
    ''' Ask interact for its opinions on a topic in this channel. If summarize == True, merge them all. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel,
        "topic": topic,
        "summarize": condense
    }
    try:
        reply = requests.post(f"{CFG.interact.url}/opinion/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
        return []

    ret = reply.json()
    if 'opinions' in ret:
        return ret['opinions']

    return []
    # return [e for e in reply.json()['nouns'] if e not in speakers()]

def get_goals(channel):
    ''' Return the goals for this channel, if any. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel
    }
    try:
        reply = requests.post(f"{CFG.interact.url}/get_goals/", params=req)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /get_goals/ to interact: {err}")
        return []

    ret = reply.json()
    if 'goals' in ret:
        return ret['goals']

    return []
    # return [e for e in reply.json()['nouns'] if e not in speakers()]

def inject_idea(channel, idea):
    ''' Directly inject an idea into the stream of consciousness. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel,
        "idea": idea
    }
    try:
        response = requests.post(f"{CFG.interact.url}/inject/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /inject/ to interact: {err}")
        return " :syringe: :interrobang: "

    return response.json()['status']

def forget_it(channel):
    ''' There is no antimemetics division. '''
    req = {
        "service": CFG.chat.service,
        "channel": channel,
    }
    try:
        response = requests.post(f"{CFG.interact.url}/amnesia/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not forget_it(): {err}")
        return " :jigsaw: :interrobang: "

    return " :exploding_head: "

@app.message(re.compile(r"^help$", re.I))
def help_me(say, context): # pylint: disable=unused-argument
    ''' TODO: These should really be / commands. '''
    say(f"""*Commands:*
  `...`: Let {CFG.id.name} keep talking without interrupting
  `summary`: Explain it all to me very briefly.
  `status`: Say exactly what is on {CFG.id.name}'s mind.
  `nouns`: Some things worth thinking about.
  `reflect`: {CFG.id.name}'s opinion of those things.
  `daydream`: Let {CFG.id.name}'s mind wander on the convo.
  `goals`: See {CFG.id.name}'s current goals

  *Image generation:*
  :art: _prompt_ : Generate a picture of _prompt_ using stable-diffusion
  :magic_wand: _prompt_ : Generate a *fancy* picture of _prompt_ using stable-diffusion
  :selfie: Take a selfie
""")

@app.message(re.compile(r"^goals$"))
def goals(say, context): # pylint: disable=unused-argument
    ''' What are we doing again? '''
    channel = context['channel_id']

    current_goals = get_goals(channel)
    if current_goals:
        for goal in current_goals:
            say(f":goal_net: {goal}")
    else:
        say(":shrug:")

@app.message(re.compile(r"^:selfie:$"))
def selfie(say, context): # pylint: disable=unused-argument
    ''' Take a picture, it'll last longer '''
    them = get_display_name(context['user_id'])
    channel = context['channel_id']

    say(f"OK, {them}.\n_{CFG.id.name} takes out a camera and smiles awkwardly_.")
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
    channel = context['channel_id']
    take_a_photo(channel, get_summary(channel, max_tokens=30), engine="stable-diffusion")

@app.message(re.compile(r"^:art:(.+)$"))
def stable_diffusion_picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture with stable diffusion '''
    speaker_id = context['user_id']
    speaker_name = get_display_name(speaker_id)
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    take_a_photo(channel, prompt, engine="stable-diffusion")
    say(f"OK, {speaker_name}.")
    say_something_later(say, channel, context, 4, ":camera_with_flash:")
    ents = get_entities(prompt)
    if ents:
        inject_idea(channel, ents)

@app.message(re.compile(r"^:magic_wand:(.+)$"))
def prompt_parrot_picture(say, context): # pylint: disable=unused-argument
    ''' Take a picture with stable diffusion '''
    speaker_id = context['user_id']
    speaker_name = get_display_name(speaker_id)
    channel = context['channel_id']
    prompt = context['matches'][0].strip()

    parrot = prompt_parrot(prompt)
    log.warning(f"🦜 {parrot}")
    take_a_photo(channel, prompt, engine="stable-diffusion", style=parrot)
    say(f"OK, {speaker_name}.")
    say_something_later(say, channel, context, 4, ":camera_with_flash:")

    ents = get_entities(prompt)
    if ents:
        inject_idea(channel, ents)

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
    say(f"_{CFG.id.name}'s mind starts to wander..._")

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

    say(f"_{CFG.id.name} blinks and looks around._")
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
        response = requests.post(f"{CFG.dreams.parrot.url}/generate/", params=req)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /generate/ to Prompt Parrot: {err}")
        return prompt
    return response.json()['parrot']

def judge(channel, topic):
    ''' Form an opinion on topic '''
    try:
        req = { "service": CFG.chat.service, "channel": channel, "topic": topic }
        response = requests.post(f"{CFG.interact.url}/judge/", params=req)
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
    channel = context['channel_id']

    if os.path.isfile('/home/rob/.u16-lights'):
        say(f'The lights are on, {them}.')
        inject_idea(channel, "The lights are on.")
    else:
        say(f'The lights are off, {them}.')
        inject_idea(channel, "The lights are off.")

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

def get_caption(url):
    ''' Fetch the image caption using CLIP Interrogator '''
    log.warning("🖼  needs a caption")

    resp = requests.get(url, headers={'Authorization': f'Bearer {CFG.chat.slack.bot_token}'})
    if not resp.ok:
        log.error(f"🖼  Could not retrieve image: {resp.text}")
        return None

    resp = requests.post(
        f"{CFG.dreams.caption.url}/caption/",
        json={"data": base64.b64encode(resp.content).decode()}
    )
    if not resp.ok:
        log.error(f"🖼  Could not get_caption(): {resp.text}")
        return None

    caption = resp.json()['caption']
    log.warning(f"🖼  got caption: '{caption}'")
    return caption


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

    (the_reply, goals_achieved) = get_reply(channel, msg, speaker_name, speaker_id)

    say(the_reply)

    for goal in goals_achieved:
        say(f"🏆 _achievement unlocked: {goal}_")

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

    reply, goals_achieved = get_reply(channel, msg, speaker_name, speaker_id)

    say(reply)

    for goal in goals_achieved:
        say(f"🏆 _achievement unlocked: {goal}_")

@app.event("reaction_added")
def handle_reaction_added_events(body, logger): # pylint: disable=unused-argument
    '''
    Handle reactions: post images to Mastodon.
    '''
    if mastodon is None:
        log.error("🎺 Mastodon is not configured, check your config.")
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
                        log.warning("🎺 Not posting:", {msg['reactions'][0]['name']})
                        return
                    log.warning("🤯 All is forgotten.")
                    forget_it(channel)
                    return
                try:
                    req = { "service": CFG.chat.service, "channel": channel }
                    response = requests.post(f"{CFG.interact.url}/amnesia/", params=req)
                    response.raise_for_status()
                except requests.exceptions.RequestException as err:
                    log.critical(f"🤖 Could not post /amnesia/ to interact: {err}")
                    return

                if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                    blk = msg['blocks'][0]
                    try:
                        if len(blk['alt_text']) > 497:
                            toot = blk['alt_text'][:497] + '...'
                        else:
                            toot = blk['alt_text']
                        with tempfile.TemporaryDirectory() as tmpdir:
                            media_ids = []
                            for blk in msg['blocks']:
                                response = requests.get(blk['image_url'])
                                response.raise_for_status()
                                fname = f"{tmpdir}/{blk['image_url'].split('/')[-1]}"
                                with open(fname, "wb") as f:
                                    for chunk in response.iter_content():
                                        f.write(chunk)
                                caption = get_caption(blk['image_url'])
                                media_ids.append(mastodon.media_post(fname, description=caption).id)

                            resp = mastodon.status_post(
                                toot,
                                media_ids=media_ids,
                                idempotency_key=sha256(blk['image_url'].encode()).hexdigest()
                            )
                            if not resp or 'url' not in resp:
                                raise MastodonError(resp)
                            log.info(f"🎺 Posted {blk['image_url']}: {resp['url']}")

                    except MastodonError as err:
                        log.error(f"🎺 Could not post {blk['image_url']}: {err}")
                else:
                    log.error(f"🎺 Unhandled reaction {msg['reactions'][0]['name']} to: {msg['text']}")

    except SlackApiError as err:
        log.error(f"Slack error: {err}")

@app.event("reaction_removed")
def handle_reaction_removed_events(body, logger): # pylint: disable=unused-argument
    ''' Skip for now '''
    log.info("Reaction removed event")

# @app.command("/echo")
# def repeat_text(ack, respond, command):
#     # Acknowledge command request
#     ack()
#     respond(f"_{command['text']}_")

@app.event("message")
def handle_message_events(body, say):
    ''' Handle uploaded images '''
    channel = body['event']['channel']

    if 'user' not in body['event']:
        log.warning("Message event with no user. 🤷")
        return

    speaker_id = body['event']['user']
    speaker_name = get_display_name(speaker_id)
    msg = substitute_names(body['event']['text'])

    if channel not in reminders:
        new_channel(channel)

    if 'files' not in body['event']:
        log.warning("Message with no picture? 🤷‍♂️")
        return

    for file in body['event']['files']:
        caption = get_caption(file['url_private_download'])

        if caption:
            prefix = random.choice(["I see", "It looks like", "Looks like", "Might be", "I think it's"])
            say(f"{prefix} {caption}")

            inject_idea(channel, f"{speaker_name} posted a photo of {caption}")

            if not msg.strip():
                msg = f"{speaker_name} posted a photo of {caption}"

            reply, goals_achieved = get_reply(channel, msg, speaker_name, speaker_id)

            say(reply)

            for goal in goals_achieved:
                say(f"🏆 _achievement unlocked: {goal}_")
        else:
            say(
                random.choice([
                    "I'm not sure.",
                    ":face_with_monocle:",
                    ":face_with_spiral_eyes:",
                    "What the...?",
                    "Um.",
                    "No idea.",
                    "Beats me."
                ])
            )

if __name__ == "__main__":
    handler = SocketModeHandler(app, CFG.chat.slack.app_token)
    try:
        handler.start()
    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        print()
        raise SystemExit(0) from kbderr
