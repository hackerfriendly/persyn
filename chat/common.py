'''
common.py

Subroutines common to all chat services
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import sys
import random

from pathlib import Path

import requests

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

# Color logging
from utils.color_logging import log

# Artist names
from utils.art import artists

# Bot config
from utils.config import load_config

CFG = load_config()

def get_reply(service, channel, msg, speaker_name, speaker_id):
    ''' Ask interact for an appropriate response. '''
    if msg != '...':
        log.info(f"[{channel}] {speaker_name}:", msg)

    req = {
        "service": service,
        "channel": channel,
        "msg": msg,
        "speaker_name": speaker_name,
        "speaker_id": speaker_id
    }
    try:
        response = requests.post(f"{CFG.interact.url}/reply/", params=req, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"🤖 Could not post /reply/ to interact: {err}")
        return (" :speech_balloon: :interrobang: ", [])

    resp = response.json()
    reply = resp['reply']
    goals_achieved = resp['goals_achieved']

    log.warning(f"[{channel}] {CFG.id.name}:", reply)
    if goals_achieved:
        log.warning(f"[{channel}] {CFG.id.name}:", f"🏆 {goals_achieved}")

    if any(verb in reply for verb in ['look', 'see', 'show', 'imagine', 'idea', 'memory', 'remember', 'watch', 'vision']):
        take_a_photo(channel, get_summary(service, channel, max_tokens=30), engine="stable-diffusion")

    return (reply, goals_achieved)

def get_summary(service, channel, save=False, photo=False, max_tokens=200, include_keywords=False, context_lines=0):
    ''' Ask interact for a channel summary. '''
    req = {
        "service": service,
        "channel": channel,
        "save": save,
        "max_tokens": max_tokens,
        "include_keywords": include_keywords,
        "context_lines": context_lines
    }
    try:
        reply = requests.post(f"{CFG.interact.url}/summary/", params=req, timeout=30)
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


def summarize_later(service, channel, reminders, when=None):
    '''
    Summarize the train of thought later. When is in seconds.

    Every time this thread executes, a new convo summary is saved. Only one
    can run at a time.
    '''
    if not when:
        when = 120 + random.randint(20,80)

    reminders.add(channel, when, get_summary, [service, channel, True, True, 50, False, 0], 'summarizer')

def take_a_photo(channel, prompt, engine=None, model=None, style=None):
    ''' Pick an image engine and generate a photo '''
    if not engine:
        engine = random.choice(CFG.dreams.all_engines)

    req = {
        "engine": engine,
        "channel": channel,
        "prompt": prompt,
        "model": model,
        "slack_bot_token": CFG.chat.slack.bot_token,
        "bot_name": CFG.id.name,
        "style": style
    }
    reply = requests.post(f"{CFG.dreams.url}/generate/", params=req, timeout=10)
    if reply.ok:
        log.warning(f"{CFG.dreams.url}/generate/", f"{prompt}: {reply.status_code}")
    else:
        log.error(f"{CFG.dreams.url}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
    return reply.ok
