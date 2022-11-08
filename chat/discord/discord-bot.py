#!/usr/bin/env python3
"""
discord-bot.py

Chat with your persyn on Discord.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import base64
import os
import random
import re
import sys
import tempfile
import asyncio

from pathlib import Path
from hashlib import sha256

import requests

# discord.py
import discord

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../../').resolve()))

# Color logging
from utils.color_logging import log

# Artist names
from utils.art import artists

# Bot config
from utils.config import load_config

# Reminders
from interaction.reminders import async_reminders as reminders

intents = discord.Intents.default()
intents.message_content = True # pylint: disable=assigning-non-slot

app = discord.Client(intents=intents)

CFG = load_config()

# Username cache
known_users = {}

# Known bots
known_bots = {}

def get_reply(guild, channel, msg, speaker_name, speaker_id):
    ''' Ask interact for an appropriate response. '''

    if msg != '...':
        log.info(f"[{channel}] {speaker_name}:", msg)

    req = {
        "service": f"discord-{guild}",
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

    # if any(verb in reply for verb in ['look', 'see', 'show', 'imagine', 'idea', 'memory', 'remember']):
    #     take_a_photo(channel, get_summary(channel, max_tokens=30), engine="stable-diffusion")

    return (reply, goals_achieved)


def say_something_later(ctx, when, what=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    reminders.cancel(ctx.channel.id)

    # response = requests.get("http://whatismyip.akamai.com/")
    # print(response.text)

    if what:
        reminders.add(ctx.channel.id, when, ctx.channel.send, [what])
    else:
        # Yadda yadda yadda
        ctx.content = "..."
        reminders.add(ctx.channel.id, when, on_message, ctx)

@app.event
async def on_ready():
    ''' on_ready '''
    log.info(f'{app.user} » O N L I N E «')

@app.event
async def on_message(ctx):
    ''' Default message handler. Prompt GPT and randomly arm a Timer for later reply. '''

    # Don't talk to yourself.
    log.warning("AUTHOR:", f"{ctx.author} == {app.user}")
    if ctx.author == app.user:
        log.warning(f"🤖 Ignoring my own message: {ctx.content}")
        return

    # Interrupt any rejoinder in progress
    reminders.cancel(ctx.channel.id)

    if ctx.author.bot:
        log.warning(f'🤖 BOT DETECTED ({ctx.author.name})')
        # 95% chance to just ignore them
        if random.random() < 0.95:
            return

    (the_reply, goals_achieved) = get_reply(ctx.guild.id, ctx.channel.id, ctx.content, ctx.author.name, ctx.author.id)

    await ctx.channel.send(the_reply)

    for goal in goals_achieved:
        await ctx.channel.send(f"🏆 _achievement unlocked: {goal}_")

    # summarize_later(channel)

    if the_reply.endswith('…') or the_reply.endswith('...'):
        say_something_later(
            ctx,
            when=1
        )
        return

    # 5% chance of random interjection later
    if random.random() < 10.05:
        say_something_later(
            ctx,
            when=random.randint(2, 5)
        )

@app.event
async def on_raw_reaction_add(ctx):
    ''' on_raw_reaction_add '''
    channel = await app.fetch_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id)

    log.info(f'Reaction added: {ctx.member} : {ctx.emoji} ({message.content})')

@app.event
async def on_raw_reaction_remove(ctx):
    ''' on_raw_reaction_remove '''
    channel = await app.fetch_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id)

    log.info(f'Reaction removed: {ctx.member} : {ctx.emoji} ({message.content})')

# @app.event
# async def on_error(event, *args, **kwargs):
#     ''' on_error '''
#     log.critical(f'ERROR: {event}')
#     log.critical(f'args: {args}')
#     log.critical(f'kwargs: {kwargs}')

app.run(CFG.chat.discord.token)
