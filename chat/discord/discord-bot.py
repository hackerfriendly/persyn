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
from interaction.reminders import AsyncReminders

# Common chat library
from chat.common import get_reply, take_a_photo, summarize_later

# Mastodon support for image posting
from chat.mastodon.login import mastodon

intents = discord.Intents.default()
intents.message_content = True # pylint: disable=assigning-non-slot

app = discord.Client(intents=intents)

CFG = load_config()

# Username cache
known_users = {}

# Known bots
known_bots = {}

# Coroutine reminders
reminders = AsyncReminders()

def say_something_later(ctx, when, what=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    reminders.cancel(ctx.channel.id)

    if what:
        reminders.add(ctx.channel.id, when, ctx.channel.send, what)
    else:
        # Yadda yadda yadda
        ctx.content = "..."
        reminders.add(ctx.channel.id, when, on_message, ctx)

@app.event
async def on_ready():
    ''' Ready player 0! '''
    log.info(f"Logged into chat.service: discord as {app.user} (guilds: {[g.name for g in app.guilds]})")

@app.event
async def on_message(ctx):
    ''' Default message handler. Prompt GPT and randomly arm a Timer for later reply. '''
    service = f"discord-{ctx.guild.id}"

    # Don't talk to yourself.
    if ctx.author == app.user:
        return

    # Interrupt any rejoinder in progress
    reminders.cancel(ctx.channel.id)

    if ctx.author.bot:
        log.warning(f'🤖 BOT DETECTED ({ctx.author.name})')
        # 95% chance to just ignore them
        if random.random() < 0.95:
            return

    if ctx.content == 'echo':
        say_something_later(ctx, when=0, what="echo echo echo...")
        return

    (the_reply, goals_achieved) = get_reply(service, ctx.channel.id, ctx.content, ctx.author.name, ctx.author.id)

    await ctx.channel.send(the_reply)

    for goal in goals_achieved:
        await ctx.channel.send(f"🏆 _achievement unlocked: {goal}_")

    summarize_later(service, ctx.channel.id, reminders, when=5)

    if the_reply.endswith('…') or the_reply.endswith('...'):
        say_something_later(
            ctx,
            when=1
        )
        return

    # 5% chance of random interjection later
    if random.random() < 0.05:
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
