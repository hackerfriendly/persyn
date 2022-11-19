#!/usr/bin/env python3
"""
discord-bot.py

Chat with your persyn on Discord.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import random
import sys
import uuid

from pathlib import Path

# discord.py
import discord

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../../').resolve()))

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# Reminders
from interaction.reminders import AsyncReminders

# Mastodon support for image posting
from chat.mastodon.login import mastodon

# Common chat library
from chat.common import Chat

intents = discord.Intents.default()
intents.message_content = True # pylint: disable=assigning-non-slot

app = discord.Client(intents=intents)

persyn_config = load_config()

# Chat library
chat = Chat(persyn_config, service='discord')

# Coroutine reminders
reminders = AsyncReminders()

def get_channel(ctx):
    ''' Return the unique identifier for this guild+channel '''
    return f"{ctx.guild.id}|{ctx.channel.id}"

def say_something_later(ctx, when, what=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    channel = get_channel(ctx)
    reminders.cancel(channel)

    if what:
        reminders.add(channel, when, ctx.channel.send, args=what)
    else:
        # Yadda yadda yadda
        ctx.content = "..."
        reminders.add(channel, when, on_message, args=ctx)

def synthesize_image(ctx, prompt, engine="stable-diffusion", style=None):
    ''' It's not AI art. It's _image synthesis_ '''
    channel = get_channel(ctx)
    chat.take_a_photo(channel, prompt, engine=engine, style=style)
    say_something_later(ctx, when=4, what=":camera_with_flash:")

    ents = chat.get_entities(prompt)
    if ents:
        chat.inject_idea(channel, ents)

async def schedule_reply(ctx):
    ''' Gather a reply and say it when ready '''
    channel = get_channel(ctx)

    log.warning("⏰ schedule_reply:", ctx)

    (the_reply, goals_achieved) = chat.get_reply(channel, ctx.content, ctx.author.name, ctx.author.id)

    await ctx.channel.send(the_reply)

    for goal in goals_achieved:
        await ctx.channel.send(f"🏆 _achievement unlocked: {goal}_")

    chat.summarize_later(channel, reminders)

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

async def dispatch(ctx):
    ''' Handle commands '''
    channel = get_channel(ctx)

    if ctx.content.startswith('🎨'):
        await ctx.channel.send(f"OK, {ctx.author.name}.")
        synthesize_image(ctx, ctx.content[1:].strip(), engine="stable-diffusion")

    elif ctx.content.startswith('🪄'):
        await ctx.channel.send(f"OK, {ctx.author.name}.")
        prompt = ctx.content[1:].strip()
        style = chat.prompt_parrot(prompt)
        log.warning(f"🦜 {style}")
        synthesize_image(ctx, prompt, engine="stable-diffusion", style=style)

    elif ctx.content == '🤳':
        await ctx.channel.send(
            f"OK, {ctx.author.name}.\n_{persyn_config.id.name} takes out a camera and smiles awkwardly_."
        )
        say_something_later(
            ctx,
            when=9,
            what=":cheese_wedge: *CHEESE!* :cheese_wedge:"
        )
        chat.take_a_photo(
            get_channel(ctx),
            f"A selfie for {ctx.author.name}",
            engine="stylegan2",
            model=random.choice(["ffhq", "waifu"])
        )

    elif ctx.content == 'help':
        await ctx.channel.send(f"""*Commands:*
  `...`: Let {persyn_config.id.name} keep talking without interrupting
  `summary`: Explain it all to me very briefly.
  `status`: Say exactly what is on {persyn_config.id.name}'s mind.
  `nouns`: Some things worth thinking about.
  `reflect`: {persyn_config.id.name}'s opinion of those things.
  `daydream`: Let {persyn_config.id.name}'s mind wander on the convo.
  `goals`: See {persyn_config.id.name}'s current goals

  *Image generation:*
  :art: _prompt_ : Generate a picture of _prompt_ using stable-diffusion
  :magic_wand: _prompt_ : Generate a *fancy* picture of _prompt_ using stable-diffusion
  :selfie: Take a selfie
""")

    elif ctx.content == 'status':
        status = ("\n".join([f"> {line.strip()}" for line in chat.get_status(channel).split("\n")])).rstrip("> \n")
        if len(status) < 2000:
            await ctx.channel.send(status.strip())
        else:
            # 2000 character limit for messages
            reply = ""
            for line in status.split("\n"):
                if len(reply) + len(line) < 1999:
                    reply = reply + line + "\n"
                else:
                    await ctx.channel.send(reply)
                    reply = line + "\n"
            if reply:
                await ctx.channel.send(reply)

    elif ctx.content == 'summary':
        await ctx.channel.send("💭 " + chat.get_summary(channel, save=False, include_keywords=False, photo=True))

    elif ctx.content == 'summary!':
        await ctx.channel.send("💭 " + chat.get_summary(channel, save=True, include_keywords=True, photo=False))

    elif ctx.content == 'nouns':
        await ctx.channel.send("> " + ", ".join(chat.get_nouns(chat.get_status(channel))))

    else:
        reminders.add(channel, 0, schedule_reply, f'reply-{uuid.uuid4()}', args=[ctx])

@app.event
async def on_ready():
    ''' Ready player 0! '''
    log.info(f"Logged into chat.service: discord as {app.user} (guilds: {[g.name for g in app.guilds]})")

@app.event
async def on_message(ctx):
    ''' Default message handler. '''
    channel = get_channel(ctx)

    # Don't talk to yourself.
    if ctx.author.id in [app.user.id, persyn_config.chat.discord.webhook_id]:
        return

    # Interrupt any rejoinder in progress
    reminders.cancel(channel)

    if ctx.author.bot:
        log.warning(f'🤖 BOT DETECTED: {ctx.author.name} ({ctx.author.id})')
        # 95% chance to just ignore them
        if random.random() < 0.95:
            return

    await dispatch(ctx)

    # Any reply is scheduled

@app.event
async def on_raw_reaction_add(ctx):
    ''' on_raw_reaction_add '''
    channel = await app.fetch_channel(ctx.channel.id)
    message = await channel.fetch_message(ctx.message_id)

    log.info(f'Reaction added: {ctx.member} : {ctx.emoji} ({message.content})')

@app.event
async def on_raw_reaction_remove(ctx):
    ''' on_raw_reaction_remove '''
    channel = await app.fetch_channel(ctx.channel.id)
    message = await channel.fetch_message(ctx.message_id)

    log.info(f'Reaction removed: {ctx.member} : {ctx.emoji} ({message.content})')

# @app.event
# async def on_error(event, *args, **kwargs):
#     ''' on_error '''
#     log.critical(f'ERROR: {event}')
#     log.critical(f'args: {args}')
#     log.critical(f'kwargs: {kwargs}')

app.run(persyn_config.chat.discord.token)
