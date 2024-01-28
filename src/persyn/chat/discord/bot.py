#!/usr/bin/env python3
"""
discord/bot.py

Chat with your persyn on Discord.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import logging
import os
import random
import tempfile
import uuid

from hashlib import sha256

# discord.py
import discord

import requests

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

# Reminders
from persyn.interaction.reminders import AsyncReminders

# Mastodon support for image posting
from persyn.chat.mastodon.bot import Mastodon

# Common chat library
from persyn.chat.common import Chat

# Coroutine reminders
reminders = AsyncReminders()

# Requests session
rs = requests.Session()

# Defined in main()
app = None
chat = None
persyn_config = None
mastodon = None

intents = discord.Intents.default()
intents.message_content = True # pylint: disable=assigning-non-slot
app = discord.Client(intents=intents)

# TODO: Fix for new persyn chat! Discord channels are fubar!

@app.event
async def on_ready():
    ''' Ready player 0! '''
    log.info(f"Logged into chat.service: discord as {app.user} (guilds: {[g.name for g in app.guilds]})")

@app.event
async def on_message(ctx):
    ''' Default message handler. '''
    channel = get_channel(ctx)

    # Don't talk to yourself.
    if it_me(ctx.author.id):
        return

    # Interrupt any rejoinder in progress
    reminders.cancel(channel)

    if ctx.author.bot:
        log.warning(f'🤖 BOT DETECTED: {ctx.author.name} ({ctx.author.id})')
        # 95% chance to just ignore them
        if random.random() < 0.95:
            return

    # Handle commands and schedule a reply (if any)
    await dispatch(ctx)

@app.event
async def on_raw_reaction_add(ctx):
    ''' on_raw_reaction_add '''
    channel = await app.fetch_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id) # type: ignore

    if not it_me(message.author.id):
        log.warning("👎 Not posting image that isn't mine.")
        return

    for embed in message.embeds:
        fetch_and_post_to_masto(embed.image.url, embed.description)

        # log.critical(embed.image.url)

    # if len(message.embeds) > 0:
    #     log.critical(message.embeds[0])


    log.info(f'Reaction added: {ctx.member} : {ctx.emoji} ({message.id})')

@app.event
async def on_raw_reaction_remove(ctx):
    ''' on_raw_reaction_remove '''
    channel = await app.fetch_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id) # type: ignore

    log.info(f'Reaction removed: {ctx.member} : {ctx.emoji} ({message.id})')

# @app.event
# async def on_error(event, *args, **kwargs):
#     ''' on_error '''
#     log.critical(f'ERROR: {event}')
#     log.critical(f'args: {args}')
#     log.critical(f'kwargs: {kwargs}')

def it_me(author_id):
    ''' Return True if the given id is one of ours '''
    return author_id in [app.user.id, persyn_config.chat.discord.webhook_id] # type: ignore

def get_channel(ctx):
    ''' Return the unique identifier for this guild/channel or DM '''
    # Note: Don't use | as a separator here as it confuses memory.py
    if getattr(ctx, 'guild'):
        return f"{ctx.guild.id}/{ctx.channel.id}"
    return f"dm/{ctx.author.id}/{ctx.channel.id}"

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

def synthesize_image(ctx, prompt, engine="dall-e", width=None, height=None, style=None):
    ''' It's not AI art. It's _image synthesis_ '''
    channel = get_channel(ctx)

    chat.take_a_photo( # type: ignore
        channel,
        prompt,
        engine=engine,
        width=width,
        height=height,
        style=style
    )
    say_something_later(ctx, when=3, what=":camera_with_flash:")

    ents = chat.get_entities(prompt)
    if ents:
        chat.inject_idea(channel, ents)

def fetch_and_post_to_masto(url, toot):
    ''' Download the image at URL and post it to Mastodon '''
    if not mastodon.client:
        log.error("🎺 Mastodon not configured, check your yaml config.")
        return

    media_ids = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            response = rs.get(url, timeout=10)
            response.raise_for_status()
            fname = f"{tmpdir}/{uuid.uuid4()}.{url[-3:]}"
            with open(fname, "wb") as f:
                for chunk in response.iter_content():
                    f.write(chunk)
            caption = chat.get_caption(url)
            media_ids.append(mastodon.client.media_post(fname, description=caption).id)

            resp = mastodon.client.status_post(
                toot,
                media_ids=media_ids,
                visibility='unlisted',
                idempotency_key=sha256(url.encode()).hexdigest()
            )
            if not resp or 'url' not in resp:
                raise RuntimeError(resp)
            log.info(f"🎺 Posted {url}: {resp['url']}")

    except RuntimeError as err:
        log.error(f"🎺 Could not post {url}: {err}")


async def schedule_reply(ctx):
    ''' Gather a reply and say it when ready '''
    channel = get_channel(ctx)

    log.warning("⏰ schedule_reply():", ctx.content)

    # Dispatch a "message received" event. Replies are handled by CNS.
    chat.chat_received(channel, ctx.content, ctx.author.name)

    # Webhooks in discord are per-channel. Skip summarizing DMs since it would bleed over.
    # if not channel.startswith('dm|'):
    #     chat.summarize_later(channel, reminders)


async def handle_attachments(ctx):
    ''' Caption photos posted to the channel '''
    channel = get_channel(ctx)
    for attachment in ctx.attachments:
        caption = chat.get_caption(attachment.url)

        if caption:
            await ctx.channel.send(caption)

            chat.inject_idea(channel, caption)

            msg = ctx.content
            if not msg.strip():
                msg = "..."

            chat.get_reply(channel, msg, ctx.author.name, send_chat=True)

        else:
            await ctx.channel.send(
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

async def dispatch(ctx):
    ''' Handle commands '''
    channel = get_channel(ctx)

    if ctx.attachments:
        await handle_attachments(ctx)

    elif ctx.content.startswith('🎨'):
        await ctx.channel.send(f"OK, {ctx.author.name}.")
        synthesize_image(ctx, ctx.content[1:].strip(), engine="dall-e")

    elif ctx.content.startswith('🖼'):
        await ctx.channel.send(f"OK, {ctx.author.name}.")
        synthesize_image(ctx, ctx.content[1:].strip(), engine="dall-e", width=1024, height=1792)

    elif ctx.content == 'help':
        await ctx.channel.send(f"""*Commands:*
  `...`: Let {persyn_config.id.name} keep talking without interrupting
  `summary`: Explain it all to me very briefly.
  `status`: Say exactly what is on {persyn_config.id.name}'s mind.
  `nouns`: Some things worth thinking about.
  `reflect`: {persyn_config.id.name}'s opinion of those things.
  `goals`: See {persyn_config.id.name}'s current goals

  *Image generation:*
  :art: _prompt_ : Generate a picture of _prompt_ using dall-e v3
  :frame_with_picture: _prompt_ : Generate a portrait of _prompt_ using dall-e v3
""")

    elif ctx.content in ['status', ':question:', '❓']:
        status = ("\n".join([f"> {line.strip()}" for line in chat.get_status(channel, ctx.author.name).split("\n")])).rstrip("> \n")
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
        await ctx.channel.send("💭 " + chat.get_summary(channel, photo=True))

    elif ctx.content == 'summary!':
        await ctx.channel.send("💭 " + chat.get_summary(channel, photo=False))

    elif ctx.content == 'nouns':
        await ctx.channel.send("> " + ", ".join(chat.get_nouns(chat.get_status(channel, ctx.author.name))))

    else:
        reminders.add(channel, 0, schedule_reply, f'reply-{uuid.uuid4()}', args=[ctx])


def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Discord chat module for Persyn'''
    )
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    # parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    global persyn_config
    persyn_config = load_config(args.config_file)

    # enable logging to disk
    if hasattr(persyn_config.id, "logdir"):
        logging.getLogger().addHandler(logging.FileHandler(f"{persyn_config.id.logdir}/{persyn_config.id.name}-discord.log"))

    # Mastodon support
    global mastodon
    mastodon = Mastodon(persyn_config)
    mastodon.login()

    # Chat library
    global chat
    chat = Chat(persyn_config=persyn_config, service='discord')

    # Discord client
    app.run(persyn_config.chat.discord.token)

if __name__ == '__main__':
    main()
