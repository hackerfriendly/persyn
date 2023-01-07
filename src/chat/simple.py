'''
simple.py

Simple posting to chat services using webhooks.

These are needed since Slack and Discord both use complex callback schemes for their
standard libraries.

Used mostly by interact/cns.py but can be called from anywhere.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import json

import requests

# Color logging
from utils.color_logging import log

def slack_msg(persyn_config, chat, channel, bot_name, msg, images=None):
    ''' Post a message to Slack with optional images '''

    blocks = []
    if images:
        # Posting multiple images in a single block doesn't seem to be possible from a bot. Hmm.
        url = ""
        for i, image in enumerate(images):
            url = f"{persyn_config.dreams.upload.url_base}/{image}"
            blocks.append(
                {
                    "type": "image",
                    "title": {
                        "type": "plain_text",
                        "text": msg if i == 0 else " "
                    },
                    "image_url" : url,
                    "alt_text": msg
                }
            )

    req = {
        "token": persyn_config.chat.slack.bot_token,
        "channel": channel,
        "username": bot_name,
        "text": msg
    }

    if blocks:
        req['blocks'] = json.dumps(blocks)

    try:
        reply = requests.post('https://slack.com/api/chat.postMessage', data=req, timeout=30)
        reply.raise_for_status()
        log.info(f"⚡️ Posted image to Slack as {bot_name}")
    except requests.exceptions.RequestException as err:
        log.critical(f"⚡️ Could not post image to Slack: {err}")

    chat.inject_idea(channel, f"{persyn_config.id.name} posted a photo of {chat.get_caption(url)}")

def discord_msg(persyn_config, chat, channel, bot_name, msg, images=None):
    ''' Post an image to Discord '''
    req = {
        "username": persyn_config.id.name,
        # webhook is a different user id from the main bot, so set the avatar accordingly
        "avatar_url": getattr(persyn_config.id, "avatar", "https://hackerfriendly.com/pub/anna/anna.png")
    }

    if images:
        embeds = []
        url = ""
        for image in images:
            url =  f"{persyn_config.dreams.upload.url_base}/{image}"
            embeds.append(
                {
                    "description": msg or "Untitled",
                    "image": {
                        "url": url
                    }
                }
            )

        req['embeds'] = embeds
    else:
        req['content'] = msg

    try:
        reply = requests.post(persyn_config.chat.discord.webhook, json=req, timeout=30)
        reply.raise_for_status()
        log.info(f"⚡️ Posted image to Discord as {bot_name}")
    except requests.exceptions.RequestException as err:
        log.critical(f"⚡️ Could not post image to Discord: {err}")

    chat.inject_idea(channel, f"{persyn_config.id.name} posted a photo of {chat.get_caption(url)}")
