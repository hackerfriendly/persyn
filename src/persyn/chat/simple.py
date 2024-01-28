'''
simple.py

Simple posting to chat services using webhooks.

These are needed since Slack and Discord both use complex callback schemes for their
standard libraries.

Used mostly by interact/autobus.py but can be called from anywhere.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import json
from typing import Optional

import requests
from persyn.chat.common import Chat
from persyn.chat.mastodon.bot import Mastodon

from persyn.utils.config import PersynConfig

# Color logging
from persyn.utils.color_logging import log

rs = requests.Session()
mastodon = None

def slack_msg(
    persyn_config: PersynConfig,
    chat: Chat,
    channel: str,
    msg: str,
    images: Optional[list[str]] = None,
    extra: Optional[str] = None # pylint: disable=unused-argument
    ) -> None:
    ''' Post a message to Slack with optional images '''

    log.debug(f"slack_msg: {persyn_config}, {chat}, {channel}, {msg}, {images}, {extra}")

    # TODO: Why does this call take ~three seconds to show up in the channel?
    bot_name = persyn_config.id.name
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
        reply = rs.post('https://slack.com/api/chat.postMessage', data=req, timeout=30)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"⚡️ Could not post image to Slack: {err}")

    if images:
        log.info(f"⚡️ Posted image to Slack as {bot_name}")
        chat.inject_idea(channel, chat.get_caption(url), verb='posts a picture')
    else:
        log.info(f"⚡️ Posted dialog to Slack as {bot_name}")


def discord_msg(
    persyn_config: PersynConfig,
    chat: Chat,
    channel: str,
    msg: str,
    images: Optional[list[str]] = None,
    extra: Optional[str] = None # pylint: disable=unused-argument
    ) -> None:
    ''' Post an image to Discord '''

    log.debug(f"discord_msg: {persyn_config}, {chat}, {channel}, {msg}, {images}, {extra}")

    bot_name = persyn_config.id.name
    req = {
        "username": persyn_config.id.name,
        # webhook is a different user id from the main bot, so set the avatar accordingly
        "avatar_url": getattr(persyn_config.id, "avatar", persyn_config.id.avatar)
    }

    if images:
        embeds = []
        url = ""
        for image in images:
            url = f"{persyn_config.dreams.upload.url_base}/{image}"
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
        reply = rs.post(persyn_config.chat.discord.webhook, json=req, timeout=30)
        reply.raise_for_status()
    except requests.exceptions.RequestException as err:
        log.critical(f"⚡️ Could not post image to Discord: {err}")

    if images:
        log.info(f"⚡️ Posted image to Discord as {bot_name}")
        chat.inject_idea(channel, chat.get_caption(url), verb='posts a picture')
    else:
        log.info(f"⚡️ Posted dialog to Discord as {bot_name}")


def mastodon_msg(
    persyn_config: PersynConfig,
    chat: Chat, # pylint: disable=unused-argument
    channel: str, # pylint: disable=unused-argument
    msg: str,
    images: Optional[list[str]] = None,
    extra: Optional[str] = None
    ) -> None:
    ''' Post a message to Mastodon with optional images '''
    global mastodon

    if mastodon is None:
        mastodon = Mastodon(persyn_config)
        mastodon.login()

    if extra is None:
        extra = '{}'

    if images:
        for image in images:
            mastodon.fetch_and_post_image(
                f"{persyn_config.dreams.upload.url_base}/{image}", f"{msg}\n#imagesynthesis #persyn", extra # type: ignore
            )
    else:
        mastodon.toot(msg, kwargs=json.loads(extra))
