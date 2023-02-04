#!/usr/bin/env python3
'''
cns-autobus.py

The central nervous system. Listen for events and inject them into interact. Uses Redis instead of Boto.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import os
import argparse

import autobus

# Common chat library
from chat.common import Chat
from chat.simple import slack_msg, discord_msg

# Mastodon support for image posting
from chat.mastodon.bot import Mastodon

# Message classes
from interaction.messages import SendChat, Idea, Summarize

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# Defined in main()
mastodon = None
persyn_config = None

def mastodon_msg(_, chat, channel, bot_name, caption, images):  # pylint: disable=unused-argument
    ''' Post images to Mastodon '''
    for image in images:
        mastodon.fetch_and_post_image(
            f"{persyn_config.dreams.upload.url_base}/{image}", f"{caption}\n#imagesynthesis #persyn"
        )

services = {
    'slack': slack_msg,
    'discord': discord_msg,
    'mastodon': mastodon_msg
}

def get_service(svc):
    ''' Find the correct service for the dispatcher '''
    if 'slack.com' in svc:
        return 'slack'
    if svc in services:
        return svc

    log.critical(f"Unknown service: {svc}")
    return None

def say_something(event):
    ''' Send a message to a service + channel '''
    chat = Chat(
        bot_name=event.bot_name,
        bot_id=event.bot_id,
        service=event.service,
        interact_url=persyn_config.interact.url,
        dreams_url=persyn_config.dreams.url,
        captions_url=persyn_config.dreams.captions.url,
        parrot_url=persyn_config.dreams.parrot.url
    )
    services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, event.msg, event.images)

def new_idea(event):
    ''' Inject a new idea '''
    chat = Chat(
        bot_name=event.bot_name,
        bot_id=event.bot_id,
        service=event.service,
        interact_url=persyn_config.interact.url,
        dreams_url=persyn_config.dreams.url,
        captions_url=persyn_config.dreams.captions.url,
        parrot_url=persyn_config.dreams.parrot.url
    )
    chat.inject_idea(
        channel=event.channel,
        idea=event.idea,
        verb=event.verb
    )

def summarize_channel(event):
    ''' Summarize the channel '''
    chat = Chat(
        bot_name=event.bot_name,
        bot_id=event.bot_id,
        service=event.service,
        interact_url=persyn_config.interact.url,
        dreams_url=persyn_config.dreams.url,
        captions_url=persyn_config.dreams.captions.url,
        parrot_url=persyn_config.dreams.parrot.url
    )
    summary = chat.get_summary(
        channel=event.channel,
        save=True,
        photo=event.photo,
        max_tokens=event.max_tokens
    )
    services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, summary)


@autobus.subscribe(SendChat)
def chat_event(event):
    ''' Dispatch chat event w/ optional images. '''
    if event.bot_id == persyn_config.id.guid:
        say_something(event)
    else:
        log.debug(f"⚡️ send_chat(): ignoring message for {event.bot_id}", f"({event.bot_name})")

@autobus.subscribe(Idea)
def idea_event(event):
    ''' Dispatch idea event. '''
    if event.bot_id == persyn_config.id.guid:
        new_idea(event)
    else:
        log.debug(f"⚡️ inject_idea(): ignoring message for {event.bot_id}", f"({event.bot_name})")

@autobus.subscribe(Summarize)
def summarize_event(event):
    ''' Dispatch summarize event. '''
    if event.bot_id == persyn_config.id.guid:
        summarize_channel(event)
    else:
        log.debug(f"⚡️ summarize_channel(): ignoring message for {event.bot_id}", f"({event.bot_name})")


def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Persyn central nervous system. Run one server for each bot.'''
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

    if not hasattr(persyn_config, 'cns'):
        raise SystemExit('cns not defined in config, exiting.')

    global mastodon
    mastodon = Mastodon(args.config_file)
    mastodon.login()

    log.info(f"⚡️ {persyn_config.id.name}'s CNS is online")

    try:
        autobus.run(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)

    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        print()
        raise SystemExit(0) from kbderr

if __name__ == '__main__':
    main()
